use burn::tensor::{Device, Tensor};
use image::RgbImage;

use crate::db_postprocess::DbPostProcess;
use crate::processors::{CtcRecognizer, DbDetector, DetModel, RecModel};
use crate::{Det, Detection, Rec, Recognition};

mod det_model {
    include!(concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_det.rs"));
}

mod rec_model {
    include!(concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_rec.rs"));
}

const DET_WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_det.bpk");
const REC_WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_rec.bpk");
const DICTIONARY: &str = include_str!("../../models/PP-OCRv6_tiny_dict.txt");

impl DetModel for det_model::Model {
    fn forward(&self, images: Tensor<4>) -> Tensor<4> {
        self.forward(images)
    }
}

impl RecModel for rec_model::Model {
    fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        self.forward(images)
    }
}

pub struct PpOcrV6TinyDet(DbDetector<det_model::Model>);

impl PpOcrV6TinyDet {
    pub fn new(device: Device) -> Self {
        let model = det_model::Model::from_file(DET_WEIGHTS_FILE, &device);
        Self(DbDetector::new(model, device, DbPostProcess::default()))
    }
}

impl Det for PpOcrV6TinyDet {
    fn forward(&self, image: &RgbImage) -> Vec<Detection> {
        self.0.forward(image)
    }
}

pub struct PpOcrV6TinyRec(CtcRecognizer<rec_model::Model>);

impl PpOcrV6TinyRec {
    pub fn new(device: Device) -> Self {
        let model = rec_model::Model::from_file(REC_WEIGHTS_FILE, &device);
        Self(CtcRecognizer::new(model, device, DICTIONARY))
    }
}

impl Rec for PpOcrV6TinyRec {
    fn forward(&self, crops: &[RgbImage]) -> Vec<Recognition> {
        self.0.forward(crops)
    }
}
