use burn::tensor::Device;

use crate::OcrProcessor;
use crate::crop::TextCropperConfig;
use crate::det::DbPostProcessConfig;
use crate::det::preprocess::DetPreProcessConfig;
use crate::rec::preprocess::RecPreProcessConfig;
use crate::rec::{CtcDecoderConfig, RecBatcherConfig};

pub mod det_model {
    use crate::det::DetModel;

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_medium/det.rs"));

    #[cfg(not(feature = "embed"))]
    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_medium/det.bpk");

    impl DetModel for Model {
        #[cfg_attr(
            feature = "profiling",
            tracing::instrument(skip_all, name = "det.model")
        )]
        fn forward(&self, images: Tensor<4>) -> Tensor<4> {
            self.forward(images)
        }
    }

    impl Model {
        pub fn load(device: &Device) -> Self {
            cfg_select! {
                feature = "embed" => Self::from_embedded(device),
                _ => Self::from_file(WEIGHTS_FILE, device),
            }
        }
    }
}

pub mod rec_model {
    use crate::rec::{CtcDecoderConfig, RecModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_medium/rec.rs"));

    #[cfg(not(feature = "embed"))]
    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_medium/rec.bpk");
    const DICTIONARY: &str = include_str!("../../models/pp_ocrv6_medium/dict.txt");

    impl RecModel for Model {
        #[cfg_attr(
            feature = "profiling",
            tracing::instrument(skip_all, name = "rec.model")
        )]
        fn forward(&self, images: Tensor<4>) -> Tensor<3> {
            self.forward(images)
        }
    }

    impl Model {
        pub fn load(device: &Device) -> Self {
            cfg_select! {
                feature = "embed" => Self::from_embedded(device),
                _ => Self::from_file(WEIGHTS_FILE, device),
            }
        }
    }

    impl CtcDecoderConfig {
        pub fn pp_ocrv6_medium() -> Self {
            let mut characters = vec![String::new()];
            characters.extend(DICTIONARY.lines().map(str::to_owned));
            characters.push(" ".to_owned());
            Self { characters }
        }
    }
}

impl OcrProcessor<det_model::Model, rec_model::Model> {
    pub fn pp_ocrv6_medium(device: Device) -> Self {
        Self {
            det_preprocess: DetPreProcessConfig::new().init(&device),
            det: det_model::Model::load(&device),
            det_postprocess: DbPostProcessConfig::new()
                .with_box_thresh(0.45)
                .init(&device),
            cropper: TextCropperConfig::new().init(&device),
            rec_batcher: RecBatcherConfig::new().init(&device),
            rec_preprocess: RecPreProcessConfig::new().init(&device),
            rec: rec_model::Model::load(&device),
            rec_postprocess: CtcDecoderConfig::pp_ocrv6_medium().init(&device),
        }
    }
}
