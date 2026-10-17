use std::num::NonZeroUsize;

use anyhow::ensure;
use burn::module::Module;
use image::{DynamicImage, GenericImageView};

use crate::crop::TextCropper;
use crate::det::preprocess::DetPreProcess;
use crate::det::{DbPostProcess, DetModel};
use crate::rec::preprocess::RecPreProcess;
use crate::rec::{CtcDecoder, RecBatcher, RecModel};
use crate::{Detection, Recognition};

/// A text region in source-image coordinates, with detection and recognition scores.
#[derive(Debug)]
pub struct OcrResult {
    pub detection: Detection,
    pub recognition: Recognition,
}

/// Compose independent processing stages and raw detection/recognition models.
#[derive(Module, Debug)]
pub struct OcrProcessor<D: DetModel, R: RecModel> {
    pub det_preprocess: DetPreProcess,
    pub det: D,
    #[module(skip)]
    pub det_postprocess: DbPostProcess,
    #[module(skip)]
    pub cropper: TextCropper,
    #[module(skip)]
    pub rec_batcher: RecBatcher,
    pub rec_preprocess: RecPreProcess,
    pub rec: R,
    #[module(skip)]
    pub rec_postprocess: CtcDecoder,
}

impl<D: DetModel, R: RecModel> OcrProcessor<D, R> {
    /// Recognize equally sized images in input and reading order.
    /// Unequal image dimensions return an error.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "ocr.forward")
    )]
    pub fn forward(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<Vec<OcrResult>>> {
        let Some(image) = images.first() else {
            return Ok(Vec::new());
        };
        let shape = image.dimensions();
        ensure!(shape.0 > 0 && shape.1 > 0);
        ensure!(images.iter().all(|img| img.dimensions() == shape));
        let mut results: Vec<Vec<OcrResult>> = (0..images.len()).map(|_| Vec::new()).collect();
        let Some(src_shape) =
            NonZeroUsize::new(shape.1 as usize).zip(NonZeroUsize::new(shape.0 as usize))
        else {
            return Ok(results);
        };
        let images: Vec<_> = images.iter().map(DynamicImage::to_rgb8).collect();
        let input = self.det_preprocess.forward(&images, src_shape);
        let maps = self.det.forward(input);
        assert_eq!(maps.dims()[0], images.len());
        let detections = self.det_postprocess.forward(maps, src_shape)?;
        let crops = self.cropper.forward(&images, detections);
        let crop_images: Vec<_> = crops.iter().map(|crop| &crop.image).collect();
        let mut recognitions = vec![None; crops.len()];
        for batch in self.rec_batcher.forward(&crop_images) {
            let input = self.rec_preprocess.forward(&batch.images);
            let predictions = self.rec.forward(input);
            assert_eq!(predictions.dims()[0], batch.indices.len());
            for (index, recognition) in batch
                .indices
                .into_iter()
                .zip(self.rec_postprocess.forward(predictions)?)
            {
                recognitions[index] = Some(recognition);
            }
        }
        for (crop, recognition) in crops.into_iter().zip(recognitions) {
            results[crop.image_index].push(OcrResult {
                detection: crop.detection,
                recognition: recognition.expect("missing recognition result"),
            });
        }
        Ok(results)
    }
}
