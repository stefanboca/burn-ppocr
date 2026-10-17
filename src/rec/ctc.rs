use anyhow::ensure;
use burn::prelude::*;
use image::RgbImage;

use super::preprocess::{RecPreProcess, orient_crop};
use super::{CtcDecoder, RecModel, Recognition};

#[derive(Module, Debug)]
pub struct CtcRecognizer<M: RecModel> {
    pub model: M,
    #[module(skip)]
    pub preprocess: RecPreProcess,
    #[module(skip)]
    pub decoder: CtcDecoder,
    pub batch_size: usize,
}

impl<M: RecModel> CtcRecognizer<M> {
    /// Upstream's default batching policy, independent of the recognition model.
    pub const DEFAULT_BATCH_SIZE: usize = 6;

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.forward")
    )]
    pub fn forward(&self, crops: &[RgbImage], device: &Device) -> anyhow::Result<Vec<Recognition>> {
        ensure!(
            crops
                .iter()
                .all(|crop| crop.width() > 0 && crop.height() > 0),
            "recognition crops must be nonempty"
        );
        let crops: Vec<_> = crops.iter().map(orient_crop).collect();
        let mut indices: Vec<_> = (0..crops.len()).collect();
        let aspect_ratio =
            |index: usize| crops[index].width() as f64 / crops[index].height() as f64;
        indices.sort_by(|&a, &b| aspect_ratio(a).total_cmp(&aspect_ratio(b)));
        let mut results = vec![None; crops.len()];
        for batch in indices.chunks(self.batch_size) {
            let images: Vec<_> = batch.iter().map(|&index| &crops[index]).collect();
            let width = self.preprocess.batch_width(&images);
            let images = images
                .into_iter()
                .map(|image| self.preprocess.forward(image, width, device))
                .collect();
            let input = Tensor::stack::<4>(images, 0);
            let predictions = {
                #[cfg(feature = "profiling")]
                let _span = tracing::info_span!("rec.model").entered();
                self.model.forward(input)
            };
            assert_eq!(
                predictions.dims()[0],
                batch.len(),
                "recognizer batch size mismatch"
            );
            for (&index, result) in batch.iter().zip(self.decoder.decode(predictions)?) {
                results[index] = Some(result);
            }
        }
        Ok(results
            .into_iter()
            .map(|result| result.expect("missing recognition result"))
            .collect())
    }
}
