use std::num::NonZeroUsize;

use burn::config::Config;
use burn::tensor::Device;
use image::RgbImage;

use super::preprocess::orient_crop;

#[derive(Debug)]
pub struct RecBatch {
    pub indices: Vec<usize>,
    pub images: Vec<RgbImage>,
}

/// Orient crops and group them by aspect ratio, retaining their original indices.
#[derive(Clone, Debug)]
pub struct RecBatcher {
    pub batch_size: NonZeroUsize,
}

#[derive(Config, Debug)]
pub struct RecBatcherConfig {
    #[config(default = "NonZeroUsize::new(6).unwrap()")]
    pub batch_size: NonZeroUsize,
}

impl RecBatcherConfig {
    pub fn init(&self, _device: &Device) -> RecBatcher {
        RecBatcher {
            batch_size: self.batch_size,
        }
    }
}

impl RecBatcher {
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.batch")
    )]
    pub fn forward(&self, crops: &[&RgbImage]) -> Vec<RecBatch> {
        let mut crops: Vec<_> = crops
            .iter()
            .enumerate()
            .filter(|(_, crop)| crop.width() > 0 && crop.height() > 0)
            .map(|(index, crop)| (index, orient_crop(crop)))
            .collect();
        crops.sort_by(|(_, a), (_, b)| {
            let aspect = |image: &RgbImage| image.width() as f64 / image.height() as f64;
            aspect(a).total_cmp(&aspect(b))
        });
        let mut crops = crops.into_iter();
        let mut batches = Vec::new();
        while crops.len() > 0 {
            let (indices, images) = crops.by_ref().take(self.batch_size.get()).unzip();
            batches.push(RecBatch { indices, images });
        }
        batches
    }
}
