use std::borrow::Cow;
use std::num::NonZeroUsize;

use burn::config::Config;
use image::{RgbImage, imageops};

#[derive(Debug)]
pub struct RecBatch<'a> {
    pub indices: Vec<usize>,
    pub images: Vec<Cow<'a, RgbImage>>,
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
    pub fn init(&self) -> RecBatcher {
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
    pub fn forward<'a>(&self, crops: &[&'a RgbImage]) -> Vec<RecBatch<'a>> {
        let mut crops: Vec<_> = crops
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, crop)| crop.width() > 0 && crop.height() > 0)
            .map(|(index, crop)| (index, orient_crop(crop)))
            .collect();
        let aspect = |image: &RgbImage| image.width() as f64 / image.height() as f64;
        crops.sort_by(|(_, a), (_, b)| aspect(a).total_cmp(&aspect(b)));
        let mut crops = crops.into_iter();
        let mut batches = Vec::new();
        while crops.len() > 0 {
            let (indices, images) = crops.by_ref().take(self.batch_size.get()).unzip();
            batches.push(RecBatch { indices, images });
        }
        batches
    }
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "rec.orient_crop")
)]
fn orient_crop(crop: &RgbImage) -> Cow<'_, RgbImage> {
    if crop.height() as f64 / crop.width() as f64 >= 1.5 {
        Cow::Owned(imageops::rotate270(crop))
    } else {
        Cow::Borrowed(crop)
    }
}
