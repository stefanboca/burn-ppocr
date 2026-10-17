use std::num::NonZeroU32;

use burn::prelude::*;
use burn::tensor::ops::PadMode;
use image::{RgbImage, imageops};

#[derive(Clone, Debug)]
pub struct RecPreProcess {
    pub height: NonZeroU32,
    pub width: NonZeroU32,
}

impl RecPreProcess {
    pub fn batch_width(&self, images: &[&RgbImage]) -> NonZeroU32 {
        let ratio = images
            .iter()
            .map(|image| image.width() as f64 / image.height() as f64)
            .fold(self.width.get() as f64 / self.height.get() as f64, f64::max);
        NonZeroU32::new((self.height.get() as f64 * ratio) as u32)
            .unwrap_or(self.width)
            .max(self.width)
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.preprocess")
    )]
    pub fn forward(&self, crop: &RgbImage, batch_width: NonZeroU32, device: &Device) -> Tensor<3> {
        let batch_width = batch_width.get();
        let height = self.height.get();
        let width = (height as f64 * crop.width() as f64 / crop.height() as f64)
            .ceil()
            .min(batch_width as f64) as u32;
        let resized = imageops::resize(crop, width, height, imageops::FilterType::Triangle);
        let image = Tensor::<3>::from_data(
            TensorData::new(resized.into_raw(), [height as usize, width as usize, 3])
                .convert::<f32>(),
            device,
        )
        .permute([2, 0, 1])
        .flip([0]);
        // Padding is zero in normalized space, not black in the source image.
        (image / 127.5 - 1.0).pad(
            [(0, (batch_width - width) as usize)],
            PadMode::Constant(0.0),
        )
    }
}

pub fn orient_crop(crop: &RgbImage) -> RgbImage {
    if crop.height() as f64 / crop.width() as f64 >= 1.5 {
        imageops::rotate270(crop)
    } else {
        crop.clone()
    }
}
