use std::num::NonZeroU32;

use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::ops::PadMode;
use image::{RgbImage, imageops};

#[derive(Config, Debug)]
pub struct RecPreProcessConfig {
    #[config(default = "NonZeroU32::new(48).unwrap()")]
    pub height: NonZeroU32,
    #[config(default = "NonZeroU32::new(320).unwrap()")]
    pub width: NonZeroU32,
    #[config(default = "[0.5; 3]")]
    pub mean: [f32; 3],
    #[config(default = "[0.5; 3]")]
    pub std: [f32; 3],
}

impl RecPreProcessConfig {
    pub fn init(&self, device: &Device) -> RecPreProcess {
        RecPreProcess {
            height: self.height,
            width: self.width,
            mean: Tensor::<1>::from_data(self.mean, device).reshape([3, 1, 1]),
            std: Tensor::<1>::from_data(self.std, device).reshape([3, 1, 1]),
        }
    }
}

#[derive(Module, Debug)]
pub struct RecPreProcess {
    #[module(skip)]
    pub height: NonZeroU32,
    #[module(skip)]
    pub width: NonZeroU32,
    pub mean: Tensor<3>,
    pub std: Tensor<3>,
}

impl RecPreProcess {
    /// Resize, normalize, pad, and stack a batch of already oriented crops.
    /// Accepts RGB crops and returns normalized BGR tensors in NCHW layout.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.preprocess")
    )]
    pub fn forward(&self, crops: &[RgbImage]) -> Tensor<4> {
        let images: Vec<_> = crops.iter().collect();
        let width = self.batch_width(&images);
        let images = crops
            .iter()
            .map(|crop| self.forward_image(crop, width))
            .collect();
        Tensor::stack(images, 0)
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.batch_width")
    )]
    pub fn batch_width(&self, images: &[&RgbImage]) -> NonZeroU32 {
        let ratio = images
            .iter()
            .map(|image| image.width() as f64 / image.height() as f64)
            .fold(self.width.get() as f64 / self.height.get() as f64, f64::max);
        NonZeroU32::new((self.height.get() as f64 * ratio) as u32)
            .unwrap_or(self.width)
            .max(self.width)
    }

    pub fn forward_image(&self, crop: &RgbImage, batch_width: NonZeroU32) -> Tensor<3> {
        let resized = self.resize_image(crop, batch_width);
        let batch_width = batch_width.get();
        let (width, height) = resized.dimensions();
        let image = Tensor::<3, Int>::from_data(
            TensorData::new(resized.into_raw(), [height as usize, width as usize, 3]),
            (&self.mean.device(), DType::U8),
        )
        .float()
        .permute([2, 0, 1])
        // The recognizer, like the detector, expects BGR input.
        .flip([0]);
        // Padding is zero in normalized space, not black in the source image.
        ((image / 255.0 - self.mean.clone()) / self.std.clone()).pad(
            [(0, (batch_width - width) as usize)],
            PadMode::Constant(0.0),
        )
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.resize_image")
    )]
    fn resize_image(&self, crop: &RgbImage, batch_width: NonZeroU32) -> RgbImage {
        let height = self.height.get();
        let width = (height as f64 * crop.width() as f64 / crop.height() as f64)
            .ceil()
            .min(batch_width.get() as f64) as u32;
        imageops::resize(crop, width, height, imageops::FilterType::Triangle)
    }
}
