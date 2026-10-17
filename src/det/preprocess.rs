use std::num::{NonZeroU32, NonZeroUsize};

use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use image::RgbImage;

#[derive(Config, Debug)]
pub struct DetPreProcessConfig {
    #[config(default = "NonZeroU32::new(960).unwrap()")]
    pub max_side: NonZeroU32,
    #[config(default = "[0.485, 0.456, 0.406]")]
    pub mean: [f32; 3],
    #[config(default = "[0.229, 0.224, 0.225]")]
    pub std: [f32; 3],
}

impl DetPreProcessConfig {
    pub fn init(&self, device: &Device) -> DetPreProcess {
        DetPreProcess {
            max_side: self.max_side,
            mean: Tensor::<1>::from_data(self.mean, device).reshape([1, 3, 1, 1]),
            std: Tensor::<1>::from_data(self.std, device).reshape([1, 3, 1, 1]),
        }
    }
}

#[derive(Module, Debug)]
pub struct DetPreProcess {
    #[module(skip)]
    pub max_side: NonZeroU32,
    pub mean: Tensor<4>,
    pub std: Tensor<4>,
}

impl DetPreProcess {
    /// Images share the validated source shape, ordered `(height, width)`.
    /// Accepts RGB images and returns normalized BGR tensors in NCHW layout.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.preprocess")
    )]
    pub fn forward(
        &self,
        images: &[RgbImage],
        src_shape: (NonZeroUsize, NonZeroUsize),
    ) -> Tensor<4> {
        let (width, height) = self.resize_shape(src_shape);
        let input = self.upload(images, src_shape);
        let image = self.resize(input, [height as usize, width as usize]);
        self.normalize(image)
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.upload")
    )]
    pub fn upload(
        &self,
        images: &[RgbImage],
        shape: (NonZeroUsize, NonZeroUsize),
    ) -> Tensor<4, Int> {
        let device = self.mean.device();
        let images = images
            .iter()
            .map(|image| {
                Tensor::<3, Int>::from_data(
                    TensorData::new(image.as_raw().clone(), [shape.0.get(), shape.1.get(), 3]),
                    (&device, DType::U8),
                )
            })
            .collect();
        Tensor::stack(images, 0)
    }

    pub fn resize(&self, input: Tensor<4, Int>, shape: [usize; 2]) -> Tensor<4> {
        let image = input.float().permute([0, 3, 1, 2]);
        if image.dims()[2..] == shape {
            return image;
        }
        // Half-pixel bilinear sampling; retain integer-valued source pixels.
        interpolate(
            image,
            InterpolateOptions::new(InterpolateMode::Bilinear)
                .with_align_corners(false)
                .with_output_size(shape),
        )
        .round()
    }

    pub fn normalize(&self, image: Tensor<4>) -> Tensor<4> {
        // Upstream applies these mean/std values to BGR, in their listed order.
        (image.flip([1]) / 255.0 - self.mean.clone()) / self.std.clone()
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.resize_shape")
    )]
    pub fn resize_shape(&self, (height, width): (NonZeroUsize, NonZeroUsize)) -> (u32, u32) {
        let ratio = (self.max_side.get() as f64 / width.max(height).get() as f64).min(1.0);
        let align = |length: NonZeroUsize| {
            let resized = (length.get() as f64 * ratio) as u32;
            ((resized as f64 / 32.0).round_ties_even() as u32 * 32).max(32)
        };
        (align(width), align(height))
    }
}
