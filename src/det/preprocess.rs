use std::num::NonZeroUsize;

use burn::prelude::*;
use image::{RgbImage, imageops};

pub struct DetPreProcess {
    pub max_side: u32,
}

impl DetPreProcess {
    /// Images share the validated source shape, ordered `(height, width)`.
    pub fn forward(
        &self,
        images: &[RgbImage],
        src_shape: (NonZeroUsize, NonZeroUsize),
        device: &Device,
    ) -> Tensor<4> {
        let (width, height) = self.resize_shape(src_shape);
        let pixels: Vec<_> = images
            .iter()
            .flat_map(|image| {
                imageops::resize(image, width, height, imageops::FilterType::Triangle).into_raw()
            })
            .collect();
        let image = Tensor::<4>::from_data(
            TensorData::new(pixels, [images.len(), height as usize, width as usize, 3])
                .convert::<f32>(),
            device,
        )
        .permute([0, 3, 1, 2])
        .flip([1]);
        let mean = Tensor::<1>::from_data([0.485f32, 0.456, 0.406], device).reshape([1, 3, 1, 1]);
        let std = Tensor::<1>::from_data([0.229f32, 0.224, 0.225], device).reshape([1, 3, 1, 1]);
        (image / 255.0 - mean) / std
    }

    fn resize_shape(&self, (height, width): (NonZeroUsize, NonZeroUsize)) -> (u32, u32) {
        let ratio = (self.max_side as f64 / width.max(height).get() as f64).min(1.0);
        let align = |length: NonZeroUsize| {
            let resized = (length.get() as f64 * ratio) as u32;
            ((resized as f64 / 32.0).round_ties_even() as u32 * 32).max(32)
        };
        (align(width), align(height))
    }
}
