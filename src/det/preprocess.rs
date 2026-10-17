use burn::prelude::*;
use image::{RgbImage, imageops};

pub struct DetPreProcess {
    pub max_side: u32,
}

impl Default for DetPreProcess {
    fn default() -> Self {
        Self { max_side: 960 }
    }
}

impl DetPreProcess {
    pub fn forward(&self, image: &RgbImage, device: &Device) -> Tensor<4> {
        let (width, height) = self.resize_shape(image.width(), image.height());
        let resized = imageops::resize(image, width, height, imageops::FilterType::Triangle);
        let image = Tensor::<4>::from_data(
            TensorData::new(resized.into_raw(), [1, height as usize, width as usize, 3])
                .convert::<f32>(),
            device,
        )
        .permute([0, 3, 1, 2])
        .flip([1]);
        let mean = Tensor::<1>::from_data([0.485f32, 0.456, 0.406], device).reshape([1, 3, 1, 1]);
        let std = Tensor::<1>::from_data([0.229f32, 0.224, 0.225], device).reshape([1, 3, 1, 1]);
        (image / 255.0 - mean) / std
    }

    fn resize_shape(&self, width: u32, height: u32) -> (u32, u32) {
        let ratio = (self.max_side as f64 / width.max(height) as f64).min(1.0);
        let align = |length: u32| {
            let resized = (length as f64 * ratio) as u32;
            ((resized as f64 / 32.0).round_ties_even() as u32 * 32).max(32)
        };
        (align(width), align(height))
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn detector_uses_bgr_channel_order_and_imagenet_constants() {
        let image = RgbImage::from_pixel(32, 32, Rgb([255, 128, 0]));
        let input = DetPreProcess::default().forward(&image, &Device::flex());
        assert_eq!(input.dims(), [1, 3, 32, 32]);
        let data = input.into_data().try_into_vec_as::<f32>().unwrap();
        assert!((data[0] - (0.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((data[1024] - (128.0 / 255.0 - 0.456) / 0.224).abs() < 1e-5);
        assert!((data[2048] - (1.0 - 0.406) / 0.225).abs() < 1e-5);
    }
}
