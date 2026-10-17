use std::num::NonZeroU32;

use burn::prelude::*;
use burn::tensor::ops::PadMode;
use image::{RgbImage, imageops};

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

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn recognizer_normalization_and_padding_are_distinct() {
        let preprocess = RecPreProcess {
            height: NonZeroU32::new(2).unwrap(),
            width: NonZeroU32::new(6).unwrap(),
        };
        let image = RgbImage::from_pixel(2, 1, Rgb([255, 0, 0]));
        let input = preprocess.forward(&image, NonZeroU32::new(6).unwrap(), &Device::flex());
        assert_eq!(input.dims(), [3, 2, 6]);
        let values = input.into_data().try_into_vec_as::<f32>().unwrap();
        for channel in 0..3 {
            for y in 0..2 {
                for x in 0..6 {
                    let expected = if x >= 4 {
                        0.0
                    } else if channel == 2 {
                        1.0
                    } else {
                        -1.0
                    };
                    assert_eq!(values[channel * 12 + y * 6 + x], expected);
                }
            }
        }
    }

    #[test]
    fn recognition_batch_width_is_not_rounded_to_32() {
        let preprocess = RecPreProcess {
            height: NonZeroU32::new(48).unwrap(),
            width: NonZeroU32::new(64).unwrap(),
        };
        let image = RgbImage::new(100, 47);
        assert_eq!(preprocess.batch_width(&[&image]).get(), 102);
        assert_eq!(
            RecPreProcess {
                height: NonZeroU32::new(48).unwrap(),
                width: NonZeroU32::new(320).unwrap()
            }
            .batch_width(&[&image])
            .get(),
            320
        );
    }

    #[test]
    fn tall_crop_is_rotated_counterclockwise() {
        let image = RgbImage::from_fn(4, 8, |x, y| Rgb([x as u8, y as u8, 0]));
        let oriented = orient_crop(&image);
        assert_eq!(oriented.dimensions(), (8, 4));
        assert_eq!(oriented.get_pixel(0, 0), image.get_pixel(3, 0));
    }
}
