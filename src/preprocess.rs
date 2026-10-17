use burn::prelude::*;
use image::{RgbImage, imageops};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

use crate::geometry::Polygon;

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
        let plane = width as usize * height as usize;
        let mut values = vec![0.0f32; 3 * plane];
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        for (index, pixel) in resized.pixels().enumerate() {
            for (channel, source) in [2, 1, 0].into_iter().enumerate() {
                values[channel * plane + index] =
                    (pixel[source] as f32 / 255.0 - mean[channel]) / std[channel];
            }
        }
        Tensor::from_data(
            TensorData::new(values, [1, 3, height as usize, width as usize]),
            device,
        )
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

pub struct RecPreProcess {
    pub height: u32,
    pub width: u32,
}

impl Default for RecPreProcess {
    fn default() -> Self {
        Self {
            height: 48,
            width: 320,
        }
    }
}

impl RecPreProcess {
    pub fn batch_width(&self, images: &[&RgbImage]) -> u32 {
        let ratio = images
            .iter()
            .map(|image| image.width() as f64 / image.height() as f64)
            .fold(self.width as f64 / self.height as f64, f64::max);
        (self.height as f64 * ratio) as u32
    }

    pub fn forward(&self, crop: &RgbImage, batch_width: u32, device: &Device) -> Tensor<3> {
        let width = (self.height as f64 * crop.width() as f64 / crop.height() as f64)
            .ceil()
            .min(batch_width as f64) as u32;
        let resized = imageops::resize(crop, width, self.height, imageops::FilterType::Triangle);
        // Padding is zero in normalized space, not black in the source image.
        let plane = batch_width as usize * self.height as usize;
        let mut values = vec![0.0f32; 3 * plane];
        for (x, y, pixel) in resized.enumerate_pixels() {
            let index = y as usize * batch_width as usize + x as usize;
            for (channel, source) in [2, 1, 0].into_iter().enumerate() {
                values[channel * plane + index] = pixel[source] as f32 / 127.5 - 1.0;
            }
        }
        Tensor::from_data(
            TensorData::new(values, [3, self.height as usize, batch_width as usize]),
            device,
        )
    }
}

/// Rectify the four ordered corners without applying model-specific orientation rules.
pub fn crop_text(image: &RgbImage, polygon: &Polygon) -> Option<RgbImage> {
    let points: [[f32; 2]; 4] = polygon.0.as_slice().try_into().ok()?;
    if !points.iter().flatten().all(|x| x.is_finite()) {
        return None;
    }
    let distance = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    let width = distance(points[0], points[1]).max(distance(points[2], points[3])) as u32;
    let height = distance(points[0], points[3]).max(distance(points[1], points[2])) as u32;
    if width == 0 || height == 0 {
        return None;
    }
    let projection = Projection::from_control_points(
        points.map(|[x, y]| (x, y)),
        [
            (0.0, 0.0),
            (width as f32, 0.0),
            (width as f32, height as f32),
            (0.0, height as f32),
        ],
    )?;
    let mut crop = RgbImage::new(width, height);
    warp_into(
        image,
        projection,
        Interpolation::Bicubic,
        Border::Replicate,
        &mut crop,
    );
    Some(crop)
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
    fn detector_uses_bgr_channel_order_and_imagenet_constants() {
        let image = RgbImage::from_pixel(32, 32, Rgb([255, 128, 0]));
        let input = DetPreProcess::default().forward(&image, &Device::flex());
        assert_eq!(input.dims(), [1, 3, 32, 32]);
        let data = input.into_data().try_into_vec_as::<f32>().unwrap();
        assert!((data[0] - (0.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((data[1024] - (128.0 / 255.0 - 0.456) / 0.224).abs() < 1e-5);
        assert!((data[2048] - (1.0 - 0.406) / 0.225).abs() < 1e-5);
    }

    #[test]
    fn recognizer_normalization_and_padding_are_distinct() {
        let preprocess = RecPreProcess {
            height: 2,
            width: 6,
        };
        let image = RgbImage::from_pixel(2, 1, Rgb([255, 0, 0]));
        let input = preprocess.forward(&image, 6, &Device::flex());
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
            height: 48,
            width: 64,
        };
        let image = RgbImage::new(100, 47);
        assert_eq!(preprocess.batch_width(&[&image]), 102);
        assert_eq!(RecPreProcess::default().batch_width(&[&image]), 320);
    }

    #[test]
    fn perspective_crop_replicates_source_borders() {
        let image = RgbImage::from_pixel(3, 3, Rgb([17, 80, 200]));
        let polygon = Polygon(vec![[-1.0, -1.0], [4.0, -1.0], [4.0, 4.0], [-1.0, 4.0]]);
        let crop = crop_text(&image, &polygon).unwrap();
        assert_eq!(crop.dimensions(), (5, 5));
        for pixel in crop.pixels() {
            for (actual, expected) in pixel.0.into_iter().zip([17u8, 80, 200]) {
                assert!(actual.abs_diff(expected) <= 1);
            }
        }
    }

    #[test]
    fn tall_crop_is_rotated_counterclockwise() {
        let image = RgbImage::from_fn(4, 8, |x, y| Rgb([x as u8, y as u8, 0]));
        let oriented = orient_crop(&image);
        assert_eq!(oriented.dimensions(), (8, 4));
        assert_eq!(oriented.get_pixel(0, 0), image.get_pixel(3, 0));
        assert!(crop_text(&image, &Polygon(vec![[0.0, 0.0]; 4])).is_none());
    }
}
