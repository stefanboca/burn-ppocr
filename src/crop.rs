use burn::config::Config;
use burn::tensor::Device;
use image::RgbImage;
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

use crate::Detection;
use crate::geometry::{Point, Polygon};

/// A rectified text region, retaining its source image and detection.
#[derive(Debug)]
pub struct TextCrop {
    pub image_index: usize,
    pub detection: Detection,
    pub image: RgbImage,
}

#[derive(Config, Debug)]
pub struct TextCropperConfig {
    #[config(default = 10.0)]
    pub reading_order_tolerance: f32,
}

impl TextCropperConfig {
    pub fn init(&self, _device: &Device) -> TextCropper {
        TextCropper {
            reading_order_tolerance: self.reading_order_tolerance,
        }
    }
}

/// Sort text regions into reading order and rectify valid quadrilaterals.
#[derive(Clone, Debug)]
pub struct TextCropper {
    pub reading_order_tolerance: f32,
}

impl TextCropper {
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "ocr.crop")
    )]
    pub fn forward(&self, images: &[RgbImage], detections: Vec<Vec<Detection>>) -> Vec<TextCrop> {
        assert_eq!(
            detections.len(),
            images.len(),
            "detector must return one result list per image"
        );
        let mut crops = Vec::new();
        for (image_index, (image, mut detections)) in images.iter().zip(detections).enumerate() {
            detections.retain(|detection| detection.points.0.len() == 4);
            sort_reading_order(&mut detections, self.reading_order_tolerance);
            for detection in detections {
                if let Some(crop) = crop_text(image, &detection.points) {
                    crops.push(TextCrop {
                        image_index,
                        detection,
                        image: crop,
                    });
                }
            }
        }
        crops
    }
}

/// Rectify the four ordered corners without applying model-specific orientation rules.
#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "ocr.crop_text")
)]
pub fn crop_text(image: &RgbImage, polygon: &Polygon) -> Option<RgbImage> {
    if image.width() == 0 || image.height() == 0 {
        return None;
    }
    let points: [Point<f32>; 4] = polygon.0.as_slice().try_into().ok()?;
    if !points
        .iter()
        .all(|point| point.x.is_finite() && point.y.is_finite())
    {
        return None;
    }
    let distance = |a: Point<f32>, b: Point<f32>| (a.x - b.x).hypot(a.y - b.y);
    let width = distance(points[0], points[1]).max(distance(points[2], points[3])) as u32;
    let height = distance(points[0], points[3]).max(distance(points[1], points[2])) as u32;
    if width == 0 || height == 0 {
        return None;
    }
    let projection = Projection::from_control_points(
        points.map(|point| (point.x, point.y)),
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

/// Sort top-to-bottom, then left-to-right within the vertical tolerance.
#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "ocr.sort_reading_order")
)]
pub fn sort_reading_order(detections: &mut [Detection], tolerance: f32) {
    detections.sort_by(|a, b| {
        let a = a.points.0[0];
        let b = b.points.0[0];
        a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x))
    });
    for i in 1..detections.len() {
        let mut j = i;
        while j > 0 {
            let a = detections[j - 1].points.0[0];
            let b = detections[j].points.0[0];
            if (b.y - a.y).abs() >= tolerance || b.x >= a.x {
                break;
            }
            detections.swap(j - 1, j);
            j -= 1;
        }
    }
}
