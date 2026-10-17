use anyhow::ensure;
use burn::module::Module;
use burn::tensor::Device;
use image::{DynamicImage, GenericImageView, RgbImage};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

use crate::geometry::{Point, Polygon};
use crate::{Det, Detection, Rec, Recognition};

/// A text region in source-image coordinates, with detection and recognition scores.
#[derive(Debug)]
pub struct OcrResult {
    pub detection: Detection,
    pub recognition: Recognition,
}

/// Coordinates image detection, rectification, and recognition.
/// Models own preprocessing, batching, and decoding policies.
#[derive(Module, Debug)]
pub struct OcrPipeline<D: Det, R: Rec> {
    pub det: D,
    pub rec: R,
}

impl<D: Det, R: Rec> OcrPipeline<D, R> {
    pub fn new(det: D, rec: R) -> Self {
        Self { det, rec }
    }

    /// Recognize a batch of equally sized images, returning results in input order.
    /// Preprocessing creates tensors on `device`, which must match the models' device.
    /// Returns an error when image dimensions differ within the batch.
    pub fn forward(
        &self,
        images: &[DynamicImage],
        device: &Device,
    ) -> anyhow::Result<Vec<Vec<OcrResult>>> {
        let Some(image) = images.first() else {
            return Ok(Vec::new());
        };
        let shape = image.dimensions();
        for (image_index, image) in images.iter().enumerate().skip(1) {
            let actual = image.dimensions();
            ensure!(
                actual == shape,
                "image {image_index} has dimensions {}x{}; expected {}x{}",
                actual.0,
                actual.1,
                shape.0,
                shape.1
            );
        }
        let mut results: Vec<Vec<OcrResult>> = (0..images.len()).map(|_| Vec::new()).collect();
        if shape.0 == 0 || shape.1 == 0 {
            return Ok(results);
        }
        let images: Vec<_> = images.iter().map(DynamicImage::to_rgb8).collect();
        let detections = self.det.forward(&images, device)?;
        assert_eq!(
            detections.len(),
            images.len(),
            "detector must return one result list per image"
        );
        let mut regions = Vec::new();
        let mut crops = Vec::new();
        for (image_index, (image, mut detections)) in images.iter().zip(detections).enumerate() {
            detections.retain(|detection| detection.points.0.len() == 4);
            sort_reading_order(&mut detections);
            for detection in detections {
                if let Some(crop) = crop_text(image, &detection.points) {
                    regions.push((image_index, detection));
                    crops.push(crop);
                }
            }
        }
        if crops.is_empty() {
            return Ok(results);
        }
        let recognitions = self.rec.forward(&crops, device)?;
        assert_eq!(
            recognitions.len(),
            regions.len(),
            "recognizer must return one result per crop"
        );
        for ((image_index, detection), recognition) in regions.into_iter().zip(recognitions) {
            results[image_index].push(OcrResult {
                detection,
                recognition,
            });
        }
        Ok(results)
    }
}

/// Rectify the four ordered corners without applying model-specific orientation rules.
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

/// PaddleOCR's top-to-bottom sort, followed by left-to-right ordering within 10 pixels.
fn sort_reading_order(detections: &mut [Detection]) {
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
            if (b.y - a.y).abs() >= 10.0 || b.x >= a.x {
                break;
            }
            detections.swap(j - 1, j);
            j -= 1;
        }
    }
}
