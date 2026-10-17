use image::{DynamicImage, RgbImage};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

use crate::geometry::Polygon;
use crate::{Det, Detection, Rec, Recognition};

/// A text region in source-image coordinates, with detection and recognition scores.
#[derive(Debug)]
pub struct OcrResult {
    pub detection: Detection,
    pub recognition: Recognition,
}

/// Coordinates image detection, rectification, and recognition.
/// Models own their device, preprocessing, batching, and decoding policies.
pub struct OcrPipeline {
    det: Box<dyn Det>,
    rec: Box<dyn Rec>,
}

impl OcrPipeline {
    pub fn new(det: Box<dyn Det>, rec: Box<dyn Rec>) -> Self {
        Self { det, rec }
    }

    #[cfg(feature = "pp_ocrv6_tiny")]
    pub fn pp_ocrv6_tiny(device: burn::tensor::Device) -> Self {
        Self::new(
            Box::new(crate::DbDetector::pp_ocrv6_tiny(device.clone())),
            Box::new(crate::CtcRecognizer::pp_ocrv6_tiny(device)),
        )
    }

    /// Recognize an image loaded with the `image` crate. No tensors cross this API.
    pub fn forward(&self, image: &DynamicImage) -> Vec<OcrResult> {
        if image.width() == 0 || image.height() == 0 {
            return Vec::new();
        }
        self.forward_rgb(&image.to_rgb8())
    }

    fn forward_rgb(&self, image: &RgbImage) -> Vec<OcrResult> {
        let mut detections = self.det.forward(image);
        detections.retain(|detection| detection.points.0.len() == 4);
        sort_reading_order(&mut detections);
        let (detections, crops): (Vec<_>, Vec<_>) = detections
            .into_iter()
            .filter_map(|detection| {
                crop_text(image, &detection.points).map(|crop| (detection, crop))
            })
            .unzip();
        if crops.is_empty() {
            return Vec::new();
        }
        let recognitions = self.rec.forward(&crops);
        assert_eq!(
            recognitions.len(),
            detections.len(),
            "recognizer must return one result per crop"
        );
        detections
            .into_iter()
            .zip(recognitions)
            .map(|(detection, recognition)| OcrResult {
                detection,
                recognition,
            })
            .collect()
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

/// PaddleOCR's top-to-bottom sort, followed by left-to-right ordering within 10 pixels.
fn sort_reading_order(detections: &mut [Detection]) {
    detections.sort_by(|a, b| {
        let a = a.points.0[0];
        let b = b.points.0[0];
        a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0]))
    });
    for i in 1..detections.len() {
        let mut j = i;
        while j > 0 {
            let a = detections[j - 1].points.0[0];
            let b = detections[j].points.0[0];
            if (b[1] - a[1]).abs() >= 10.0 || b[0] >= a[0] {
                break;
            }
            detections.swap(j - 1, j);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use image::Rgb;

    use super::*;

    struct Detector;
    impl Det for Detector {
        fn forward(&self, _: &RgbImage) -> Vec<Detection> {
            vec![region(20.0, 10.0), region(2.0, 12.0)]
        }
    }

    struct Recognizer;
    impl Rec for Recognizer {
        fn forward(&self, crops: &[RgbImage]) -> Vec<Recognition> {
            assert_eq!(crops.len(), 2);
            assert_eq!(crops[0].dimensions(), (12, 6));
            vec![
                Recognition {
                    text: "left".into(),
                    score: 0.9,
                },
                Recognition {
                    text: "right".into(),
                    score: 0.8,
                },
            ]
        }
    }

    fn region(x: f32, y: f32) -> Detection {
        Detection {
            points: Polygon(vec![
                [x, y],
                [x + 12.0, y],
                [x + 12.0, y + 6.0],
                [x, y + 6.0],
            ]),
            score: 0.95,
        }
    }

    #[test]
    fn image_to_text_api_preserves_detection_recognition_associations() {
        let pipeline = OcrPipeline::new(Box::new(Detector), Box::new(Recognizer));
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 30, Rgb([255, 255, 255])));
        let results = pipeline.forward(&image);
        assert_eq!(results[0].recognition.text, "left");
        assert_eq!(results[0].detection.points.0[0], [2.0, 12.0]);
        assert_eq!(results[1].recognition.text, "right");
    }

    struct EmptyDetector;
    impl Det for EmptyDetector {
        fn forward(&self, _: &RgbImage) -> Vec<Detection> {
            Vec::new()
        }
    }
    struct CountingRecognizer(Rc<Cell<usize>>);
    impl Rec for CountingRecognizer {
        fn forward(&self, _: &[RgbImage]) -> Vec<Recognition> {
            self.0.set(self.0.get() + 1);
            Vec::new()
        }
    }

    #[test]
    fn empty_images_and_empty_detections_skip_recognition() {
        let calls = Rc::new(Cell::new(0));
        let pipeline = OcrPipeline::new(
            Box::new(EmptyDetector),
            Box::new(CountingRecognizer(calls.clone())),
        );
        assert!(pipeline.forward(&DynamicImage::new_rgb8(10, 10)).is_empty());
        assert_eq!(calls.get(), 0);
        assert!(pipeline.forward(&DynamicImage::new_rgb8(0, 0)).is_empty());
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
        assert!(crop_text(&image, &Polygon(vec![[0.0, 0.0]; 4])).is_none());
    }
}
