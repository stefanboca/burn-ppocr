use image::{DynamicImage, RgbImage};

use crate::db_postprocess::Detection;
use crate::models::{Det, Rec};
use crate::preprocess::crop_text;
use crate::rec_postprocess::Recognition;

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
            Box::new(crate::models::PpOcrV6TinyDet::new(device.clone())),
            Box::new(crate::models::PpOcrV6TinyRec::new(device)),
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
    use crate::geometry::Polygon;

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
}
