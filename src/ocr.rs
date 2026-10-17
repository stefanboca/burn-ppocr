use anyhow::ensure;
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

    /// Recognize a batch of equally sized images, returning results in input order.
    /// Returns an error when image dimensions differ within the batch.
    pub fn forward(&self, images: &[DynamicImage]) -> anyhow::Result<Vec<Vec<OcrResult>>> {
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
        let detections = self.det.forward(&images)?;
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
        let recognitions = self.rec.forward(&crops)?;
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use image::Rgb;

    use super::*;

    struct Detector;
    impl Det for Detector {
        fn forward(&self, images: &[RgbImage]) -> anyhow::Result<Vec<Vec<Detection>>> {
            Ok(images
                .iter()
                .map(|_| {
                    vec![
                        region(20.0, 10.0),
                        region(2.0, 12.0),
                        Detection {
                            points: Polygon(Vec::new()),
                            score: 0.9,
                        },
                        Detection {
                            points: Polygon(vec![Point::new(0.0, 0.0); 4]),
                            score: 0.9,
                        },
                    ]
                })
                .collect())
        }
    }

    struct Recognizer;
    impl Rec for Recognizer {
        fn forward(&self, crops: &[RgbImage]) -> anyhow::Result<Vec<Recognition>> {
            assert_eq!(crops.len(), 4);
            Ok(crops
                .iter()
                .enumerate()
                .map(|(index, crop)| {
                    assert_eq!(crop.dimensions(), (12, 6));
                    let color = if crop.get_pixel(0, 0)[0] > 0 {
                        "red"
                    } else {
                        "green"
                    };
                    let side = if index % 2 == 0 { "left" } else { "right" };
                    Recognition {
                        text: format!("{color} {side}"),
                        score: 0.9,
                    }
                })
                .collect())
        }
    }

    fn region(x: f32, y: f32) -> Detection {
        Detection {
            points: Polygon(vec![
                Point::new(x, y),
                Point::new(x + 12.0, y),
                Point::new(x + 12.0, y + 6.0),
                Point::new(x, y + 6.0),
            ]),
            score: 0.95,
        }
    }

    #[test]
    fn image_to_text_api_preserves_detection_recognition_associations() {
        let pipeline = OcrPipeline::new(Box::new(Detector), Box::new(Recognizer));
        let images = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 30, Rgb([255, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 30, Rgb([0, 255, 0]))),
        ];
        let results = pipeline.forward(&images).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2]);
        assert_eq!(results[0][0].recognition.text, "red left");
        assert_eq!(results[0][1].recognition.text, "red right");
        assert_eq!(results[1][0].recognition.text, "green left");
        assert_eq!(results[1][1].recognition.text, "green right");
        assert_eq!(results[0][0].detection.points.0[0], Point::new(2.0, 12.0));
    }

    struct EmptyDetector;
    impl Det for EmptyDetector {
        fn forward(&self, images: &[RgbImage]) -> anyhow::Result<Vec<Vec<Detection>>> {
            Ok(vec![Vec::new(); images.len()])
        }
    }
    struct CountingRecognizer(Rc<Cell<usize>>);
    impl Rec for CountingRecognizer {
        fn forward(&self, _: &[RgbImage]) -> anyhow::Result<Vec<Recognition>> {
            self.0.set(self.0.get() + 1);
            Ok(Vec::new())
        }
    }

    #[test]
    fn empty_images_and_empty_detections_skip_recognition() {
        let calls = Rc::new(Cell::new(0));
        let pipeline = OcrPipeline::new(
            Box::new(EmptyDetector),
            Box::new(CountingRecognizer(calls.clone())),
        );
        let results = pipeline
            .forward(&[
                DynamicImage::new_rgb8(10, 10),
                DynamicImage::new_rgb8(10, 10),
            ])
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(Vec::is_empty));
        assert!(pipeline.forward(&[]).unwrap().is_empty());
        let results = pipeline
            .forward(&[DynamicImage::new_rgb8(0, 0), DynamicImage::new_rgb8(0, 0)])
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(Vec::is_empty));
        assert_eq!(calls.get(), 0);
    }
    #[test]
    #[should_panic(expected = "detector must return one result list per image")]
    fn invalid_detector_result_count_remains_an_invariant_failure() {
        struct BrokenDetector;
        impl Det for BrokenDetector {
            fn forward(&self, _: &[RgbImage]) -> anyhow::Result<Vec<Vec<Detection>>> {
                Ok(Vec::new())
            }
        }
        let pipeline = OcrPipeline::new(Box::new(BrokenDetector), Box::new(Recognizer));
        let _ = pipeline.forward(&[DynamicImage::new_rgb8(40, 30)]);
    }

    #[test]
    fn batches_require_equal_image_dimensions() {
        let pipeline = OcrPipeline::new(Box::new(Detector), Box::new(Recognizer));
        let result = pipeline.forward(&[
            DynamicImage::new_rgb8(40, 30),
            DynamicImage::new_rgb8(80, 60),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn perspective_crop_replicates_source_borders() {
        let image = RgbImage::from_pixel(3, 3, Rgb([17, 80, 200]));
        let polygon = Polygon(vec![
            Point::new(-1.0, -1.0),
            Point::new(4.0, -1.0),
            Point::new(4.0, 4.0),
            Point::new(-1.0, 4.0),
        ]);
        let crop = crop_text(&image, &polygon).unwrap();
        assert_eq!(crop.dimensions(), (5, 5));
        for pixel in crop.pixels() {
            for (actual, expected) in pixel.0.into_iter().zip([17u8, 80, 200]) {
                assert!(actual.abs_diff(expected) <= 1);
            }
        }
        assert!(crop_text(&image, &Polygon(vec![Point::new(0.0, 0.0); 4])).is_none());
    }
}
