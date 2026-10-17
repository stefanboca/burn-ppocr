use burn::prelude::*;
use image::{GrayImage, Luma, RgbImage};

use super::preprocess::DetPreProcess;
use super::{Det, DetModel, Detection};
use crate::geometry::{Point, Polygon};

pub struct DbDetector {
    pub model: Box<dyn DetModel>,
    pub device: Device,
    pub preprocess: DetPreProcess,
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
}

impl DbDetector {
    fn postprocess(&self, maps: Tensor<4>, src_shape: (usize, usize)) -> Vec<Vec<Detection>> {
        let [_, channels, height, width] = maps.dims();
        assert_eq!(channels, 1);
        assert!(height > 0 && width > 0);
        // Contours and geometry are CPU operations. Read the batch once,
        // rather than transferring a separate polygon mask for every candidate.
        let pred = maps
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("detection probabilities must be convertible to f32");
        assert!(
            pred.iter().all(|value| value.is_finite()),
            "detector returned nonfinite probabilities"
        );
        pred.chunks_exact(height * width)
            .map(|pred| self.boxes_from_bitmap(pred, (height, width), src_shape))
            .collect()
    }

    fn boxes_from_bitmap(
        &self,
        pred: &[f32],
        shape: (usize, usize),
        src_shape: (usize, usize),
    ) -> Vec<Detection> {
        let (height, width) = shape;
        let mask = GrayImage::from_raw(
            width as u32,
            height as u32,
            pred.iter()
                .map(|&value| u8::from(value > self.thresh))
                .collect(),
        )
        .unwrap();
        let mut detections = Vec::new();
        for contour in find_contours(&mask).into_iter().take(self.max_candidates) {
            let (points, short_side) = get_mini_boxes(&contour);
            if short_side < 3.0 {
                continue;
            }
            let score = polygon_score(pred, shape, &points);
            if score < self.box_thresh {
                continue;
            }
            let expanded = unclip(&points, self.unclip_ratio);
            let [expanded] = expanded.as_slice() else {
                continue;
            };
            if expanded.0.is_empty() {
                continue;
            }
            let (points, short_side) = get_mini_boxes(expanded);
            if short_side < 5.0 {
                continue;
            }
            detections.push(Detection {
                points: points.rescaled(shape, src_shape),
                score,
            });
        }
        detections
    }
}

impl Det for DbDetector {
    fn forward(&self, images: &[RgbImage]) -> Vec<Vec<Detection>> {
        let Some(image) = images.first() else {
            return Vec::new();
        };
        assert!(
            images
                .iter()
                .all(|other| other.dimensions() == image.dimensions()),
            "detector images must have the same dimensions"
        );
        if image.width() == 0 || image.height() == 0 {
            return vec![Vec::new(); images.len()];
        }
        let input = self.preprocess.forward(images, &self.device);
        let maps = self.model.forward(input);
        assert_eq!(maps.dims()[0], images.len(), "detector batch size mismatch");
        self.postprocess(maps, (image.height() as usize, image.width() as usize))
    }
}

fn polygon_score(pred: &[f32], shape: (usize, usize), polygon: &Polygon) -> f32 {
    let [ys, xs] = polygon.aabb().clamp(shape.0, shape.1).pixel_ranges();
    let local = polygon.translated(-(xs.start as f32), -(ys.start as f32));
    let mask = fill_poly(&local, (ys.len(), xs.len()));
    let mut sum = 0.0;
    let mut count = 0;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] != 0 {
            sum += pred[(ys.start + y as usize) * shape.1 + xs.start + x as usize];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn get_mini_boxes(contour: &Polygon) -> (Polygon, f32) {
    let (mut points, short_side) = min_area_rect(contour);
    points.sort_by(|a, b| a.x.total_cmp(&b.x));
    let [
        mut left_top,
        mut left_bottom,
        mut right_top,
        mut right_bottom,
    ] = points;
    if left_top.y > left_bottom.y {
        std::mem::swap(&mut left_top, &mut left_bottom);
    }
    if right_top.y > right_bottom.y {
        std::mem::swap(&mut right_top, &mut right_bottom);
    }
    (
        Polygon(vec![left_top, right_top, right_bottom, left_bottom]),
        short_side,
    )
}

fn unclip(polygon: &Polygon, ratio: f32) -> Vec<Polygon> {
    let perimeter = polygon.perimeter();
    if perimeter == 0.0 {
        return Vec::new();
    }
    polygon.offset(polygon.area() * ratio / perimeter)
}

fn find_contours(mask: &GrayImage) -> Vec<Polygon> {
    imageproc::contours::find_contours::<i32>(mask)
        .into_iter()
        .map(|contour| {
            Polygon(
                contour
                    .points
                    .into_iter()
                    .map(|point| Point::new(point.x as f32, point.y as f32))
                    .collect(),
            )
        })
        .filter(|polygon| !polygon.0.is_empty())
        .collect()
}

fn min_area_rect(contour: &Polygon) -> ([Point<f32>; 4], f32) {
    let hull = imageproc::geometry::convex_hull(
        contour
            .0
            .iter()
            .map(|point| Point::new(point.x as f64, point.y as f64))
            .collect::<Vec<_>>(),
    );
    if hull.len() < 2 {
        let point = hull.first().map_or(Point::new(0.0, 0.0), |point| {
            Point::new(point.x as f32, point.y as f32)
        });
        return ([point; 4], 0.0);
    }
    // imageproc's min_area_rect rounds corners to integers, even for float inputs.
    // Project the hull onto each edge to retain OpenCV-style floating-point corners.
    let mut best_area = f64::INFINITY;
    let mut best = ([Point::new(0.0, 0.0); 4], 0.0);
    for (a, b) in hull.iter().zip(hull.iter().cycle().skip(1)) {
        let length = (b.x - a.x).hypot(b.y - a.y);
        if length == 0.0 {
            continue;
        }
        let (ux, uy) = ((b.x - a.x) / length, (b.y - a.y) / length);
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for point in &hull {
            let x = point.x * ux + point.y * uy;
            let y = -point.x * uy + point.y * ux;
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        let (width, height) = (xmax - xmin, ymax - ymin);
        let area = width * height;
        if area < best_area {
            best_area = area;
            let points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                .map(|[x, y]| Point::new((x * ux - y * uy) as f32, (x * uy + y * ux) as f32));
            best = (points, width.min(height) as f32);
        }
    }
    best
}

fn fill_poly(polygon: &Polygon, shape: (usize, usize)) -> GrayImage {
    let mut mask = GrayImage::new(shape.1 as u32, shape.0 as u32);
    let mut points: Vec<_> = polygon
        .0
        .iter()
        .map(|point| Point::new(point.x as i32, point.y as i32))
        .collect();
    if points.len() > 1 && points.first() == points.last() {
        points.pop();
    }
    if points.len() >= 3 {
        imageproc::drawing::draw_polygon_mut(&mut mask, &points, Luma([1]));
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MapModel(Tensor<4>);

    impl DetModel for MapModel {
        fn forward(&self, images: Tensor<4>) -> Tensor<4> {
            assert_eq!(images.dims()[0], self.0.dims()[0]);
            self.0.clone()
        }
    }

    fn detector(maps: Tensor<4>) -> DbDetector {
        DbDetector {
            model: Box::new(MapModel(maps)),
            device: Device::flex(),
            preprocess: DetPreProcess { max_side: 960 },
            thresh: 0.2,
            box_thresh: 0.4,
            max_candidates: 3000,
            unclip_ratio: 1.4,
        }
    }

    #[test]
    fn detector_extracts_expands_and_rescales_a_text_region() {
        let mut probabilities = vec![0.0f32; 80 * 48];
        for y in 10..=25 {
            for x in 10..=40 {
                probabilities[y * 80 + x] = 0.9;
            }
        }
        let maps = Tensor::from_data(
            TensorData::new(probabilities, [1, 1, 48, 80]),
            &Device::flex(),
        );
        let detector = detector(maps);
        let detections = detector.forward(&[RgbImage::new(160, 96)]);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].len(), 1);
        let detection = &detections[0][0];
        assert_eq!(detection.points.0.len(), 4);
        assert!((detection.score - 0.9).abs() < 1e-4);
        let bounds = detection.points.aabb();
        assert!(bounds.min.x < 20.0 && bounds.max.x > 80.0);
        assert!(bounds.min.y < 20.0 && bounds.max.y > 50.0);
    }

    #[test]
    fn detector_preserves_each_map_and_empty_results_in_a_batch() {
        let mut probabilities = vec![0.0f32; 3 * 32 * 32];
        for y in 8..=22 {
            for x in 6..=16 {
                probabilities[y * 32 + x] = 0.9;
            }
        }
        for y in 5..=18 {
            for x in 14..=24 {
                probabilities[32 * 32 + y * 32 + x] = 0.8;
            }
        }
        let maps = Tensor::from_data(
            TensorData::new(probabilities, [3, 1, 32, 32]),
            &Device::flex(),
        );
        let detector = detector(maps);
        let images = vec![RgbImage::new(64, 96); 3];
        let results = detector.forward(&images);
        assert_eq!(results.iter().map(Vec::len).collect::<Vec<_>>(), [1, 1, 0]);
        assert!((results[0][0].score - 0.9).abs() < 1e-4);
        assert!((results[1][0].score - 0.8).abs() < 1e-4);
        assert!(results[0][0].points.aabb().min.x < results[1][0].points.aabb().min.x);
        assert!(detector.forward(&[]).is_empty());
        let empty = detector.forward(&vec![RgbImage::new(0, 0); 3]);
        assert_eq!(empty.len(), 3);
        assert!(empty.iter().all(Vec::is_empty));
    }

    #[test]
    fn minimum_rectangle_keeps_fractional_rotated_corners() {
        let polygon = Polygon(vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 2.0),
            Point::new(3.0, 4.0),
            Point::new(-1.0, 2.0),
        ]);
        let (rectangle, short_side) = get_mini_boxes(&polygon);
        assert!((rectangle.area() - 10.0).abs() < 1e-4);
        assert!((short_side - 5.0f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn threshold_comparison_is_strict() {
        let maps = Tensor::full([1, 1, 16, 16], 0.2, &Device::flex());
        let detector = detector(maps);
        assert!(detector.forward(&[RgbImage::new(16, 16)])[0].is_empty());
    }
}
