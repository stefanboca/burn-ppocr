use burn::prelude::*;
use image::{GrayImage, Luma, RgbImage};
use imageproc::point::Point;

use super::preprocess::DetPreProcess;
use super::{Det, DetModel, Detection};
use crate::geometry::Polygon;

pub struct DbDetector<M> {
    model: M,
    device: Device,
    preprocess: DetPreProcess,
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
}

impl<M: DetModel> DbDetector<M> {
    pub fn new(model: M, device: Device) -> Self {
        Self {
            model,
            device,
            preprocess: DetPreProcess::default(),
            thresh: 0.2,
            box_thresh: 0.4,
            max_candidates: 3000,
            unclip_ratio: 1.4,
        }
    }

    fn postprocess(&self, maps: Tensor<4>, src_shape: (usize, usize)) -> Vec<Detection> {
        let [batch, channels, height, width] = maps.dims();
        assert_eq!(batch, 1);
        assert_eq!(channels, 1);
        assert!(height > 0 && width > 0);
        // Contours and geometry are CPU operations. Read the probability map once,
        // rather than transferring a separate polygon mask for every candidate.
        let pred = maps
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("detection probabilities must be convertible to f32");
        assert!(
            pred.iter().all(|value| value.is_finite()),
            "detector returned nonfinite probabilities"
        );
        self.boxes_from_bitmap(&pred, (height, width), src_shape)
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

impl<M: DetModel> Det for DbDetector<M> {
    fn forward(&self, image: &RgbImage) -> Vec<Detection> {
        if image.width() == 0 || image.height() == 0 {
            return Vec::new();
        }
        let input = self.preprocess.forward(image, &self.device);
        let maps = self.model.forward(input);
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
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let [
        mut left_top,
        mut left_bottom,
        mut right_top,
        mut right_bottom,
    ] = points;
    if left_top[1] > left_bottom[1] {
        std::mem::swap(&mut left_top, &mut left_bottom);
    }
    if right_top[1] > right_bottom[1] {
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
                    .map(|point| [point.x as f32, point.y as f32])
                    .collect(),
            )
        })
        .filter(|polygon| !polygon.0.is_empty())
        .collect()
}

fn min_area_rect(contour: &Polygon) -> ([[f32; 2]; 4], f32) {
    let hull = imageproc::geometry::convex_hull(
        contour
            .0
            .iter()
            .map(|&[x, y]| Point::new(x as f64, y as f64))
            .collect::<Vec<_>>(),
    );
    if hull.len() < 2 {
        let point = hull
            .first()
            .map_or([0.0; 2], |point| [point.x as f32, point.y as f32]);
        return ([point; 4], 0.0);
    }
    // imageproc's min_area_rect rounds corners to integers, even for float inputs.
    // Project the hull onto each edge to retain OpenCV-style floating-point corners.
    let mut best_area = f64::INFINITY;
    let mut best = ([[0.0; 2]; 4], 0.0);
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
                .map(|[x, y]| [(x * ux - y * uy) as f32, (x * uy + y * ux) as f32]);
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
        .map(|&[x, y]| Point::new(x as i32, y as i32))
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
        fn forward(&self, _: Tensor<4>) -> Tensor<4> {
            self.0.clone()
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
        let detector = DbDetector::new(MapModel(maps), Device::flex());
        let detections = detector.forward(&RgbImage::new(160, 96));
        assert_eq!(detections.len(), 1);
        let detection = &detections[0];
        assert_eq!(detection.points.0.len(), 4);
        assert!((detection.score - 0.9).abs() < 1e-4);
        let bounds = detection.points.aabb();
        assert!(bounds.min[0] < 20.0 && bounds.max[0] > 80.0);
        assert!(bounds.min[1] < 20.0 && bounds.max[1] > 50.0);
    }

    #[test]
    fn minimum_rectangle_keeps_fractional_rotated_corners() {
        let polygon = Polygon(vec![[0.0, 0.0], [4.0, 2.0], [3.0, 4.0], [-1.0, 2.0]]);
        let (rectangle, short_side) = get_mini_boxes(&polygon);
        assert!((rectangle.area() - 10.0).abs() < 1e-4);
        assert!((short_side - 5.0f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn threshold_comparison_is_strict() {
        let maps = Tensor::full([1, 1, 16, 16], 0.2, &Device::flex());
        let detector = DbDetector::new(MapModel(maps), Device::flex());
        assert!(detector.forward(&RgbImage::new(16, 16)).is_empty());
    }
}
