use std::num::NonZeroUsize;

use anyhow::{Context, ensure};
use burn::config::Config;
use burn::tensor::Tensor;
use image::{GrayImage, Luma};
use imageproc::contours::{BorderType, find_contours};

use super::Detection;
use crate::geometry::{Point, Polygon};

#[derive(Config, Debug)]
pub struct DbPostProcessConfig {
    #[config(default = 0.2)]
    pub thresh: f32,
    #[config(default = 0.45)]
    pub box_thresh: f32,
    #[config(default = 3000)]
    pub max_candidates: usize,
    #[config(default = 1.4)]
    pub unclip_ratio: f32,
}

impl DbPostProcessConfig {
    pub fn init(&self) -> DbPostProcess {
        DbPostProcess {
            thresh: self.thresh,
            box_thresh: self.box_thresh,
            max_candidates: self.max_candidates,
            unclip_ratio: self.unclip_ratio,
        }
    }
}

/// DB probability maps to scored text polygons, independent of the detection model.
#[derive(Clone, Debug)]
pub struct DbPostProcess {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
}

impl DbPostProcess {
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.postprocess")
    )]
    pub fn forward(
        &self,
        maps: Tensor<4>,
        src_shape: (NonZeroUsize, NonZeroUsize),
    ) -> anyhow::Result<Vec<Vec<Detection>>> {
        let channels = maps.dims()[1];
        ensure!(channels == 1);
        maps.squeeze_dim::<3>(1)
            .iter_dim(0)
            .map(|map| self.boxes_from_bitmap(map.squeeze_dim(0), src_shape))
            .collect()
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.boxes_from_bitmap")
    )]
    pub fn boxes_from_bitmap(
        &self,
        pred: Tensor<2>,
        src_shape: (NonZeroUsize, NonZeroUsize),
    ) -> anyhow::Result<Vec<Detection>> {
        let [height, width] = pred.dims();
        let shape = (
            NonZeroUsize::new(height).unwrap(),
            NonZeroUsize::new(width).unwrap(),
        );
        let pred = pred
            .try_into_vec_as::<f32>()
            .context("reading detector probabilities")?;
        Ok(self.boxes_from_probabilities(&pred, shape, src_shape))
    }

    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.boxes_from_probabilities")
    )]
    pub fn boxes_from_probabilities(
        &self,
        pred: &[f32],
        shape: (NonZeroUsize, NonZeroUsize),
        src_shape: (NonZeroUsize, NonZeroUsize),
    ) -> Vec<Detection> {
        let mask = GrayImage::from_raw(
            shape.1.get() as u32,
            shape.0.get() as u32,
            pred.iter()
                .map(|&value| u8::from(value > self.thresh))
                .collect(),
        )
        .unwrap();
        let mut detections = Vec::new();
        for contour in find_contours::<i32>(&mask)
            .into_iter()
            .filter(|contour| contour.border_type == BorderType::Outer && contour.parent.is_none())
            .take(self.max_candidates)
        {
            let (points, short_side) = get_mini_boxes(&contour.points);
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
            let (points, short_side) = get_mini_boxes(&expanded.0);
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

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "det.polygon_score")
)]
pub fn polygon_score(pred: &[f32], shape: (NonZeroUsize, NonZeroUsize), polygon: &Polygon) -> f32 {
    let [ys, xs] = polygon.aabb().clamp(shape.0, shape.1).pixel_ranges();
    let local = polygon.translated(-(xs.start as f32), -(ys.start as f32));
    let mask = fill_poly(&local, (ys.len(), xs.len()));
    let mut sum = 0.0;
    let mut count = 0;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] != 0 {
            sum += pred[(ys.start + y as usize) * shape.1.get() + xs.start + x as usize];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "det.get_mini_boxes")
)]
fn get_mini_boxes<T: Copy + Into<f64>>(contour: &[Point<T>]) -> (Polygon, f32) {
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

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "det.unclip")
)]
fn unclip(polygon: &Polygon, ratio: f32) -> Vec<Polygon> {
    let perimeter = polygon.perimeter();
    if perimeter == 0.0 {
        return Vec::new();
    }
    polygon.offset(polygon.area() * ratio / perimeter)
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "geometry.min_area_rect")
)]
pub fn min_area_rect<T: Copy + Into<f64>>(contour: &[Point<T>]) -> ([Point<f32>; 4], f32) {
    let hull = imageproc::geometry::convex_hull(
        contour
            .iter()
            .map(|point| Point::new(point.x.into(), point.y.into()))
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

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip_all, name = "geometry.fill_poly")
)]
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
