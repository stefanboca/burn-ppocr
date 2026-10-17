use burn::prelude::*;

use crate::geometry::Polygon;

/// One text region in source-image coordinates, with its mean prediction score.
#[derive(Debug)]
pub struct Detection {
    pub points: Polygon,
    pub score: f32,
}

#[derive(Debug)]
pub struct DbPostProcess {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: u16,
    pub unclip_ratio: f32,
}

impl Default for DbPostProcess {
    fn default() -> Self {
        Self {
            thresh: 0.2,
            box_thresh: 0.4,
            max_candidates: 3000,
            unclip_ratio: 1.4,
        }
    }
}

impl DbPostProcess {
    pub fn forward(&self, maps: Tensor<4>, src_shapes: Vec<(usize, usize)>) -> Vec<Vec<Detection>> {
        let [batch, channels, height, width] = maps.dims();
        assert_eq!(batch, src_shapes.len());
        assert_eq!(channels, 1);
        assert!(height > 0 && width > 0);
        assert!(src_shapes.iter().all(|&(h, w)| h > 0 && w > 0));

        let segmentation = maps.clone().greater_elem(self.thresh);

        let max_candidates = usize::from(self.max_candidates);
        maps.iter_dim(0)
            .zip(segmentation.iter_dim(0))
            .zip(src_shapes)
            .map(move |((pred, mask), src_shape)| {
                let contours = find_contours(mask.squeeze_dims::<2>(&[0, 1]))
                    .into_iter()
                    .take(max_candidates)
                    .collect();
                self.boxes_from_bitmap(pred.squeeze_dims::<2>(&[0, 1]), contours, src_shape)
            })
            .collect()
    }

    fn boxes_from_bitmap(
        &self,
        pred: Tensor<2>,
        contours: Vec<Polygon>,
        src_shape: (usize, usize),
    ) -> Vec<Detection> {
        const MIN_SIZE: f32 = 3.0;

        let [height, width] = pred.dims();
        let mut detections = Vec::new();
        for contour in contours {
            let (points, short_side) = get_mini_boxes(&contour);
            if short_side < MIN_SIZE {
                continue;
            }
            let score = polygon_score(pred.clone(), &points);
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
            if short_side < MIN_SIZE + 2.0 {
                continue;
            }

            detections.push(Detection {
                points: points.rescaled((height, width), src_shape),
                score,
            });
        }
        detections
    }
}

fn polygon_score(pred: Tensor<2>, polygon: &Polygon) -> f32 {
    let [height, width] = pred.dims();
    let [ys, xs] = polygon.aabb().clamp(height, width).pixel_ranges();
    let local_polygon = polygon.translated(-(xs.start as f32), -(ys.start as f32));

    let mask = fill_poly(&local_polygon, (ys.len(), xs.len()), &pred.device());
    let count = mask.clone().float().sum().clamp_min(1.0);
    let values = pred.slice([ys, xs]).mask_fill(mask.bool_not(), 0.0);
    (values.sum() / count).into_scalar::<f32>()
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

fn find_contours(mask: Tensor<2, Bool>) -> Vec<Polygon> {
    todo!("cv2.findContours")
}

fn min_area_rect(contour: &Polygon) -> ([[f32; 2]; 4], f32) {
    todo!("cv2.minAreaRect followed by cv2.boxPoints")
}

fn fill_poly(polygon: &Polygon, shape: (usize, usize), device: &Device) -> Tensor<2, Bool> {
    todo!("cv2.fillPoly")
}
