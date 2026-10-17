use std::ops::Range;

use clipper2_rust::{EndType, JoinType, Point64, inflate_paths_64};

#[derive(Clone, Debug, PartialEq)]
pub struct Polygon(pub Vec<[f32; 2]>);

impl Polygon {
    /// Scale between `(height, width)` shapes, round, and clip to width/height.
    pub fn rescaled(&self, from: (usize, usize), to: (usize, usize)) -> Self {
        assert!(from.0 > 0 && from.1 > 0);
        Self(
            self.0
                .iter()
                .map(|&point| rescale_point(point, from, to))
                .collect(),
        )
    }

    /// Offset the closed polygon by `distance`, using round joins.
    pub fn offset(&self, distance: f32) -> Vec<Self> {
        if self.0.len() < 3 || !distance.is_finite() {
            return Vec::new();
        }
        let path = self
            .0
            .iter()
            .map(|&[x, y]| Point64::new(x as i64, y as i64))
            .collect();
        inflate_paths_64(
            &vec![path],
            distance as f64,
            JoinType::Round,
            EndType::Polygon,
            2.0,
            0.25,
        )
        .into_iter()
        .map(|path| {
            Self(
                path.into_iter()
                    .map(|point| [point.x as f32, point.y as f32])
                    .collect(),
            )
        })
        .collect()
    }

    /// Absolute area from the shoelace formula.
    pub fn area(&self) -> f32 {
        self.edges()
            .map(|([x1, y1], [x2, y2])| x1 * y2 - x2 * y1)
            .sum::<f32>()
            .abs()
            / 2.0
    }

    /// Length of the closed boundary, including the last-to-first edge.
    pub fn perimeter(&self) -> f32 {
        self.edges()
            .map(|([x1, y1], [x2, y2])| (x2 - x1).hypot(y2 - y1))
            .sum()
    }

    fn edges(&self) -> impl Iterator<Item = ([f32; 2], [f32; 2])> + '_ {
        self.0
            .iter()
            .copied()
            .zip(self.0.iter().copied().cycle().skip(1))
    }

    pub fn aabb(&self) -> Rect {
        let mut min = self.0[0];
        let mut max = self.0[0];
        for &[x, y] in &self.0[1..] {
            min = [min[0].min(x), min[1].min(y)];
            max = [max[0].max(x), max[1].max(y)];
        }
        Rect { min, max }
    }

    pub fn translated(&self, dx: f32, dy: f32) -> Self {
        Self(self.0.iter().map(|&[x, y]| [x + dx, y + dy]).collect())
    }
}

fn rescale_point([x, y]: [f32; 2], from: (usize, usize), to: (usize, usize)) -> [f32; 2] {
    let (height, width) = from;
    let (dest_height, dest_width) = to;
    [
        (x / width as f32 * dest_width as f32)
            .round_ties_even()
            .clamp(0.0, dest_width as f32),
        (y / height as f32 * dest_height as f32)
            .round_ties_even()
            .clamp(0.0, dest_height as f32),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Rect {
    pub fn clamp(self, height: usize, width: usize) -> Self {
        let xmax = (width - 1) as f32;
        let ymax = (height - 1) as f32;
        Self {
            min: [self.min[0].clamp(0.0, xmax), self.min[1].clamp(0.0, ymax)],
            max: [self.max[0].clamp(0.0, xmax), self.max[1].clamp(0.0, ymax)],
        }
    }

    /// Round outward into half-open tensor slice ranges, ordered `[y, x]`.
    pub fn pixel_ranges(self) -> [Range<usize>; 2] {
        let xmin = self.min[0].floor() as usize;
        let ymin = self.min[1].floor() as usize;
        let xmax = self.max[0].ceil() as usize;
        let ymax = self.max[1].ceil() as usize;
        [ymin..ymax + 1, xmin..xmax + 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_offset_expands_both_polygon_windings() {
        let mut polygon = Polygon(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        for _ in 0..2 {
            let expanded = polygon.offset(2.0);
            assert_eq!(expanded.len(), 1);
            assert!(expanded[0].area() > polygon.area());
            assert_eq!(expanded[0].aabb().min, [-2.0, -2.0]);
            assert_eq!(expanded[0].aabb().max, [12.0, 12.0]);
            polygon.0.reverse();
        }
    }

    #[test]
    fn closed_boundary_and_area_do_not_depend_on_winding() {
        let mut triangle = Polygon(vec![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]);
        assert_eq!(triangle.area(), 6.0);
        assert_eq!(triangle.perimeter(), 12.0);
        triangle.0.reverse();
        assert_eq!(triangle.area(), 6.0);
        assert_eq!(triangle.perimeter(), 12.0);
    }

    #[test]
    fn crop_bounds_clip_and_include_the_last_pixel() {
        let polygon = Polygon(vec![[-2.5, 1.2], [4.1, 7.5], [0.0, 3.0]]);
        assert_eq!(polygon.aabb().clamp(6, 8).pixel_ranges(), [1..6, 0..6]);
    }

    #[test]
    fn rescaling_uses_each_axis_and_rounds_ties_to_even() {
        let polygon = Polygon(vec![[-1.0, -1.0], [1.0, 1.0], [3.0, 3.0], [30.0, 20.0]]);
        assert_eq!(
            polygon.rescaled((10, 20), (5, 10)),
            Polygon(vec![[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [10.0, 5.0]])
        );
    }

    #[test]
    fn rescaling_preserves_corners_when_rectangle_becomes_parallelogram() {
        let polygon = Polygon(vec![[3.0, 1.0], [5.0, 3.0], [3.0, 5.0], [1.0, 3.0]]);
        let scaled = polygon.rescaled((6, 6), (6, 12));
        assert_eq!(
            scaled,
            Polygon(vec![[6.0, 1.0], [10.0, 3.0], [6.0, 5.0], [2.0, 3.0]])
        );
        let [a, b, c] = [scaled.0[0], scaled.0[1], scaled.0[2]];
        let dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]);
        assert_ne!(dot, 0.0, "adjacent edges are no longer perpendicular");
    }
}
