use std::num::NonZeroUsize;
use std::ops::Range;

use clipper2_rust::{EndType, JoinType, Point64, inflate_paths_64};
pub use imageproc::point::Point;

#[derive(Clone, Debug, PartialEq)]
pub struct Polygon(pub Vec<Point<f32>>);

impl Polygon {
    /// Scale between `(height, width)` shapes, round, and clip to width/height.
    pub fn rescaled(
        &self,
        from: (NonZeroUsize, NonZeroUsize),
        to: (NonZeroUsize, NonZeroUsize),
    ) -> Self {
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
            .map(|point| Point64::new(point.x as i64, point.y as i64))
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
                    .map(|point| Point::new(point.x as f32, point.y as f32))
                    .collect(),
            )
        })
        .collect()
    }

    /// Absolute area from the shoelace formula.
    pub fn area(&self) -> f32 {
        self.edges()
            .map(|(a, b)| a.x * b.y - b.x * a.y)
            .sum::<f32>()
            .abs()
            / 2.0
    }

    /// Length of the closed boundary, including the last-to-first edge.
    pub fn perimeter(&self) -> f32 {
        self.edges()
            .map(|(a, b)| (b.x - a.x).hypot(b.y - a.y))
            .sum()
    }

    fn edges(&self) -> impl Iterator<Item = (Point<f32>, Point<f32>)> + '_ {
        self.0
            .iter()
            .copied()
            .zip(self.0.iter().copied().cycle().skip(1))
    }

    pub fn aabb(&self) -> Rect {
        let mut min = self.0[0];
        let mut max = self.0[0];
        for point in &self.0[1..] {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        Rect { min, max }
    }

    pub fn translated(&self, dx: f32, dy: f32) -> Self {
        let delta = Point::new(dx, dy);
        Self(self.0.iter().map(|&point| point + delta).collect())
    }
}

fn rescale_point(
    point: Point<f32>,
    from: (NonZeroUsize, NonZeroUsize),
    to: (NonZeroUsize, NonZeroUsize),
) -> Point<f32> {
    let (height, width) = from;
    let (dest_height, dest_width) = to;
    Point::new(
        (point.x / width.get() as f32 * dest_width.get() as f32)
            .round_ties_even()
            .clamp(0.0, dest_width.get() as f32),
        (point.y / height.get() as f32 * dest_height.get() as f32)
            .round_ties_even()
            .clamp(0.0, dest_height.get() as f32),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min: Point<f32>,
    pub max: Point<f32>,
}

impl Rect {
    pub fn clamp(self, height: NonZeroUsize, width: NonZeroUsize) -> Self {
        let xmax = (width.get() - 1) as f32;
        let ymax = (height.get() - 1) as f32;
        Self {
            min: Point::new(self.min.x.clamp(0.0, xmax), self.min.y.clamp(0.0, ymax)),
            max: Point::new(self.max.x.clamp(0.0, xmax), self.max.y.clamp(0.0, ymax)),
        }
    }

    /// Round outward into half-open tensor slice ranges, ordered `[y, x]`.
    pub fn pixel_ranges(self) -> [Range<usize>; 2] {
        let xmin = self.min.x.floor() as usize;
        let ymin = self.min.y.floor() as usize;
        let xmax = self.max.x.ceil() as usize;
        let ymax = self.max.y.ceil() as usize;
        [ymin..ymax + 1, xmin..xmax + 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    #[test]
    fn round_offset_expands_both_polygon_windings() {
        let mut polygon = Polygon(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        for _ in 0..2 {
            let expanded = polygon.offset(2.0);
            assert_eq!(expanded.len(), 1);
            assert!(expanded[0].area() > polygon.area());
            assert_eq!(expanded[0].aabb().min, Point::new(-2.0, -2.0));
            assert_eq!(expanded[0].aabb().max, Point::new(12.0, 12.0));
            polygon.0.reverse();
        }
    }

    #[test]
    fn closed_boundary_and_area_do_not_depend_on_winding() {
        let mut triangle = Polygon(vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(0.0, 4.0),
        ]);
        assert_eq!(triangle.area(), 6.0);
        assert_eq!(triangle.perimeter(), 12.0);
        triangle.0.reverse();
        assert_eq!(triangle.area(), 6.0);
        assert_eq!(triangle.perimeter(), 12.0);
    }

    #[test]
    fn crop_bounds_clip_and_include_the_last_pixel() {
        let polygon = Polygon(vec![
            Point::new(-2.5, 1.2),
            Point::new(4.1, 7.5),
            Point::new(0.0, 3.0),
        ]);
        assert_eq!(
            polygon.aabb().clamp(nz(6), nz(8)).pixel_ranges(),
            [1..6, 0..6]
        );
    }

    #[test]
    fn rescaling_uses_each_axis_and_rounds_ties_to_even() {
        let polygon = Polygon(vec![
            Point::new(-1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(3.0, 3.0),
            Point::new(30.0, 20.0),
        ]);
        assert_eq!(
            polygon.rescaled((nz(10), nz(20)), (nz(5), nz(10))),
            Polygon(vec![
                Point::new(0.0, 0.0),
                Point::new(0.0, 0.0),
                Point::new(2.0, 2.0),
                Point::new(10.0, 5.0)
            ])
        );
    }

    #[test]
    fn rescaling_preserves_corners_when_rectangle_becomes_parallelogram() {
        let polygon = Polygon(vec![
            Point::new(3.0, 1.0),
            Point::new(5.0, 3.0),
            Point::new(3.0, 5.0),
            Point::new(1.0, 3.0),
        ]);
        let scaled = polygon.rescaled((nz(6), nz(6)), (nz(6), nz(12)));
        assert_eq!(
            scaled,
            Polygon(vec![
                Point::new(6.0, 1.0),
                Point::new(10.0, 3.0),
                Point::new(6.0, 5.0),
                Point::new(2.0, 3.0)
            ])
        );
        let [a, b, c] = [scaled.0[0], scaled.0[1], scaled.0[2]];
        let dot = (b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y);
        assert_ne!(dot, 0.0, "adjacent edges are no longer perpendicular");
    }
}
