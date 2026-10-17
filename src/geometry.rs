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
