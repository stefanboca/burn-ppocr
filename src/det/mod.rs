pub mod db;
pub mod preprocess;

use burn::tensor::Tensor;
pub use db::DbDetector;
use image::RgbImage;

use crate::geometry::Polygon;

/// A detector including its model-specific input and output processing.
/// All images in a batch must have the same dimensions. Results follow input order.
pub trait Det {
    fn forward(&self, images: &[RgbImage]) -> anyhow::Result<Vec<Vec<Detection>>>;
}

/// The tensor interface implemented by generated detection models.
pub trait DetModel {
    fn forward(&self, images: Tensor<4>) -> Tensor<4>;
}

/// One text region in source-image coordinates, with its mean prediction score.
#[derive(Clone, Debug)]
pub struct Detection {
    pub points: Polygon,
    pub score: f32,
}
