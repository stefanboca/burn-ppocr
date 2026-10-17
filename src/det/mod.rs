pub mod postprocess;
pub mod preprocess;

use burn::module::{Module, ModuleDisplay};
use burn::tensor::Tensor;
pub use postprocess::{DbPostProcess, DbPostProcessConfig};

use crate::geometry::Polygon;

/// The tensor interface implemented by generated detection models.
pub trait DetModel: Module + ModuleDisplay {
    fn forward(&self, images: Tensor<4>) -> Tensor<4>;
}

/// One text region in source-image coordinates, with its mean prediction score.
#[derive(Clone, Debug)]
pub struct Detection {
    pub points: Polygon,
    pub score: f32,
}
