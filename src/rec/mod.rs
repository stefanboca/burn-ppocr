pub mod ctc;
pub mod decode;
pub mod preprocess;

use burn::module::{Module, ModuleDisplay};
use burn::tensor::Tensor;
pub use ctc::CtcRecognizer;
pub use decode::CtcDecoder;

/// The tensor interface implemented by generated recognition models.
pub trait RecModel: Module + ModuleDisplay {
    fn forward(&self, images: Tensor<4>) -> Tensor<3>;
}

/// A decoded text line and the mean confidence of its emitted CTC tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Recognition {
    pub text: String,
    pub score: f32,
}
