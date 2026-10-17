pub mod ctc;
pub mod preprocess;

use burn::tensor::Tensor;
pub use ctc::CtcRecognizer;
use image::RgbImage;

/// A recognizer including orientation, normalization, batching, and decoding.
/// Results must correspond to the crops in their original order.
pub trait Rec {
    fn forward(&self, crops: &[RgbImage]) -> anyhow::Result<Vec<Recognition>>;
}

/// The tensor interface implemented by generated recognition models.
pub trait RecModel {
    fn forward(&self, images: Tensor<4>) -> Tensor<3>;
}

/// A decoded text line and the mean confidence of its emitted CTC tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Recognition {
    pub text: String,
    pub score: f32,
}
