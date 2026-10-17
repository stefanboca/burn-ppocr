use image::RgbImage;

use crate::{Detection, Recognition};

#[cfg(feature = "pp_ocrv6_tiny")]
pub mod pp_ocrv6_tiny;
#[cfg(feature = "pp_ocrv6_tiny")]
pub use pp_ocrv6_tiny::{PpOcrV6TinyDet, PpOcrV6TinyRec};

/// A detector including its model-specific input and output processing.
pub trait Det {
    fn forward(&self, image: &RgbImage) -> Vec<Detection>;
}

/// A recognizer including orientation, normalization, batching, and decoding.
/// Results must correspond to the crops in their original order.
pub trait Rec {
    fn forward(&self, crops: &[RgbImage]) -> Vec<Recognition>;
}
