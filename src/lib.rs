pub mod det;
pub mod geometry;
pub mod models;
pub mod ocr;
pub mod rec;

pub use det::Detection;
pub use ocr::{OcrProcessor, OcrResult};
pub use rec::Recognition;
pub mod crop;
