#[cfg(feature = "pp_ocrv6_tiny")]
use burn_onnx::ModelGen;

fn main() {
    #[cfg(feature = "pp_ocrv6_tiny")]
    for input in [
        "models/PP-OCRv6_tiny_det.onnx",
        "models/PP-OCRv6_tiny_rec.onnx",
    ] {
        ModelGen::new()
            .input(input)
            .out_dir("model/")
            .run_from_script();
    }
}
