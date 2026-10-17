#[cfg(feature = "pp_ocrv6_tiny")]
use burn_onnx::ModelGen;

fn main() {
    #[cfg(feature = "pp_ocrv6_tiny")]
    ModelGen::new()
        .input("models/pp_ocrv6_tiny/det.onnx")
        .input("models/pp_ocrv6_tiny/rec.onnx")
        .out_dir("models/pp_ocrv6_tiny")
        .run_from_script();
}
