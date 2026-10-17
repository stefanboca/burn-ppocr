use burn_onnx::ModelGen;

fn main() {
    #[cfg(feature = "pp_ocrv6_tiny_det")]
    ModelGen::new()
        .input("models/PP-OCRv6_tiny_det.onnx")
        .out_dir("model/")
        .run_from_script();

    #[cfg(feature = "pp_ocrv6_tiny_rec")]
    ModelGen::new()
        .input("models/PP-OCRv6_tiny_rec.onnx")
        .out_dir("model/")
        .run_from_script();
}
