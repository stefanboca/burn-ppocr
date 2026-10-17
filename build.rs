use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    ModelGen::new()
        .input("models/PP-OCRv6_tiny_det.onnx")
        .out_dir("model/")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();

    ModelGen::new()
        .input("models/PP-OCRv6_tiny_rec.onnx")
        .out_dir("model/")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}
