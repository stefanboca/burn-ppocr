#![allow(unused)]

use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    let load_strategy = if cfg!(feature = "embed") {
        LoadStrategy::Embedded
    } else {
        LoadStrategy::File
    };

    #[cfg(feature = "pp_ocrv6_tiny")]
    ModelGen::new()
        .input("models/pp_ocrv6_tiny/det.onnx")
        .input("models/pp_ocrv6_tiny/rec.onnx")
        .out_dir("models/pp_ocrv6_tiny")
        .load_strategy(load_strategy)
        .run_from_script();

    #[cfg(feature = "pp_ocrv6_small")]
    ModelGen::new()
        .input("models/pp_ocrv6_small/det.onnx")
        .input("models/pp_ocrv6_small/rec.onnx")
        .out_dir("models/pp_ocrv6_small")
        .load_strategy(load_strategy)
        .run_from_script();

    #[cfg(feature = "pp_ocrv6_medium")]
    ModelGen::new()
        .input("models/pp_ocrv6_medium/det.onnx")
        .input("models/pp_ocrv6_medium/rec.onnx")
        .out_dir("models/pp_ocrv6_medium")
        .load_strategy(load_strategy)
        .run_from_script();
}
