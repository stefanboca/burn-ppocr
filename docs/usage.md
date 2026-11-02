# Usage

[Project overview](../README.md)

Run commands from the repository root.

## Run the example

From the repository root:

```sh
cargo run --release --example ocr --features pp_ocrv6_medium -- image.png
```

The example prints recognition scores and text. It accepts multiple images
when they have the same dimensions, and uses FP16 inference. It needs a Burn
device with support for that configuration. A Nix development shell is
available with CUDA and Vulkan libraries via `nix develop`.

## Use from Rust

Enable `pp_ocrv6_medium`, `pp_ocrv6_small`, or `pp_ocrv6_tiny`. No model is
enabled by default.

```rust
use burn::tensor::Device;
use burn_ppocr::OcrProcessor;

let processor = OcrProcessor::pp_ocrv6_medium(Device::default());
let images = vec![image::open("image.png")?];
let results = processor.forward(&images)?;
```

The build imports `det.onnx` and `rec.onnx` from the corresponding directory
under `models/`. The `embed` feature includes the generated weights in the
binary; otherwise the model loader uses files in the build output directory.

This is an early crate with a changing API. See
[the OCR example](../examples/ocr.rs) for device configuration and
[the pipeline](../src/ocr.rs) for the processing stages. PaddleOCR model notices
and licenses are included under `models/`.
