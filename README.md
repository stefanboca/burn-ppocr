# burn-ppocr

A Rust crate for running PP-OCRv6 with the Burn deep learning framework. It
provides an OCR pipeline that finds text in images and recognizes its contents.

The repository contains the model integration, image processing, and examples
needed to use the pipeline from Rust. The pretrained models come from
PaddleOCR; this crate brings them into Burn and connects the detection and
recognition stages.

## Try it

```sh
cargo run --release --example ocr --features pp_ocrv6_medium -- image.png
```

See [usage and model configuration](docs/usage.md) for the Rust API, model
variants, and device setup.
