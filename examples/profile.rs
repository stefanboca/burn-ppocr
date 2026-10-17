//! `cargo run --profile profiling --example profile --features pp_ocrv6_medium,profiling -- IMAGE`
//! Set `BURN_DEVICE`, `OCR_WARMUP` (10), `OCR_ITERATIONS` (10), and `OCR_TRACE`
//! (`target/ocr-trace.json`). Open the trace in https://ui.perfetto.dev.
//! Host spans include readback waits; they do not measure asynchronous GPU kernels.
//! Use `RUST_LOG=off,burn_ppocr=trace,profile=info` for individual geometry operations.
use std::hint::black_box;
use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, DeviceConfig, FloatDType, Tensor};
use burn_ppocr::OcrPipeline;
use tracing_subscriber::prelude::*;

struct F16;

impl ModuleMapper for F16 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(FloatDType::F16))
    }
}

fn main() -> anyhow::Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(!paths.is_empty(), "usage: profile IMAGE [IMAGE ...]");
    let warmup: usize = std::env::var("OCR_WARMUP").unwrap_or("10".into()).parse()?;
    let iterations: NonZeroU32 = std::env::var("OCR_ITERATIONS")
        .unwrap_or("10".into())
        .parse()?;
    let path = std::env::var_os("OCR_TRACE").unwrap_or("target/ocr-trace.json".into());
    let images = paths
        .iter()
        .map(|path| {
            Ok(image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut device = Device::default();
    device.configure(DeviceConfig::default().float_dtype(FloatDType::F16))?;
    let pipeline = OcrPipeline::pp_ocrv6_medium(device.clone()).map(&mut F16);
    // Loading, kernel compilation, and autotuning stay outside the recorded interval.
    for _ in 0..warmup {
        black_box(pipeline.forward(&images, &device)?);
    }
    device.sync()?;

    let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(std::fs::File::create(&path)?)
        .include_args(true)
        .build();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "off,burn_ppocr=info,profile=info".into());
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);
    let _subscriber = tracing::subscriber::set_default(subscriber);
    eprintln!("Writing trace to {}", std::path::Path::new(&path).display());

    let mut total = Duration::ZERO;
    for iteration in 0..iterations.get() {
        let _span = tracing::info_span!("inference", iteration).entered();
        let start = Instant::now();
        black_box(pipeline.forward(&images, &device)?);
        // Host spans alone cannot account for asynchronously submitted GPU work.
        device.sync()?;
        let elapsed = start.elapsed();
        eprintln!("inference {iteration}: {elapsed:?}");
        total += elapsed;
    }
    eprintln!(
        "mean: {:?} ({iterations} iterations)",
        total / iterations.get()
    );
    Ok(())
}
