//! `cargo run --profile profiling --example profile --features pp_ocrv6_medium,profiling -- IMAGE`
//! Open Tracy 0.13.1 and connect to this process to capture CPU and GPU zones.
use std::hint::black_box;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, DeviceConfig, FloatDType, Tensor};
use burn_ppocr::OcrProcessor;
use cubecl::config::profiling::ProfilingLogLevel;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
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
    let images = paths
        .iter()
        .map(|path| {
            Ok(image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Set before creating any device. Preserve the project's other CubeCL settings.
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.profiling.logger.level = ProfilingLogLevel::Basic;
    CubeClRuntimeConfig::set(config);

    let client = tracing_tracy::client::Client::start();
    let mut device = Device::default();
    device.configure(DeviceConfig::default().float_dtype(FloatDType::F16))?;
    let pipeline = OcrProcessor::pp_ocrv6_medium(device.clone()).map(&mut F16);
    // Finish loading, kernel compilation, and autotuning before the measured frames.
    for _ in 0..10 {
        black_box(pipeline.forward(&images)?);
    }

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "off,burn_ppocr=info,profile=info".into());
    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_tracy::TracyLayer::default());
    tracing::subscriber::set_global_default(subscriber)?;
    eprintln!("Connect Tracy to this process to collect the capture.");
    client.frame_mark();

    for _ in 0..10 {
        black_box(pipeline.forward(&images)?);
        client.frame_mark();
    }
    Ok(())
}
