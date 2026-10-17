use std::hint::black_box;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, DeviceConfig, FloatDType, Tensor};
use burn_ppocr::OcrPipeline;

struct F16;

impl ModuleMapper for F16 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(FloatDType::F16))
    }
}

fn main() -> anyhow::Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.is_empty() {
        anyhow::bail!("usage: ocr IMAGE [IMAGE ...] (all images must have the same dimensions)");
    }
    let mut images = Vec::new();
    for path in &paths {
        images.push(
            image::ImageReader::open(path)?
                .with_guessed_format()?
                .decode()?,
        );
    }
    let mut device = Device::default();
    device.configure(DeviceConfig::default().float_dtype(FloatDType::F16))?;
    let pipeline = OcrPipeline::pp_ocrv6_medium(device.clone()).map(&mut F16);
    for (path, results) in paths.iter().zip(pipeline.forward(&images, &device)?) {
        if paths.len() > 1 {
            println!("{}", std::path::Path::new(path).display());
        }
        for result in results {
            println!(
                "{:.3}\t{}",
                result.recognition.score, result.recognition.text
            );
        }
    }

    for _ in 0..10 {
        let _ = black_box(pipeline.forward(&images, &device));
    }
    device.sync()?;

    let mut total = std::time::Duration::ZERO;
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let _ = black_box(pipeline.forward(&images, &device));
        device.sync()?;
        let elapsed = start.elapsed();
        dbg!(start.elapsed());
        total += elapsed;
    }
    dbg!(total / 10);

    Ok(())
}
