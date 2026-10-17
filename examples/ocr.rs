use burn::tensor::Device;
use burn_ppocr::OcrPipeline;

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
    let pipeline = OcrPipeline::pp_ocrv6_medium(Device::default());
    for (path, results) in paths.iter().zip(pipeline.forward(&images)?) {
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
    Ok(())
}
