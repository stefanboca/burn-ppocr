use burn::tensor::Device;
use burn_ppocr::OcrPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).ok_or("usage: ocr IMAGE")?;
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let pipeline = OcrPipeline::pp_ocrv6_tiny(Device::default());
    for result in pipeline.forward(&image) {
        println!(
            "{:.3}\t{}",
            result.recognition.score, result.recognition.text
        );
    }
    Ok(())
}
