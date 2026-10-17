//! Model-free component benchmarks: `cargo bench --bench pipeline -- det/postprocess`.
//! Uses Burn's `BURN_DEVICE` override and fixed FP16. Device stages wait for completion.
//! For a sampling profiler, pass a filter and `--profile-time 30` to Criterion.
use std::hint::black_box;
use std::num::NonZeroUsize;

use burn::tensor::{Device, DeviceConfig, FloatDType, Tensor, TensorData};
use burn_ppocr::Detection;
use burn_ppocr::crop::TextCropperConfig;
use burn_ppocr::det::DbPostProcessConfig;
use burn_ppocr::det::preprocess::DetPreProcessConfig;
use burn_ppocr::geometry::{Point, Polygon};
use burn_ppocr::rec::preprocess::RecPreProcessConfig;
use burn_ppocr::rec::{CtcDecoderConfig, RecBatcherConfig};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use image::{Rgb, RgbImage};

fn components(c: &mut Criterion) {
    let mut device = Device::default();
    device
        .configure(DeviceConfig::default().float_dtype(FloatDType::F16))
        .unwrap();
    eprintln!("Device: {device:?}; dtype: F16");
    let shape = (
        NonZeroUsize::new(640).unwrap(),
        NonZeroUsize::new(960).unwrap(),
    );
    let images = vec![RgbImage::from_fn(960, 640, |x, y| {
        Rgb([x as u8, y as u8, (x + y) as u8])
    })];
    let large_images = vec![RgbImage::from_fn(1920, 1280, |x, y| {
        Rgb([x as u8, y as u8, (x + y) as u8])
    })];
    let batch_images = vec![images[0].clone(); 6];
    let large_shape = (
        NonZeroUsize::new(1280).unwrap(),
        NonZeroUsize::new(1920).unwrap(),
    );
    let det_preprocess = DetPreProcessConfig::new().init(&device);
    let det_postprocess = DbPostProcessConfig::new()
        .with_box_thresh(0.4)
        .init(&device);
    // 64 disconnected text-like regions; avoid a trivially empty postprocessing workload.
    let mut pred = vec![0.01f32; 640 * 960];
    let mut detections = Vec::new();
    for row in 0..8 {
        for col in 0..8 {
            let (x, y) = (20 + col * 115, 20 + row * 75);
            for yy in y..y + 18 {
                pred[yy * 960 + x..yy * 960 + x + 80].fill(0.95);
            }
            detections.push(Detection {
                points: Polygon(vec![
                    Point::new(x as f32, y as f32),
                    Point::new((x + 80) as f32, y as f32),
                    Point::new((x + 80) as f32, (y + 18) as f32),
                    Point::new(x as f32, (y + 18) as f32),
                ]),
                score: 0.95,
            });
        }
    }
    let maps = Tensor::<4>::from_data(TensorData::new(pred, [1, 1, 640, 960]), &device);
    let crops: Vec<_> = [48, 96, 160, 320, 640, 960]
        .map(|width| RgbImage::from_pixel(width, 48, Rgb([100, 150, 200])))
        .into_iter()
        .collect();
    let rec_preprocess = RecPreProcessConfig::new().init(&device);
    let characters = std::iter::once(String::new())
        .chain(
            include_str!("../models/pp_ocrv6_tiny/dict.txt")
                .lines()
                .map(str::to_owned),
        )
        .chain(std::iter::once(" ".to_owned()))
        .collect();
    let decoder = CtcDecoderConfig::new(characters).init(&device);
    let cropper = TextCropperConfig::new().init(&device);
    let batcher = RecBatcherConfig::new().init(&device);
    let classes = decoder.characters.len();
    let mut probabilities = vec![0.0f32; 6 * 80 * classes];
    for t in 0..6 * 80 {
        let token = if t % 4 == 0 {
            0
        } else {
            1 + (t / 4) % (classes - 1)
        };
        probabilities[t * classes + token] = 0.99;
    }
    let predictions =
        Tensor::<3>::from_data(TensorData::new(probabilities, [6, 80, classes]), &device);
    // Make sure tensors/fixtures are ready before any measurement.
    device.sync().unwrap();
    assert_eq!(
        det_postprocess.forward(maps.clone(), shape).unwrap()[0].len(),
        64
    );
    assert_eq!(decoder.forward(predictions.clone()).unwrap().len(), 6);

    c.bench_function("det/preprocess/1x640x960", |b| {
        b.iter(|| {
            let input = det_preprocess.forward(black_box(&images), shape);
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("det/preprocess/1x1280x1920_to640x960", |b| {
        b.iter(|| {
            let input = det_preprocess.forward(black_box(&large_images), large_shape);
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("det/preprocess/6x640x960", |b| {
        b.iter(|| {
            let input = det_preprocess.forward(black_box(&batch_images), shape);
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("det/postprocess/64_regions", |b| {
        b.iter(|| {
            black_box(
                det_postprocess
                    .forward(black_box(maps.clone()), shape)
                    .unwrap(),
            );
            device.sync().unwrap();
        })
    });
    // Include reading-order sorting in the cropper workload.
    detections.reverse();
    c.bench_function("ocr/crops/64_regions", |b| {
        b.iter_batched(
            || vec![detections.clone()],
            |detections| black_box(cropper.forward(black_box(&images), detections)),
            BatchSize::SmallInput,
        )
    });
    let crop_refs: Vec<_> = crops.iter().collect();
    c.bench_function("rec/batch/6_mixed_widths", |b| {
        b.iter(|| black_box(batcher.forward(black_box(&crop_refs))))
    });
    c.bench_function("rec/preprocess/6_mixed_widths", |b| {
        b.iter(|| {
            let input = rec_preprocess.forward(black_box(&crops));
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("rec/decode/6x80x6906", |b| {
        b.iter(|| {
            black_box(decoder.forward(black_box(predictions.clone())).unwrap());
            device.sync().unwrap();
        })
    });
}

criterion_group!(benches, components);
criterion_main!(benches);
