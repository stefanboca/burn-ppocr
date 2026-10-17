//! Model-free component benchmarks: `cargo bench --bench pipeline -- det/postprocess`.
//! Uses Burn's `BURN_DEVICE` override and fixed FP16. Device stages wait for completion.
//! For a sampling profiler, pass a filter and `--profile-time 30` to Criterion.
use std::hint::black_box;
use std::num::{NonZeroU32, NonZeroUsize};

use burn::tensor::{Device, DeviceConfig, FloatDType, Tensor, TensorData};
use burn_ppocr::Detection;
use burn_ppocr::det::DbPostProcess;
use burn_ppocr::det::postprocess::{min_area_rect, polygon_score};
use burn_ppocr::det::preprocess::DetPreProcess;
use burn_ppocr::geometry::{Point, Polygon};
use burn_ppocr::ocr::{crop_text, sort_reading_order};
use burn_ppocr::rec::CtcDecoder;
use burn_ppocr::rec::preprocess::{RecPreProcess, orient_crop};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use image::{Rgb, RgbImage, imageops};

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
    let large_shape = (
        NonZeroUsize::new(1280).unwrap(),
        NonZeroUsize::new(1920).unwrap(),
    );
    let det_preprocess = DetPreProcess { max_side: 960 };
    let det_postprocess = DbPostProcess {
        thresh: 0.2,
        box_thresh: 0.4,
        max_candidates: 3000,
        unclip_ratio: 1.4,
    };
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
    let maps = Tensor::<3>::from_data(TensorData::new(pred.clone(), [1, 640, 960]), &device);
    let polygon = Polygon(vec![
        Point::new(30., 30.),
        Point::new(350., 45.),
        Point::new(348., 93.),
        Point::new(28., 78.),
    ]);
    let crops: Vec<_> = [48, 96, 160, 320, 640, 960]
        .map(|width| RgbImage::from_pixel(width, 48, Rgb([100, 150, 200])))
        .into_iter()
        .collect();
    let rec_preprocess = RecPreProcess {
        height: NonZeroU32::new(48).unwrap(),
        width: NonZeroU32::new(320).unwrap(),
    };
    let decoder = CtcDecoder {
        characters: std::iter::once(String::new())
            .chain(
                include_str!("../models/pp_ocrv6_tiny/dict.txt")
                    .lines()
                    .map(str::to_owned),
            )
            .chain(std::iter::once(" ".to_owned()))
            .collect(),
    };
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
    let indices: Vec<i64> = (0..80)
        .map(|t| if t % 4 == 0 { 0 } else { 1 + t / 4 })
        .collect();
    let scores = vec![0.99; 80];
    let crop_refs: Vec<_> = crops.iter().collect();
    let batch_width = rec_preprocess.batch_width(&crop_refs);
    let portrait = RgbImage::new(48, 320);
    // Make sure tensors/fixtures are ready before any measurement.
    device.sync().unwrap();
    assert_eq!(
        det_postprocess.forward(maps.clone(), shape).unwrap()[0].len(),
        64
    );
    assert_eq!(decoder.decode(predictions.clone()).unwrap().len(), 6);

    c.bench_function("det/preprocess/1x640x960", |b| {
        b.iter(|| {
            let input = det_preprocess.forward(black_box(&images), shape, &device);
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("det/preprocess/1x1280x1920_to640x960", |b| {
        b.iter(|| {
            let input = det_preprocess.forward(black_box(&large_images), large_shape, &device);
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("det/resize/1280x1920_to640x960", |b| {
        b.iter(|| {
            black_box(imageops::resize(
                black_box(&large_images[0]),
                960,
                640,
                imageops::FilterType::Triangle,
            ))
        })
    });
    c.bench_function("det/convert_pixels/640x960", |b| {
        b.iter_batched(
            || images[0].clone().into_raw(),
            |pixels| black_box(TensorData::new(pixels, [1, 640, 960, 3]).convert::<f32>()),
            BatchSize::LargeInput,
        )
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
    c.bench_function("det/polygon_score", |b| {
        b.iter(|| black_box(polygon_score(black_box(&pred), shape, black_box(&polygon))))
    });
    c.bench_function("geometry/min_area_rect", |b| {
        b.iter(|| black_box(min_area_rect(black_box(&polygon.0))))
    });
    c.bench_function("geometry/offset", |b| {
        b.iter(|| black_box(black_box(&polygon).offset(10.0)))
    });
    // Clone outside the timed routine so every iteration sorts the same shuffled input.
    detections.reverse();
    c.bench_function("ocr/sort/64_regions", |b| {
        b.iter_batched(
            || detections.clone(),
            |mut detections| {
                sort_reading_order(black_box(&mut detections));
                black_box(detections);
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("ocr/crop/320x48", |b| {
        b.iter(|| black_box(crop_text(black_box(&images[0]), black_box(&polygon)).unwrap()))
    });
    c.bench_function("rec/orient/portrait", |b| {
        b.iter(|| black_box(orient_crop(black_box(&portrait))))
    });
    c.bench_function("rec/preprocess/6_mixed_widths", |b| {
        b.iter(|| {
            let inputs: Vec<_> = crops
                .iter()
                .map(|crop| rec_preprocess.forward(black_box(crop), batch_width, &device))
                .collect();
            let input = Tensor::stack::<4>(inputs, 0);
            device.sync().unwrap();
            black_box(input);
        })
    });
    c.bench_function("rec/decode/6x80x6906", |b| {
        b.iter(|| {
            black_box(decoder.decode(black_box(predictions.clone())).unwrap());
            device.sync().unwrap();
        })
    });
    c.bench_function("rec/decode_tokens/80", |b| {
        b.iter(|| black_box(decoder.decode_tokens(black_box(&indices), black_box(&scores))))
    });
}

criterion_group!(benches, components);
criterion_main!(benches);
