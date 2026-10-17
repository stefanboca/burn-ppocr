use burn::tensor::{Device, Tensor};
use image::RgbImage;

use crate::db_postprocess::DbPostProcess;
use crate::preprocess::{DetPreProcess, RecPreProcess, orient_crop};
use crate::rec_postprocess::CtcDecoder;
use crate::{Detection, Recognition};

pub trait DetModel {
    fn forward(&self, images: Tensor<4>) -> Tensor<4>;
}

pub trait RecModel {
    fn forward(&self, images: Tensor<4>) -> Tensor<3>;
}

pub struct DbDetector<M> {
    model: M,
    device: Device,
    preprocess: DetPreProcess,
    postprocess: DbPostProcess,
}

impl<M: DetModel> DbDetector<M> {
    pub fn new(model: M, device: Device, postprocess: DbPostProcess) -> Self {
        Self {
            model,
            device,
            preprocess: DetPreProcess::default(),
            postprocess,
        }
    }

    pub fn forward(&self, image: &RgbImage) -> Vec<Detection> {
        if image.width() == 0 || image.height() == 0 {
            return Vec::new();
        }
        let input = self.preprocess.forward(image, &self.device);
        let maps = self.model.forward(input);
        self.postprocess
            .forward(
                maps,
                vec![(image.height() as usize, image.width() as usize)],
            )
            .pop()
            .expect("detector must return one batch item")
    }
}

pub struct CtcRecognizer<M> {
    model: M,
    device: Device,
    preprocess: RecPreProcess,
    decoder: CtcDecoder,
    batch_size: usize,
}

impl<M: RecModel> CtcRecognizer<M> {
    pub fn new(model: M, device: Device, dictionary: &str) -> Self {
        Self {
            model,
            device,
            preprocess: RecPreProcess::default(),
            decoder: CtcDecoder::new(dictionary),
            batch_size: 6,
        }
    }

    pub fn forward(&self, crops: &[RgbImage]) -> Vec<Recognition> {
        assert!(
            crops
                .iter()
                .all(|crop| crop.width() > 0 && crop.height() > 0),
            "recognition crops must be nonempty"
        );
        let crops: Vec<_> = crops.iter().map(orient_crop).collect();
        let mut indices: Vec<_> = (0..crops.len()).collect();
        let aspect_ratio =
            |index: usize| crops[index].width() as f64 / crops[index].height() as f64;
        indices.sort_by(|&a, &b| aspect_ratio(a).total_cmp(&aspect_ratio(b)));
        let mut results = vec![None; crops.len()];
        for batch in indices.chunks(self.batch_size) {
            let images: Vec<_> = batch.iter().map(|&index| &crops[index]).collect();
            let width = self.preprocess.batch_width(&images);
            let images = images
                .into_iter()
                .map(|image| self.preprocess.forward(image, width, &self.device))
                .collect();
            let predictions = self.model.forward(Tensor::stack::<4>(images, 0));
            assert_eq!(
                predictions.dims()[0],
                batch.len(),
                "recognizer batch size mismatch"
            );
            for (&index, result) in batch.iter().zip(self.decoder.forward(predictions)) {
                results[index] = Some(result);
            }
        }
        results
            .into_iter()
            .map(|result| result.expect("missing recognition result"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use burn::tensor::TensorData;
    use image::Rgb;

    use super::*;

    struct ColorRecognizer(Rc<RefCell<Vec<[usize; 4]>>>);
    impl RecModel for ColorRecognizer {
        fn forward(&self, images: Tensor<4>) -> Tensor<3> {
            let dims = images.dims();
            self.0.borrow_mut().push(dims);
            let device = images.device();
            let values = images.into_data().try_into_vec_as::<f32>().unwrap();
            let plane = dims[2] * dims[3];
            let mut predictions = vec![0.0f32; dims[0] * 4];
            for index in 0..dims[0] {
                let red = values[index * 3 * plane + 2 * plane];
                let blue = values[index * 3 * plane];
                let token = if red > blue { 1 } else { 2 };
                predictions[index * 4 + token] = 0.9;
            }
            Tensor::from_data(TensorData::new(predictions, [dims[0], 1, 4]), &device)
        }
    }

    #[test]
    fn recognition_batches_by_aspect_ratio_then_restores_original_order() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let recognizer =
            CtcRecognizer::new(ColorRecognizer(calls.clone()), Device::flex(), "a\nb\n");
        let crops: Vec<_> = [80, 12, 48, 8, 32, 16, 64]
            .into_iter()
            .enumerate()
            .map(|(index, width)| {
                RgbImage::from_pixel(
                    width,
                    4,
                    if index % 2 == 0 {
                        Rgb([255, 0, 0])
                    } else {
                        Rgb([0, 0, 255])
                    },
                )
            })
            .collect();
        let result = recognizer.forward(&crops);
        assert_eq!(
            result
                .iter()
                .map(|value| value.text.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "a", "b", "a", "b", "a"]
        );
        assert_eq!(*calls.borrow(), [[6, 3, 48, 768], [1, 3, 48, 960]]);
        assert!(recognizer.forward(&[]).is_empty());
        assert_eq!(calls.borrow().len(), 2);
    }
}
