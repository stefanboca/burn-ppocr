use burn::prelude::*;
use image::RgbImage;

use super::preprocess::{RecPreProcess, orient_crop};
use super::{Rec, RecModel, Recognition};

pub struct CtcRecognizer<M> {
    model: M,
    device: Device,
    preprocess: RecPreProcess,
    characters: Vec<String>,
    batch_size: usize,
}

impl<M: RecModel> CtcRecognizer<M> {
    /// The dictionary excludes the CTC blank and trailing space, as in PaddleOCR.
    pub fn new(model: M, device: Device, dictionary: &str) -> Self {
        let mut characters = vec![String::new()];
        characters.extend(dictionary.lines().map(str::to_owned));
        characters.push(" ".to_owned());
        Self {
            model,
            device,
            preprocess: RecPreProcess::default(),
            characters,
            batch_size: 6,
        }
    }

    fn decode(&self, predictions: Tensor<3>) -> Vec<Recognition> {
        let [batch, timesteps, classes] = predictions.dims();
        assert_eq!(
            classes,
            self.characters.len(),
            "recognizer and dictionary class counts differ"
        );
        if timesteps == 0 {
            return (0..batch)
                .map(|_| Recognition {
                    text: String::new(),
                    score: 0.0,
                })
                .collect();
        }
        // The exported recognizer already includes softmax. Reduce before CPU transfer.
        let (scores, indices) = predictions.max_dim_with_indices(2);
        let scores = scores
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("recognition scores must be convertible to f32");
        let indices = indices
            .into_data()
            .try_into_vec_as::<i64>()
            .expect("recognition indices must be convertible to i64");
        indices
            .chunks_exact(timesteps)
            .zip(scores.chunks_exact(timesteps))
            .map(|(indices, scores)| self.decode_tokens(indices, scores))
            .collect()
    }

    fn decode_tokens(&self, indices: &[i64], scores: &[f32]) -> Recognition {
        let mut text = String::new();
        let mut sum = 0.0;
        let mut count = 0;
        let mut previous = 0;
        for (&index, &score) in indices.iter().zip(scores) {
            assert!(score.is_finite(), "recognizer returned a nonfinite score");
            if index != 0 && index != previous {
                let token = self
                    .characters
                    .get(index as usize)
                    .expect("recognizer returned an invalid class index");
                text.push_str(token);
                sum += score;
                count += 1;
            }
            previous = index;
        }
        Recognition {
            text,
            score: if count == 0 { 0.0 } else { sum / count as f32 },
        }
    }
}

impl<M: RecModel> Rec for CtcRecognizer<M> {
    fn forward(&self, crops: &[RgbImage]) -> Vec<Recognition> {
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
            for (&index, result) in batch.iter().zip(self.decode(predictions)) {
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

    use image::Rgb;

    use super::*;

    fn recognizer(dictionary: &str) -> CtcRecognizer<ColorRecognizer> {
        CtcRecognizer::new(
            ColorRecognizer(Rc::new(RefCell::new(Vec::new()))),
            Device::flex(),
            dictionary,
        )
    }

    #[test]
    fn ctc_collapses_repeats_but_blank_allows_repeated_characters() {
        let decoder = recognizer("a\nb\n");
        let result = decoder.decode_tokens(
            &[0, 1, 1, 0, 1, 2, 2, 3, 0],
            &[1.0, 0.9, 0.99, 1.0, 0.7, 0.8, 0.99, 0.6, 1.0],
        );
        assert_eq!(result.text, "aab ");
        assert!((result.score - 0.75).abs() < 1e-6);
        assert_eq!(decoder.decode_tokens(&[0, 0], &[1.0, 1.0]).score, 0.0);
    }

    #[test]
    fn ctc_handles_unicode_tokens_and_crlf_dictionaries() {
        let decoder = recognizer("é\r\n東京\r\n");
        let result = decoder.decode_tokens(&[1, 0, 2], &[0.8, 1.0, 0.9]);
        assert_eq!(result.text, "é東京");
    }

    #[test]
    fn tensor_decoding_uses_probabilities() {
        let decoder = recognizer("a\n");
        let device = Device::flex();
        let predictions = Tensor::from_data(
            [[
                [0.1f32, 0.8, 0.1],
                [0.1, 0.9, 0.0],
                [1.0, 0.0, 0.0],
                [0.1, 0.7, 0.2],
            ]],
            &device,
        );
        let result = decoder.decode(predictions);
        assert_eq!(result[0].text, "aa");
        assert!((result[0].score - 0.75).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "recognizer and dictionary class counts differ")]
    fn dictionary_size_must_match_model() {
        recognizer("a\n").decode(Tensor::zeros([1, 1, 2], &Device::flex()));
    }

    #[test]
    fn bundled_tiny_dictionary_matches_the_exported_model() {
        let decoder = recognizer(include_str!("../../models/pp_ocrv6_tiny/dict.txt"));
        assert_eq!(decoder.characters.len(), 6906);
    }
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
