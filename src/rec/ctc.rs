use anyhow::{Context, ensure};
use burn::prelude::*;
use image::RgbImage;

use super::preprocess::{RecPreProcess, orient_crop};
use super::{Rec, RecModel, Recognition};

#[derive(Module, Debug)]
pub struct CtcRecognizer<M: RecModel> {
    pub model: M,
    #[module(skip)]
    pub preprocess: RecPreProcess,
    #[module(skip)]
    pub characters: Vec<String>,
    pub batch_size: usize,
}

impl<M: RecModel> CtcRecognizer<M> {
    /// Upstream's default batching policy, independent of the recognition model.
    pub const DEFAULT_BATCH_SIZE: usize = 6;

    fn decode(&self, predictions: Tensor<3>) -> anyhow::Result<Vec<Recognition>> {
        let [_, timesteps, classes] = predictions.dims();
        assert_eq!(
            classes,
            self.characters.len(),
            "recognizer and dictionary class counts differ"
        );
        // The exported recognizer already includes softmax. Reduce before CPU transfer.
        let (scores, indices) = predictions.max_dim_with_indices(2);
        let scores = scores
            .try_into_vec_as::<f32>()
            .context("reading recognition scores")?;
        let indices = indices
            .try_into_vec_as::<i64>()
            .context("reading recognition indices")?;
        Ok(indices
            .chunks_exact(timesteps)
            .zip(scores.chunks_exact(timesteps))
            .map(|(indices, scores)| self.decode_tokens(indices, scores))
            .collect())
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
    fn forward(&self, crops: &[RgbImage], device: &Device) -> anyhow::Result<Vec<Recognition>> {
        ensure!(
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
                .map(|image| self.preprocess.forward(image, width, device))
                .collect();
            let predictions = self.model.forward(Tensor::stack::<4>(images, 0));
            assert_eq!(
                predictions.dims()[0],
                batch.len(),
                "recognizer batch size mismatch"
            );
            for (&index, result) in batch.iter().zip(self.decode(predictions)?) {
                results[index] = Some(result);
            }
        }
        Ok(results
            .into_iter()
            .map(|result| result.expect("missing recognition result"))
            .collect())
    }
}
