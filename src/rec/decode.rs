use anyhow::Context;
use burn::tensor::Tensor;

use super::Recognition;

/// CTC reduction and decoding, independent of the recognition model.
#[derive(Clone, Debug)]
pub struct CtcDecoder {
    pub characters: Vec<String>,
}

impl CtcDecoder {
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "rec.decode")
    )]
    pub fn decode(&self, predictions: Tensor<3>) -> anyhow::Result<Vec<Recognition>> {
        let [_, timesteps, classes] = predictions.dims();
        assert_eq!(
            classes,
            self.characters.len(),
            "recognizer and dictionary class counts differ"
        );
        // The exported recognizer already includes softmax. Reduce before CPU transfer.
        let (scores, indices) = predictions.max_dim_with_indices(2);
        let (scores, indices) = {
            #[cfg(feature = "profiling")]
            let _span = tracing::info_span!("rec.readback").entered();
            (
                scores
                    .try_into_vec_as::<f32>()
                    .context("reading recognition scores")?,
                indices
                    .try_into_vec_as::<i64>()
                    .context("reading recognition indices")?,
            )
        };
        Ok(indices
            .chunks_exact(timesteps)
            .zip(scores.chunks_exact(timesteps))
            .map(|(indices, scores)| self.decode_tokens(indices, scores))
            .collect())
    }

    pub fn decode_tokens(&self, indices: &[i64], scores: &[f32]) -> Recognition {
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
