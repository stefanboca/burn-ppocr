use burn::prelude::*;

/// A decoded text line and the mean confidence of its emitted CTC tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Recognition {
    pub text: String,
    pub score: f32,
}

pub struct CtcDecoder {
    characters: Vec<String>,
}

impl CtcDecoder {
    /// The dictionary excludes the CTC blank and trailing space, as in PaddleOCR.
    pub fn new(dictionary: &str) -> Self {
        let mut characters = vec![String::new()];
        characters.extend(dictionary.lines().map(str::to_owned));
        characters.push(" ".to_owned());
        Self { characters }
    }

    pub fn forward(&self, predictions: Tensor<3>) -> Vec<Recognition> {
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
            .map(|(indices, scores)| self.decode(indices, scores))
            .collect()
    }

    fn decode(&self, indices: &[i64], scores: &[f32]) -> Recognition {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_collapses_repeats_but_blank_allows_repeated_characters() {
        let decoder = CtcDecoder::new("a\nb\n");
        let result = decoder.decode(
            &[0, 1, 1, 0, 1, 2, 2, 3, 0],
            &[1.0, 0.9, 0.99, 1.0, 0.7, 0.8, 0.99, 0.6, 1.0],
        );
        assert_eq!(result.text, "aab ");
        assert!((result.score - 0.75).abs() < 1e-6);
        assert_eq!(decoder.decode(&[0, 0], &[1.0, 1.0]).score, 0.0);
    }

    #[test]
    fn ctc_handles_unicode_tokens_and_crlf_dictionaries() {
        let decoder = CtcDecoder::new("é\r\n東京\r\n");
        let result = decoder.decode(&[1, 0, 2], &[0.8, 1.0, 0.9]);
        assert_eq!(result.text, "é東京");
    }

    #[test]
    fn tensor_decoding_uses_probabilities() {
        let decoder = CtcDecoder::new("a\n");
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
        let result = decoder.forward(predictions);
        assert_eq!(result[0].text, "aa");
        assert!((result[0].score - 0.75).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "recognizer and dictionary class counts differ")]
    fn dictionary_size_must_match_model() {
        CtcDecoder::new("a\n").forward(Tensor::zeros([1, 1, 2], &Device::flex()));
    }

    #[test]
    fn bundled_tiny_dictionary_matches_the_exported_model() {
        let decoder = CtcDecoder::new(include_str!("../models/PP-OCRv6_tiny_dict.txt"));
        assert_eq!(decoder.characters.len(), 6906);
    }
}
