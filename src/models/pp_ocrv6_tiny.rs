pub mod det_model {
    use crate::det::preprocess::DetPreProcess;
    use crate::det::{DbDetector, DbPostProcess, DetModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/det.rs"));

    #[cfg(not(feature = "embed"))]
    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/det.bpk");

    impl DetModel for Model {
        fn forward(&self, images: Tensor<4>) -> Tensor<4> {
            self.forward(images)
        }
    }

    impl DbDetector<Model> {
        pub fn pp_ocrv6_tiny(device: Device) -> Self {
            #[cfg(feature = "embed")]
            let model = Model::from_embedded(&device);
            #[cfg(not(feature = "embed"))]
            let model = Model::from_file(WEIGHTS_FILE, &device);
            Self {
                model,
                preprocess: DetPreProcess { max_side: 960 },
                postprocess: DbPostProcess {
                    thresh: 0.2,
                    box_thresh: 0.4,
                    max_candidates: 3000,
                    unclip_ratio: 1.4,
                },
            }
        }
    }
}

pub mod rec_model {
    use std::num::NonZeroU32;

    use crate::rec::preprocess::RecPreProcess;
    use crate::rec::{CtcDecoder, CtcRecognizer, RecModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/rec.rs"));

    #[cfg(not(feature = "embed"))]
    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/rec.bpk");
    const DICTIONARY: &str = include_str!("../../models/pp_ocrv6_tiny/dict.txt");

    impl RecModel for Model {
        fn forward(&self, images: Tensor<4>) -> Tensor<3> {
            self.forward(images)
        }
    }

    impl CtcRecognizer<Model> {
        pub fn pp_ocrv6_tiny(device: Device) -> Self {
            #[cfg(feature = "embed")]
            let model = Model::from_embedded(&device);
            #[cfg(not(feature = "embed"))]
            let model = Model::from_file(WEIGHTS_FILE, &device);
            // The dictionary excludes the CTC blank and trailing space.
            let mut characters = vec![String::new()];
            characters.extend(DICTIONARY.lines().map(str::to_owned));
            characters.push(" ".to_owned());
            Self {
                model,
                preprocess: RecPreProcess {
                    height: NonZeroU32::new(48).unwrap(),
                    width: NonZeroU32::new(320).unwrap(),
                },
                decoder: CtcDecoder { characters },
                batch_size: Self::DEFAULT_BATCH_SIZE,
            }
        }
    }
}

impl crate::OcrPipeline<det_model::Model, rec_model::Model> {
    pub fn pp_ocrv6_tiny(device: burn::tensor::Device) -> Self {
        Self::new(
            crate::DbDetector::pp_ocrv6_tiny(device.clone()),
            crate::CtcRecognizer::pp_ocrv6_tiny(device),
        )
    }
}
