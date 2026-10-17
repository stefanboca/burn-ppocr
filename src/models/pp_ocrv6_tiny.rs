mod det_model {
    use crate::det::preprocess::DetPreProcess;
    use crate::det::{DbDetector, DetModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/det.rs"));

    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/det.bpk");

    impl DetModel for Model {
        fn forward(&self, images: Tensor<4>) -> Tensor<4> {
            self.forward(images)
        }
    }

    impl DbDetector {
        pub fn pp_ocrv6_tiny(device: Device) -> Self {
            Self {
                model: Box::new(Model::from_file(WEIGHTS_FILE, &device)),
                device,
                preprocess: DetPreProcess { max_side: 960 },
                thresh: 0.2,
                box_thresh: 0.4,
                max_candidates: 3000,
                unclip_ratio: 1.4,
            }
        }
    }
}

mod rec_model {
    use std::num::NonZeroU32;

    use crate::rec::preprocess::RecPreProcess;
    use crate::rec::{CtcRecognizer, RecModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/rec.rs"));

    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/rec.bpk");
    const DICTIONARY: &str = include_str!("../../models/pp_ocrv6_tiny/dict.txt");

    impl RecModel for Model {
        fn forward(&self, images: Tensor<4>) -> Tensor<3> {
            self.forward(images)
        }
    }

    impl CtcRecognizer {
        pub fn pp_ocrv6_tiny(device: Device) -> Self {
            // The dictionary excludes the CTC blank and trailing space.
            let mut characters = vec![String::new()];
            characters.extend(DICTIONARY.lines().map(str::to_owned));
            characters.push(" ".to_owned());
            Self {
                model: Box::new(Model::from_file(WEIGHTS_FILE, &device)),
                device,
                preprocess: RecPreProcess {
                    height: NonZeroU32::new(48).unwrap(),
                    width: NonZeroU32::new(320).unwrap(),
                },
                characters,
                batch_size: Self::DEFAULT_BATCH_SIZE,
            }
        }
    }
}
