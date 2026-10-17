mod det_model {
    use crate::det::{DbDetector, DetModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/det.rs"));

    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/det.bpk");

    impl DetModel for Model {
        fn forward(&self, images: Tensor<4>) -> Tensor<4> {
            self.forward(images)
        }
    }

    impl DbDetector<Model> {
        pub fn pp_ocrv6_tiny(device: Device) -> Self {
            Self::new(Model::from_file(WEIGHTS_FILE, &device), device)
        }
    }
}

mod rec_model {
    use crate::rec::{CtcRecognizer, RecModel};

    include!(concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/rec.rs"));

    const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/models/pp_ocrv6_tiny/rec.bpk");
    const DICTIONARY: &str = include_str!("../../models/pp_ocrv6_tiny/dict.txt");

    impl RecModel for Model {
        fn forward(&self, images: Tensor<4>) -> Tensor<3> {
            self.forward(images)
        }
    }

    impl CtcRecognizer<Model> {
        pub fn pp_ocrv6_tiny(device: Device) -> Self {
            Self::new(Model::from_file(WEIGHTS_FILE, &device), device, DICTIONARY)
        }
    }
}
