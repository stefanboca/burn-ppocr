use burn::prelude::*;
use burn::tensor::{Distribution, TensorCreationOptions, shape};

fn main() {
    let device = Device::default();

    let det = burn_ppocr::models::pp_ocrv6_tiny_det::Model::from_embedded(&device);
    // let rec = burn_ppocr::models::pp_ocrv6_tiny_det::Model::from_embedded(&device);

    let input = Tensor::random(
        shape!(1, 3, 256, 256),
        Distribution::Uniform(0., 1.),
        TensorCreationOptions::new(device),
    );
    let output = det.forward(input);
    dbg!(output.shape());
}
