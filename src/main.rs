use burn::prelude::*;
use burn::tensor::{Distribution, TensorCreationOptions, shape};
use burn_ppocr::models::{pp_ocrv6_tiny_det, pp_ocrv6_tiny_rec};

fn main() {
    let device = Device::default();

    let det = pp_ocrv6_tiny_det::Model::from_file(pp_ocrv6_tiny_det::WEIGHTS_FILE, &device);
    let rec = pp_ocrv6_tiny_rec::Model::from_file(pp_ocrv6_tiny_rec::WEIGHTS_FILE, &device);

    let input = Tensor::random(
        shape!(10, 3, 256, 256),
        Distribution::Uniform(0., 1.),
        TensorCreationOptions::new(device.clone()),
    );
    let det_out = det.forward(input);
    dbg!(det_out.shape());

    let input = Tensor::random(
        shape!(10, 3, 48, 256),
        Distribution::Uniform(0., 1.),
        TensorCreationOptions::new(device.clone()),
    );
    let rec_out = rec.forward(input);
    dbg!(rec_out.shape());
}
