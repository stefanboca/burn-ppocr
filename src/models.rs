#[cfg(feature = "pp_ocrv6_tiny_det")]
pub mod pp_ocrv6_tiny_det {
    include!(concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_det.rs"));

    pub const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_det.bpk");
}

#[cfg(feature = "pp_ocrv6_tiny_rec")]
pub mod pp_ocrv6_tiny_rec {
    include!(concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_rec.rs"));

    pub const WEIGHTS_FILE: &str = concat!(env!("OUT_DIR"), "/model/PP-OCRv6_tiny_rec.bpk");
}
