use std::num::NonZeroUsize;

use anyhow::ensure;
use burn::prelude::*;
use image::RgbImage;

use super::preprocess::DetPreProcess;
use super::{DbPostProcess, DetModel, Detection};

#[derive(Module, Debug)]
pub struct DbDetector<M: DetModel> {
    pub model: M,
    #[module(skip)]
    pub preprocess: DetPreProcess,
    #[module(skip)]
    pub postprocess: DbPostProcess,
}

impl<M: DetModel> DbDetector<M> {
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "det.forward")
    )]
    pub fn forward(
        &self,
        images: &[RgbImage],
        device: &Device,
    ) -> anyhow::Result<Vec<Vec<Detection>>> {
        let Some(image) = images.first() else {
            return Ok(Vec::new());
        };
        ensure!(
            images
                .iter()
                .all(|other| other.dimensions() == image.dimensions()),
            "detector images must have the same dimensions"
        );
        let Some(src_shape) = NonZeroUsize::new(image.height() as usize)
            .zip(NonZeroUsize::new(image.width() as usize))
        else {
            return Ok(vec![Vec::new(); images.len()]);
        };
        let input = self.preprocess.forward(images, src_shape, device);
        let maps = {
            #[cfg(feature = "profiling")]
            let _span = tracing::info_span!("det.model").entered();
            self.model.forward(input)
        };
        assert_eq!(maps.dims()[0], images.len(), "detector batch size mismatch");
        let channels = maps.dims()[1];
        ensure!(
            channels == 1,
            "detector maps must have one channel, got {channels}"
        );
        self.postprocess.forward(maps.squeeze_dim(1), src_shape)
    }
}
