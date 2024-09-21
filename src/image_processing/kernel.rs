use super::ImageProcessor;

use image::ImageBuffer;
use image::Rgb;
use wgpu::*;

impl ImageProcessor {
    pub async fn compile_shaders(&mut self) {
        let kernel_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("kernel.wgsl").into()),
        });

        self.kernel_shader = Some(kernel_shader);
    }
}

pub struct Kernel<const i: usize, const j: usize>(pub [[f32; j]; i]);

impl<const i: usize, const j: usize> Kernel<i, j> {
    pub async fn apply(
        &self,
        image: ImageBuffer<Rgb<u8>, Vec<u8>>,
        image_processor: &ImageProcessor,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let mut new_image = ImageBuffer::new(image.width(), image.height());

        let texture = image_processor.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgb8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        });

        new_image
    }
}
