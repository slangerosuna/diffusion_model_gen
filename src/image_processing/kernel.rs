use image::ImageBuffer;
use image::Rgb;
use lazy_static::lazy_static;
use wgpu::*;

lazy_static! {
    pub static ref KERNEL_SHADER: ShaderModule =
        Device::new().create_shader_module(&ShaderModuleDescriptor {
            label: Label::None,
            source: ShaderSource::Wgsl(include_str!("kernel.wgsl").into()),
        });
}

pub struct Kernel<const i: usize, const j: usize>(pub [[f32; j]; i]);

impl<const i: usize, const j: usize> Kernel<i, j> {
    pub fn apply(&self, image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let mut new_image = ImageBuffer::new(image.width(), image.height());

        new_image
    }
}
