use super::GPU;

use image::ImageBuffer;
use image::Rgb;
use wgpu::*;

impl GPU {
    pub async fn compile_kernel_shader(&self) -> ShaderModule {
        let kernel_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("kernel.wgsl").into()),
        });

        kernel_shader
    }
}

pub struct Kernel<const i: usize, const j: usize>(pub [[[f16; 4]; j]; i]);

impl<const i: usize, const j: usize> Kernel<i, j> {
    pub async fn apply(
        &self,
        image: ImageBuffer<Rgb<u8>, Vec<u8>>,
        gpu: &GPU,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        });

        

        let kernel_image = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: j as u32,
                height: i as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba16Float],
        });

        todo!()
    }
}
