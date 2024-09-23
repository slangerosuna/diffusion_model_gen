use crate::gpu::{
    GPU, pad_to_multiple_of_256,
};

use image::ImageBuffer;
use image::Rgba;
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

pub struct Kernel(Texture);

impl Kernel {
    pub fn new(
        data: &[f32],
        i: u32,
        j: u32,
        gpu: &GPU,
    ) -> Self {
        #[cfg(debug_assertions)]
        assert!(data.len() as u32 == i * j * 4);

        #[cfg(debug_assertions)]
        print!("Creating kernel with size {}x{}...\n", i, j);

        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: i,
                height: j,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba32Float],
        });

        gpu.queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4 * i),
                rows_per_image: Some(j),
            },
            Extent3d {
                width: i,
                height: j,
                depth_or_array_layers: 1,
            },
        );

        Self(texture)
    }

    pub async fn apply_to_image(
        &self,
        image: ImageBuffer<Rgba<u8>, Vec<u8>>,
        gpu: &GPU,
    ) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let width = image.width();
        let height = image.height();

        #[cfg(debug_assertions)]
        print!("Creating input texture...\n");
        let input_texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        #[cfg(debug_assertions)]
        print!("Writing input texture...\n");
        gpu.queue.write_texture(
            ImageCopyTexture {
                texture: &input_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            image.into_vec().as_slice(),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width as u32),
                rows_per_image: Some(height as u32),
            },
            Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
        );

        #[cfg(debug_assertions)]
        print!("Creating output texture...\n");
        let output_texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: pad_to_multiple_of_256(width),
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        self.apply(&input_texture, &output_texture, width, height, gpu).await;

        #[cfg(debug_assertions)]
        print!("Reading output texture...\n");
        gpu.texture_to_image(&output_texture, width).await
    }

    pub async fn apply(
        &self,
        input_texture: &Texture,
        output_texture: &Texture,
        width: u32,
        height: u32,
        gpu: &GPU,
    ) {
        #[cfg(debug_assertions)]
        print!("Applying kernel...\n");
        let bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        #[cfg(debug_assertions)]
        assert!(gpu.kernel_shader.is_some());

        let kernel_shader = gpu.kernel_shader.as_ref().unwrap();

        let pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: kernel_shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&input_texture.create_view(&TextureViewDescriptor::default())),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.0.create_view(&TextureViewDescriptor::default())),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&output_texture.create_view(&TextureViewDescriptor::default())),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor {
            label: None,
        });

        #[cfg(debug_assertions)]
        print!("Dispatching workgroups...\n");
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                timestamp_writes: Default::default(),
                label: None,
            });

            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(width, height, 1);
        }

        #[cfg(debug_assertions)]
        print!("Submitting work...\n");
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
