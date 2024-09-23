pub mod kernel;

use image::{ImageBuffer, Rgba, GenericImageView};
use wgpu::*;
use tokio::join;

pub struct GPU {
    pub device: Device,
    pub queue: Queue,
    pub kernel_shader: Option<ShaderModule>,
}

pub fn pad_to_multiple_of_256(n: u32) -> u32 {
    (n + 255) & !255
}

impl GPU {
    pub async fn new() -> Self {
        let instance = Instance::new(InstanceDescriptor {
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    required_features: Features::default() | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        Self { device, queue, kernel_shader: None }
    }
    pub async fn compile_shaders(&mut self) {
        let (kernel_shader,) = join!(self.compile_kernel_shader());

        self.kernel_shader = Some(kernel_shader);
    }

    pub async fn texture_to_image(&self, texture: &Texture, width: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let size = texture.size();
        #[cfg(debug_assertions)]
        print!("Converting texture to image with size {}x{}...\n", width, size.height);
        let buffer_size = (size.width * size.height * 4) as u64;
        let buffer_desc = BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let buffer = self.device.create_buffer(&buffer_desc);

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyBuffer {
                buffer: &buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(pad_to_multiple_of_256(4 * size.width)),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );

        self.queue.submit(Some(encoder.finish()));
        let buffer_slice = buffer.slice(..);

        buffer_slice.map_async(
            MapMode::Read, 
            |result| {
                if let Err(e) = result {
                    eprintln!("Failed to map buffer: {:?}", e);
                    return;
                }
            }
        );
        self.device.poll(Maintain::Wait);

        let data = buffer_slice.get_mapped_range();

        let image = ImageBuffer::from_raw(size.width, size.height, data.to_vec()).unwrap();
        //crop off the padding
        image.view(0, 0, width, size.height).to_image()
    }
}