#![feature(f16)]
#[tokio::main]
async fn main() {
    let mut image_processor = GPU::new().await;
    image_processor.compile_shaders().await;
}

mod kernel;
pub use kernel::*;

use wgpu::*;
use tokio::join;

pub struct GPU {
    device: Device,
    queue: Queue,
    kernel_shader: Option<ShaderModule>,
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
}
