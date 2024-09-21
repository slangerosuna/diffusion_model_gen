mod kernel;
pub use kernel::*;

use wgpu::*;

pub struct ImageProcessor {
    device: Device,
    kernel_shader: Option<ShaderModule>,
}

impl ImageProcessor {
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
        let (device, _queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        Self { device, None }
    }
}
