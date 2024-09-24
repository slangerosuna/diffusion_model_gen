use crate::GpuDevice;

use wgpu::*;

impl GpuDevice {
    pub async fn compile_convolution_shader(&self) -> ShaderModule {
        let kernel_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("convolution.wgsl").into()),
        });

        kernel_shader
    }
}
