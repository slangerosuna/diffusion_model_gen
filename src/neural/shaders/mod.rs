use tokio::join;
use wgpu::ShaderModule;

use crate::GpuDevice;

impl GpuDevice {
    pub async fn compile_neural_shaders(&self) -> Vec<ShaderModule> {
        () = join!();

        vec![]
    }
}