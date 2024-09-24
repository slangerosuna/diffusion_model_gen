#![feature(generic_const_exprs)]

mod gpu;

use gpu::GPU;
use gpu::kernel::Kernel;

#[tokio::main]
async fn main() {
    let mut gpu = GPU::new().await;
    gpu.compile_shaders().await;

    let input = image::open("input.png").unwrap().to_rgba8();
    let kernel = Kernel::gaussian_kernel::<10, 10>(&gpu);

    let now = std::time::Instant::now();
    let output = kernel.apply_to_image(input, &gpu).await;
    println!("Elapsed: {:?}", now.elapsed());
    output.save("output.png").unwrap();
}