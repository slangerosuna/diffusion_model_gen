#![feature(generic_const_exprs)]

mod gpu;

use gpu::kernel::Kernel;
use gpu::GPU;

#[tokio::main]
async fn main() {
    let mut gpu = GPU::new().await;
    gpu.compile_shaders().await;

    let input = image::open("input.png").unwrap().to_rgba8();
    let kernel = Kernel::gaussian_kernel::<10, 10>(&gpu);

    #[cfg(debug_assertions)]
    let now = std::time::Instant::now();
    let output = kernel.apply_to_image(input, &gpu).await;
    #[cfg(debug_assertions)]
    print!("Elapsed: {:?}\n", now.elapsed());

    output.save("output.png").unwrap();
}
