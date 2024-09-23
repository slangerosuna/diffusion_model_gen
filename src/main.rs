#![feature(generic_const_exprs)]

mod gpu;

use gpu::GPU;
use gpu::kernel::Kernel;

#[tokio::main]
async fn main() {
    let mut gpu = GPU::new().await;
    gpu.compile_shaders().await;

    let input = image::open("input.png").unwrap().to_rgba8();
    let kernel = Kernel::new(
        &left_edge_detection_kernel::<11, 11>(),
        11,
        11,
        &gpu,
    );

    let now = std::time::Instant::now();
    let output = kernel.apply_to_image(input, &gpu).await;
    println!("Elapsed: {:?}", now.elapsed());
    output.save("output.png").unwrap();
}

fn gaussian_kernel<const I: usize, const J: usize>() -> [f32; I * J * 4] {
    let mut kernel = [0.0; I * J * 4];
    let sigma = I as f32 / 3.0;
    let mut sum = 0.0;
    for x in 0..I {
        for y in 0..J {
            let value = (-(x as f32 * x as f32 + y as f32 * y as f32) / (2.0 * sigma * sigma)).exp();
            kernel[(x * I + y) * 4] = value;
            kernel[(x * I + y) * 4 + 1] = value;
            kernel[(x * I + y) * 4 + 2] = value;
            kernel[(x * I + y) * 4 + 3] = 1.0;
            sum += value;
        }
    }
    for i in 0..I * J * 4 {
        kernel[i] /= sum;
    }

    kernel
}

fn left_edge_detection_kernel<const I: usize, const J: usize>() -> [f32; I * J * 4] {
    #[cfg(debug_assertions)]
    assert!(I % 2 == 1 && J % 2 == 1);

    let mut kernel = [0.0; I * J * 4];    

    /*
     * ie 3x3 kernel
     * -1 0 1
     * -2 0 2
     * -1 0 1
     * 
     * or 5x5 kernel
     * 0 -1 0 1 0
     * -1 -2 0 2 1
     * -2 -4 0 4 2
     * -1 -2 0 2 1
     * 0 -1 0 1 0
     */

    let center = (I / 2 + 1, J / 2 + 1);
    for x in 0..I {
        for y in 0..J {
            let value = match x {
                x if x < center.0 => -1.0,
                x if x > center.0 => 1.0,
                _ => 0.0,
            };

            let value = value * (I+J).abs_diff(center.1.abs_diff(y)).abs_diff(center.0.abs_diff(x)) as f32;

            kernel[(x * I + y) * 4] = value;
            kernel[(x * I + y) * 4 + 1] = value;
            kernel[(x * I + y) * 4 + 2] = value;
            kernel[(x * I + y) * 4 + 3] = 1.0;
        }
    }

    kernel
}