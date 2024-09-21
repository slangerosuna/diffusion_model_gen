mod image_processing;
use image_processing::*;

#[tokio::main]
async fn main() {
    let mut image_processor = ImageProcessor::new().await;
    image_processor.compile_shaders().await;
}
