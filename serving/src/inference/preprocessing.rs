use anyhow::Result;
use candle_core::Tensor;
use image::{ImageReader, imageops::FilterType};
use std::io::Cursor;

fn load_image(img: &[u8]) -> Result<image::DynamicImage> {
    Ok(ImageReader::new(Cursor::new(img))
        .with_guessed_format()?
        .decode()?)
}

fn img_resize(img: &image::DynamicImage) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    img.resize_exact(224, 224, FilterType::Triangle).to_rgb8()
}

fn img_to_vec(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut red = Vec::with_capacity(224 * 224);
    let mut green = Vec::with_capacity(224 * 224);
    let mut blue = Vec::with_capacity(224 * 224);

    for pixel in img.pixels() {
        red.push((pixel[0] as f32 / 255.0 - 0.5) / 0.5);
        green.push((pixel[1] as f32 / 255.0 - 0.5) / 0.5);
        blue.push((pixel[2] as f32 / 255.0 - 0.5) / 0.5);
    }

    Ok(Tensor::new(vec![red, green, blue], device)?
        .reshape((3, 224, 224))?
        .unsqueeze(0)?)
}

pub fn preprocess_image(img: &[u8], device: &candle_core::Device) -> Result<Tensor> {
    let img = load_image(img)?;
    let img = img_resize(&img);
    img_to_vec(&img, device)
}
