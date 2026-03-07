use anyhow::Result;
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
