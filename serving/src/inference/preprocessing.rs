use anyhow::Result;
use image::ImageReader;
use std::io::Cursor;

fn load_image(img: &[u8]) -> Result<image::DynamicImage> {
    Ok(ImageReader::new(Cursor::new(img))
        .with_guessed_format()?
        .decode()?)
}
