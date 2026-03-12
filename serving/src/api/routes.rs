use axum::body::Bytes;
use axum::extract::Multipart;
use axum::http::StatusCode;

async fn extract_image(mut multipart: Multipart) -> Result<Bytes, StatusCode> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
    {
        if field.name() == Some("image") {
            return Ok(field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?);
        }
    }
    Err(StatusCode::BAD_REQUEST)
}
