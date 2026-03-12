use crate::api::models::{AppState, Prediction};
use crate::inference::preprocessing::preprocess_image;
use axum::Json;
use axum::body::Bytes;
use axum::extract::{Multipart, State};
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

pub async fn predict_image(
    state: State<AppState>,
    multipart: Multipart,
) -> Result<Json<Prediction>, StatusCode> {
    let image_bytes = extract_image(multipart).await?;
    let preprocessed_image = preprocess_image(&image_bytes, &state.device)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let prediction = state
        .model
        .predict(preprocessed_image)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(Prediction { label: prediction }))
}
