use crate::api::models::{AppState, Prediction};
use crate::inference::preprocessing::preprocess_image;
use axum::body::Bytes;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::{
    Json, Router,
    routing::{get, post},
};

async fn extract_image(mut multipart: Multipart) -> Result<Bytes, StatusCode> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
    {
        if field.name() == Some("image") {
            return field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST);
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

async fn health() -> (StatusCode, &'static str) {
    let status = StatusCode::OK;
    let reason = status.canonical_reason().unwrap_or("");

    (status, reason)
}

pub fn create_router(app_state: AppState) -> Router {
    Router::new()
        .route("/predict", post(predict_image))
        .route("/health", get(health))
        .with_state(app_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::engine::ImageClassifier;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum_test::TestServer;
    use axum_test::multipart::{MultipartForm, Part};
    use candle_core::Device;
    use serde_json::json;
    use std::sync::Arc;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn health_check_works() {
        let model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "../models/vit-pets-final".to_string());
        let device = Device::Cpu;

        let model =
            Arc::new(ImageClassifier::new(model_path.as_str()).expect("Failed to load model"));
        let app_state = AppState { model, device };

        let app = create_router(app_state);
        let request = Request::builder()
            .uri("/health")
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn predict_image_empty_body() {
        let model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "../models/vit-pets-final".to_string());
        let device = Device::Cpu;

        let model =
            Arc::new(ImageClassifier::new(model_path.as_str()).expect("Failed to load model"));
        let app_state = AppState { model, device };

        let app = create_router(app_state);
        let request = Request::builder()
            .uri("/predict")
            .method("POST")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn predict_image_correctly() {
        let model_path =
            std::env::var("MODEL_PATH").unwrap_or_else(|_| "../models/vit-pets-final".to_string());
        let device = Device::Cpu;

        let model =
            Arc::new(ImageClassifier::new(model_path.as_str()).expect("Failed to load model"));
        let app_state = AppState { model, device };

        let app = create_router(app_state);

        let server = TestServer::new(app);
        let img = include_bytes!("../../../test_images/test_image.jpg");

        let part = Part::bytes(img.as_slice())
            .file_name("test_image.jpg")
            .mime_type("image/jpeg");

        let form = MultipartForm::new().add_part("image", part);

        let response = server.post("/predict").multipart(form).await;

        response.assert_json(&json!({
            "label": "yorkshire_terrier",
        }));
    }
}
