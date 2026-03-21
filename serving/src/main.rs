mod api;
mod inference;

use api::models::AppState;
use axum::{Router, routing::post};
use candle_core::Device;
use inference::engine::ImageClassifier;
use std::sync::Arc;
use tokio::net::TcpListener;

use api::routes::predict_image;

#[tokio::main]
async fn main() {
    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "../models/vit-pets-final".to_string());
    let device = Device::Cpu;

    let model = Arc::new(ImageClassifier::new(model_path.as_str()).expect("Failed to load model"));
    let app_state = AppState { model, device };

    let app = Router::new()
        .route("/predict", post(predict_image))
        .with_state(app_state);
    let listener = TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.unwrap();
}
