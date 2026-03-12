use serde::Serialize;
use std::sync::Arc;

use crate::inference::engine::ImageClassifier;

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub model: Arc<ImageClassifier>,
}
