use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub label: String,
}
