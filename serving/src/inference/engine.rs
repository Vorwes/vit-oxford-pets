use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::vit::{Config, Model as VisionTransformer};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
struct ModelConfig {
    id2label: HashMap<String, String>,
}

fn load_model(model_path: &str) -> Result<VisionTransformer> {
    let config_path = fs::read_to_string(format!("{}/config.json", model_path))?;

    let config: Config = serde_json::from_str(&config_path)?;

    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[format!("{}/model.safetensors", model_path)],
            DType::F32,
            &device,
        )?
    };

    let model = VisionTransformer::new(&config, 37, vb)?;
    Ok(model)
}
