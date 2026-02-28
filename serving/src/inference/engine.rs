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

fn load_labels(model_path: &str) -> Result<ModelConfig> {
    let config_path = fs::read_to_string(format!("{}/config.json", &model_path))?;

    let label_config: ModelConfig = serde_json::from_str(&config_path)?;
    Ok(label_config)
}

fn get_label(config: &ModelConfig, max_index: u32) -> Result<String> {
    Ok(config
        .id2label
        .get(&max_index.to_string())
        .unwrap_or(&"Unknown".to_string())
        .clone())
}

fn predict(model: &VisionTransformer, img: Tensor, config: &ModelConfig) -> Result<String> {
    let logits = model.forward(&img)?;

    let max_index = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
    Ok(get_label(&config, max_index)?)
}
