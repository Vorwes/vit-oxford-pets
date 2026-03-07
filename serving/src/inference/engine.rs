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

#[derive(Debug)]
pub struct ImageClassifier {
    model: VisionTransformer,
    config: ModelConfig,
}

impl ImageClassifier {
    pub fn new(model_path: &str) -> Result<Self> {
        let raw_config = fs::read_to_string(format!("{}/config.json", model_path))?;

        let config: Config = serde_json::from_str(&raw_config)?;
        let label_config: ModelConfig = serde_json::from_str(&raw_config)?;
        let num_labels: usize = label_config.id2label.len();

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[format!("{}/model.safetensors", model_path)],
                DType::F32,
                &device,
            )?
        };

        let model = VisionTransformer::new(&config, num_labels, vb)?;

        Ok(Self {
            model,
            config: label_config,
        })
    }

    fn get_label(&self, max_index: u32) -> Result<String> {
        Ok(self
            .config
            .id2label
            .get(&max_index.to_string())
            .unwrap_or(&"Unknown".to_string())
            .clone())
    }

    pub fn predict(&self, img: Tensor) -> Result<String> {
        let logits = self.model.forward(&img)?;

        let max_index = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
        self.get_label(max_index)
    }
}
