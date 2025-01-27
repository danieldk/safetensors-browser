use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub quantization_config: Option<QuantizationConfig>,
}

impl Config {
    pub fn layer_quantizers(&self) -> HashMap<String, QuantizationGroup> {
        let mut layer_quantizers = HashMap::new();

        let quantization_config = match &self.quantization_config {
            Some(config) => config,
            None => return layer_quantizers,
        };

        let quantization_groups = match &quantization_config.config_groups {
            Some(groups) => groups,
            None => return layer_quantizers,
        };

        for group in quantization_groups.values() {
            for target in &group.targets {
                layer_quantizers.insert(target.to_string(), group.clone());
            }
        }

        layer_quantizers
    }
}

#[derive(Debug, Deserialize)]
pub struct QuantizationConfig {
    config_groups: Option<HashMap<String, QuantizationGroup>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuantizationGroup {
    pub input_activations: Option<QuantizationSpec>,
    pub output_activations: Option<QuantizationSpec>,
    pub targets: Vec<String>,
    pub weights: Option<QuantizationSpec>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuantizationSpec {
    pub dynamic: bool,
    pub symmetric: bool,
    pub strategy: String,
    #[serde(rename = "type")]
    pub dtype: String,
}
