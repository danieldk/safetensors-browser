use std::num::NonZeroUsize;

use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model_type: String,

    // For quantization, we'd rather not show any information than
    // fail when we don't know the quantization method.
    #[serde(deserialize_with = "ok_or_none")]
    pub quantization_config: Option<QuantizationConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", tag = "quant_method")]
pub enum QuantizationConfig {
    Awq {
        bits: NonZeroUsize,
        group_size: usize,
        version: AwqVersion,
        zero_point: bool,
    },
    Gptq {
        bits: NonZeroUsize,
        desc_act: bool,
        group_size: usize,
        static_groups: bool,
        sym: bool,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AwqVersion {
    Gemm,
    Gemmv,
    GemmvFast,
}

fn ok_or_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let v = Value::deserialize(deserializer)?;
    Ok(T::deserialize(v).ok())
}
