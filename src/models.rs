use std::{collections::HashMap, fmt::Debug, sync::OnceLock};

pub trait ParamToLayer: Debug {
    fn parameter_layer(&self) -> &'static HashMap<&'static str, &'static str>;

    fn param_to_layer(&self, name: &str) -> Option<&str> {
        let layers = self.parameter_layer();
        for part in name.split('.').rev() {
            if let Some(layer) = layers.get(part) {
                return Some(layer);
            }
        }

        None
    }
}

static LLAMA_PARAMS: OnceLock<HashMap<&str, &str>> = OnceLock::new();

#[derive(Debug)]
pub struct Llama;

impl ParamToLayer for Llama {
    fn parameter_layer(&self) -> &'static HashMap<&'static str, &'static str> {
        LLAMA_PARAMS.get_or_init(|| {
            HashMap::from([
                ("q_proj", "Linear"),
                ("k_proj", "Linear"),
                ("v_proj", "Linear"),
                ("o_proj", "Linear"),
                ("gate_proj", "Linear"),
                ("down_proj", "Linear"),
                ("up_proj", "Linear"),
            ])
        })
    }
}

pub fn get_param_layer(model_type: &str) -> Option<Box<dyn ParamToLayer>> {
    match model_type {
        "llama" => Some(Box::new(Llama)),
        _ => None,
    }
}
