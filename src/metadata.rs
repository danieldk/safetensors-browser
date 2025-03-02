use std::collections::HashMap;
use std::fmt::Display;
use std::num::NonZeroUsize;
use std::str;
use std::{cmp::Ordering, str::FromStr};

use color_eyre::eyre::Result;
use num_bigint::BigInt;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use safetensors::tensor::TensorInfo;

use crate::config::{AwqVersion, Config, QuantizationConfig};
use crate::repo::CheckpointMetadata;

#[derive(Debug)]
pub struct TensorMetadata {
    pub name: String,
    pub tensor_info: TensorInfo,
    pub checkpoint: String,
    pub quantization: Option<Quantization>,
}

#[derive(Debug)]
pub struct Quantization {
    dtype: QuantizedDType,
    group_size: usize,
    qtype: QuantizationType,
    zero_point: bool,
}

#[derive(Clone, Debug)]
enum LinearLayout {
    InFeaturesOutFeatures,
    OutFeaturesInFeatures,
}

#[derive(Clone, Debug)]
struct DequantizedShape {
    shape: Vec<usize>,
    layout: LinearLayout,
}

impl RenderMetadata for DequantizedShape {
    fn render_metadata(&self, _tensor_metadata: &TensorMetadata, lines: &mut Vec<Line>) {
        let field_style = Style::new().magenta();
        lines.push(Line::from(vec![
            Span::styled("Dequantized shape: ", field_style),
            Span::raw(format!("{:?}", self.shape)),
        ]));

        match self.layout {
            LinearLayout::InFeaturesOutFeatures => {
                lines.push(Line::from(vec![
                    Span::styled("Linear shape: ", field_style),
                    Span::raw(format!(
                        "{:?}",
                        self.shape.iter().rev().cloned().collect::<Vec<_>>()
                    )),
                ]));
            }
            LinearLayout::OutFeaturesInFeatures => (),
        };
    }
}

impl Quantization {
    fn dequantized_weight_shape(&self, quantized_shape: &[usize]) -> Option<DequantizedShape> {
        match self.qtype {
            QuantizationType::Awq {
                version: AwqVersion::Gemm,
            } => {
                let n_packed = 32 / self.dtype.n_bits();
                let mut dequantized_shape = quantized_shape.to_owned();
                let last = dequantized_shape.last_mut()?;
                *last *= n_packed;
                Some(DequantizedShape {
                    shape: dequantized_shape,
                    layout: LinearLayout::InFeaturesOutFeatures,
                })
            }
            QuantizationType::Gptq { .. } => {
                let n_packed = 32 / self.dtype.n_bits();
                let mut dequantized_shape = quantized_shape.to_owned();
                let first = dequantized_shape.first_mut()?;
                *first *= n_packed;
                Some(DequantizedShape {
                    shape: dequantized_shape,
                    layout: LinearLayout::InFeaturesOutFeatures,
                })
            }
            _ => None,
        }
    }

    fn dequantized_zero_point_shape(&self, quantized_shape: &[usize]) -> Option<DequantizedShape> {
        match self.qtype {
            QuantizationType::Awq {
                version: AwqVersion::Gemm,
            }
            | QuantizationType::Gptq { .. } => {
                let n_packed = 32 / self.dtype.n_bits();
                let mut dequantized_shape = quantized_shape.to_owned();
                let last = dequantized_shape.last_mut()?;
                *last *= n_packed;
                Some(DequantizedShape {
                    shape: dequantized_shape,
                    layout: LinearLayout::OutFeaturesInFeatures,
                })
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum QuantizationType {
    Awq { version: AwqVersion },
    Gptq { desc_act: bool, static_groups: bool },
}

impl Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationType::Awq { .. } => f.write_str("AWQ"),
            QuantizationType::Gptq { .. } => f.write_str("GPTQ"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum QuantizedDType {
    Int(NonZeroUsize),
}

impl QuantizedDType {
    fn n_bits(&self) -> NonZeroUsize {
        match self {
            QuantizedDType::Int(bits) => *bits,
        }
    }
}

impl Display for QuantizedDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizedDType::Int(bits) => write!(f, "I{}", bits),
        }
    }
}

impl Quantization {
    fn new(config: &Config) -> Option<Self> {
        let quantization_config = config.quantization_config.as_ref()?;
        match quantization_config {
            QuantizationConfig::Awq {
                bits,
                group_size,
                version,
                zero_point,
                ..
            } => Some(Quantization {
                dtype: QuantizedDType::Int(*bits),
                group_size: *group_size,
                qtype: QuantizationType::Awq { version: *version },
                zero_point: *zero_point,
            }),
            QuantizationConfig::Gptq {
                bits,
                desc_act,
                group_size,
                static_groups,
                sym,
            } => Some(Quantization {
                dtype: QuantizedDType::Int(*bits),
                group_size: *group_size,
                qtype: QuantizationType::Gptq {
                    desc_act: *desc_act,
                    static_groups: *static_groups,
                },
                zero_point: !sym,
            }),
        }
    }
}

pub fn get_tensors(
    config: &Config,
    checkpoint_metadatas: &[CheckpointMetadata],
) -> Result<HashMap<String, TensorMetadata>> {
    let mut tensors = HashMap::new();
    for metadata in checkpoint_metadatas {
        tensors.extend(
            metadata
                .metadata
                .tensors()
                .into_iter()
                .map(|(name, tensor_info)| {
                    (
                        name.clone(),
                        TensorMetadata {
                            checkpoint: metadata.filename.clone(),
                            name,
                            quantization: Quantization::new(config),
                            tensor_info: tensor_info.clone(),
                        },
                    )
                }),
        );
    }

    Ok(tensors)
}

pub trait RenderMetadata {
    fn render_metadata(&self, tensor_metadata: &TensorMetadata, lines: &mut Vec<Line>);
}

impl RenderMetadata for TensorMetadata {
    fn render_metadata(&self, tensor_metadata: &TensorMetadata, lines: &mut Vec<Line>) {
        let field_style = Style::new().magenta();
        lines.extend([
            Line::from(vec![
                Span::styled("Name: ", field_style),
                Span::raw(tensor_metadata.name.clone()),
            ]),
            Line::from(vec![
                Span::styled("File: ", field_style),
                Span::raw(tensor_metadata.checkpoint.clone()),
            ]),
            Line::from(vec![
                Span::styled("DType: ", field_style),
                Span::raw(format!("{:?}", tensor_metadata.tensor_info.dtype)),
            ]),
            Line::from(vec![
                Span::styled("Shape: ", field_style),
                Span::raw(format!("{:?}", tensor_metadata.tensor_info.shape)),
            ]),
            Line::from(vec![
                Span::styled("Offsets: ", field_style),
                Span::raw(format!("{:?}", tensor_metadata.tensor_info.data_offsets)),
            ]),
        ]);

        if let Some(quantization) = tensor_metadata.quantization.as_ref() {
            quantization.render_metadata(tensor_metadata, lines);
        }
    }
}

impl RenderMetadata for Quantization {
    fn render_metadata(&self, tensor_metadata: &TensorMetadata, lines: &mut Vec<Line>) {
        let name = tensor_metadata
            .name
            .rsplit_once('.')
            .map(|parts| parts.1)
            .unwrap_or(&tensor_metadata.name);

        if !(name == "qweight" || name == "qzeros" || name == "scales") {
            return;
        }

        let field_style = Style::new().magenta();
        lines.extend([
            Line::default(),
            Line::from(vec![Span::styled(
                "Quantization",
                ratatui::style::Style::new().blue().underlined(),
            )]),
            Line::default(),
            Line::from(vec![
                Span::styled("Quantizer: ", field_style),
                Span::raw(self.qtype.to_string()),
            ]),
            Line::from(vec![
                Span::styled("DType: ", field_style),
                Span::raw(self.dtype.to_string()),
            ]),
            Line::from(vec![
                Span::styled("Group size: ", field_style),
                Span::raw(self.group_size.to_string()),
            ]),
            Line::from(vec![
                Span::styled("Zero point: ", field_style),
                Span::raw(self.zero_point.to_string()),
            ]),
        ]);

        let dequantized_shape = match name {
            "qweight" => self.dequantized_weight_shape(&tensor_metadata.tensor_info.shape),
            "qzeros" => self.dequantized_zero_point_shape(&tensor_metadata.tensor_info.shape),
            "scales" => None,
            _ => unreachable!(),
        };

        if let Some(dequantized_shape) = dequantized_shape {
            dequantized_shape.render_metadata(tensor_metadata, lines);
        }
    }
}

pub fn cmp_numeric_lexicographic(s1: &str, s2: &str) -> Ordering {
    let mut b1 = s1.as_bytes();
    let mut b2 = s2.as_bytes();

    while !b1.is_empty() && !b2.is_empty() {
        if b1[0].is_ascii_digit() && b2[0].is_ascii_digit() {
            // Do a numerical compare if we encounter some digits.
            let b1_digits = count_digit_bytes(b1);
            let b2_digits = count_digit_bytes(b2);

            // Unwraps are safe. A run of ASCII digits is always valid
            // UTF-8 and always a valid number.
            let num1 = BigInt::from_str(str::from_utf8(&b1[..b1_digits]).unwrap()).unwrap();
            let num2 = BigInt::from_str(str::from_utf8(&b2[..b2_digits]).unwrap()).unwrap();

            match num1.cmp(&num2) {
                Ordering::Equal => {
                    b1 = &b1[b1_digits..];
                    b2 = &b2[b2_digits..];
                }
                ord => return ord,
            }
        } else {
            // If the byte is not a digit, do a lexicographical compare.
            match b1[0].cmp(&b2[0]) {
                Ordering::Equal => {
                    b1 = &b1[1..];
                    b2 = &b2[1..];
                }
                ord => return ord,
            }
        }
    }

    b1.cmp(b2)
}

fn count_digit_bytes(b: &[u8]) -> usize {
    b.iter().take_while(|b| b.is_ascii_digit()).count()
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use crate::metadata::cmp_numeric_lexicographic;

    #[test]
    fn test_cmp_lexicographic_numeric() {
        assert_eq!(cmp_numeric_lexicographic("aaa", "aaa"), Ordering::Equal);
        assert_eq!(cmp_numeric_lexicographic("aaa", "aa"), Ordering::Greater);
        assert_eq!(cmp_numeric_lexicographic("aa", "aaa"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("aaa", "aab"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("baa", "aaa"), Ordering::Greater);
        assert_eq!(cmp_numeric_lexicographic("aaa1", "aaa2"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("1aaa", "2aaa"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("aaa1a", "aaa2a"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("aaa1a", "aaa11a"), Ordering::Less);
        assert_eq!(
            cmp_numeric_lexicographic("aaa2a", "aaa1a"),
            Ordering::Greater
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa11a", "aaa1a"),
            Ordering::Greater
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa11a", "aaa011a"),
            Ordering::Equal
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa1abb1b", "aaa1abb1b"),
            Ordering::Equal
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa1abb1b", "aaa1abb12b"),
            Ordering::Less
        );
    }
}
