use std::path::PathBuf;
use std::str;
use std::{cmp::Ordering, str::FromStr};
use std::{collections::HashMap, fs::File, path::Path};

use color_eyre::eyre::{Context, Result};
use memmap2::Mmap;
use num_bigint::BigInt;
use safetensors::{tensor::TensorInfo, SafeTensors};

pub struct TensorMetadata {
    pub tensor_info: TensorInfo,
    pub checkpoint: PathBuf,
}

pub fn get_tensors<P>(checkpoint_paths: &[P]) -> Result<HashMap<String, TensorMetadata>>
where
    P: AsRef<Path>,
{
    let mut tensors = HashMap::new();
    for path in checkpoint_paths {
        let path = path.as_ref();
        let f = File::open(&path)
            .with_context(|| format!("Cannot open checkpoint: {}", path.to_string_lossy()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        let (_, metadata) = SafeTensors::read_metadata(&mmap)?;
        tensors.extend(metadata.tensors().into_iter().map(|(name, tensor_info)| {
            (
                name,
                TensorMetadata {
                    checkpoint: path.to_owned(),
                    tensor_info: tensor_info.clone(),
                },
            )
        }));
    }

    Ok(tensors)
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
