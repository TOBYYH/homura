//! Reading and writing safetensors files.
//!
//! The format is an 8-byte little-endian header length, a JSON header describing each tensor
//! (`dtype`, `shape` and `data_offsets` into the data section) and the raw little-endian data.

use std::path::Path;

use super::json::{self, Json};
use crate::backend::cpu;
use crate::{DType, Device, Error, Result, Shape, Tensor};

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "BOOL",
        DType::U8 => "U8",
        DType::I8 => "I8",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
    }
}

fn parse_dtype(name: &str) -> Result<DType> {
    DType::ALL
        .into_iter()
        .find(|&d| dtype_name(d) == name)
        .ok_or_else(|| Error::Format(format!("unsupported safetensors dtype {name:?}")))
}

/// Reverses the bytes of each element on big-endian hosts (the file format is little-endian).
fn to_from_le(bytes: &mut [u8], elem_size: usize) {
    if cfg!(target_endian = "big") && elem_size > 1 {
        for chunk in bytes.chunks_mut(elem_size) {
            chunk.reverse();
        }
    }
}

/// Parses safetensors data held in memory. Tensors are created on the CPU, in header order.
pub fn load_safetensors_from_bytes(bytes: &[u8]) -> Result<Vec<(String, Tensor)>> {
    let bad = |msg: &str| Error::Format(format!("invalid safetensors data: {msg}"));
    let len_bytes: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| bad("missing header length"))?
        .try_into()
        .unwrap();
    let header_len = u64::from_le_bytes(len_bytes);
    let data_start = 8u64
        .checked_add(header_len)
        .filter(|&n| n <= bytes.len() as u64);
    let data_start = data_start.ok_or_else(|| bad("header length exceeds file size"))? as usize;
    let header =
        std::str::from_utf8(&bytes[8..data_start]).map_err(|_| bad("header is not UTF-8"))?;
    let header = json::parse(header)?;
    let members = header
        .as_object()
        .ok_or_else(|| bad("header is not an object"))?;
    let data = &bytes[data_start..];

    let mut out = vec![];
    for (name, info) in members {
        if name == "__metadata__" {
            continue;
        }
        let field = |key: &str| {
            info.get(key)
                .ok_or_else(|| bad(&format!("{name}: missing {key}")))
        };
        let dtype = parse_dtype(
            field("dtype")?
                .as_str()
                .ok_or_else(|| bad("dtype is not a string"))?,
        )?;
        let ints = |v: &Json| -> Result<Vec<u64>> {
            v.as_array()
                .ok_or_else(|| bad("expected an array"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .ok_or_else(|| bad("expected an unsigned integer"))
                })
                .collect()
        };
        let shape: Vec<usize> = ints(field("shape")?)?
            .into_iter()
            .map(|d| d as usize)
            .collect();
        let offsets = ints(field("data_offsets")?)?;
        let [begin, end] = offsets[..] else {
            return Err(bad(&format!("{name}: data_offsets must have two entries")));
        };
        let shape = Shape::new(&shape);
        let nbytes = shape
            .iter()
            .try_fold(dtype.size_in_bytes(), |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad(&format!("{name}: shape too large")))?;
        if begin > end || end > data.len() as u64 || end - begin != nbytes as u64 {
            return Err(bad(&format!(
                "{name}: data_offsets don't match the shape and dtype"
            )));
        }
        let mut raw = data[begin as usize..end as usize].to_vec();
        if dtype == DType::Bool && raw.iter().any(|&b| b > 1) {
            return Err(bad(&format!(
                "{name}: BOOL data must consist of the bytes 0 and 1"
            )));
        }
        to_from_le(&mut raw, dtype.size_in_bytes());
        let t = Tensor::empty_on(shape, dtype, Device::Cpu);
        cpu::write_bytes(&t.data(), &raw);
        out.push((name.clone(), t));
    }
    Ok(out)
}

/// Loads every tensor of a safetensors file onto the CPU, in header order.
///
/// The whole file is read into memory; memory mapping and sharded checkpoints come later.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Vec<(String, Tensor)>> {
    load_safetensors_from_bytes(&std::fs::read(path)?)
}

/// Serializes tensors in the safetensors format.
///
/// # Errors
///
/// Returns an error for meta tensors (they have no data) and duplicate names.
pub fn save_safetensors_to_bytes<'a, K: AsRef<str>>(
    tensors: impl IntoIterator<Item = (K, &'a Tensor)>,
) -> Result<Vec<u8>> {
    let mut header = String::from("{");
    let mut data: Vec<u8> = vec![];
    let mut names = std::collections::HashSet::new();
    for (i, (name, t)) in tensors.into_iter().enumerate() {
        let name = name.as_ref();
        if !names.insert(name.to_string()) {
            return Err(Error::Format(format!("duplicate tensor name {name:?}")));
        }
        if t.device() == Device::Meta {
            return Err(Error::Format(format!(
                "can't save meta tensor {name:?}: it has no data"
            )));
        }
        let c = t.contiguous().data();
        let mut raw = cpu::read_bytes(&c);
        to_from_le(&mut raw, c.dtype.size_in_bytes());
        if i > 0 {
            header.push(',');
        }
        json::write_string(&mut header, name);
        let shape: Vec<String> = c.layout.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            ":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            dtype_name(c.dtype),
            shape.join(","),
            data.len(),
            data.len() + raw.len()
        ));
        data.extend_from_slice(&raw);
    }
    header.push('}');
    // Pad the header with spaces so the data starts 8-byte aligned.
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    Ok(out)
}

/// Writes tensors to a safetensors file.
///
/// ```no_run
/// let w = homura::randn([2, 3]);
/// homura::save_safetensors([("weight", &w)], "weights.safetensors")?;
/// # Ok::<(), homura::Error>(())
/// ```
pub fn save_safetensors<'a, K: AsRef<str>>(
    tensors: impl IntoIterator<Item = (K, &'a Tensor)>,
    path: impl AsRef<Path>,
) -> Result<()> {
    std::fs::write(path, save_safetensors_to_bytes(tensors)?)?;
    Ok(())
}
