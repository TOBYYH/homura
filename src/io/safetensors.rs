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

/// Upper limit on the header size, as in the reference implementation.
const MAX_HEADER_LEN: u64 = 100_000_000;

/// A validated tensor entry of a safetensors header.
struct Entry<'a> {
    name: &'a str,
    dtype: DType,
    shape: Shape,
    begin: usize,
    end: usize,
}

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid safetensors data: {msg}"))
}

/// Parses one tensor entry, checking that its byte range matches its shape and dtype.
fn parse_entry<'a>(name: &'a str, info: &Json) -> Result<Entry<'a>> {
    let field = |key: &str| {
        info.get(key)
            .ok_or_else(|| bad(&format!("{name}: missing {key}")))
    };
    let dtype = parse_dtype(
        field("dtype")?
            .as_str()
            .ok_or_else(|| bad(&format!("{name}: dtype is not a string")))?,
    )?;
    let ints = |key: &str| -> Result<Vec<usize>> {
        field(key)?
            .as_array()
            .ok_or_else(|| bad(&format!("{name}: {key} is not an array")))?
            .iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| bad(&format!("{name}: {key} must hold unsigned integers")))
            })
            .collect()
    };
    let shape = ints("shape")?;
    let [begin, end] = ints("data_offsets")?[..] else {
        return Err(bad(&format!("{name}: data_offsets must have two entries")));
    };
    let nbytes = crate::shape::checked_num_elements(&shape)
        .and_then(|n| n.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| bad(&format!("{name}: shape too large")))?;
    if begin > end || end - begin != nbytes {
        return Err(bad(&format!(
            "{name}: data_offsets don't match the shape and dtype"
        )));
    }
    Ok(Entry {
        name,
        dtype,
        shape: Shape::new(&shape),
        begin,
        end,
    })
}

/// Parses safetensors data held in memory. Tensors are created on the CPU, in header order.
///
/// The whole header is validated before any tensor is created: names must be unique,
/// `__metadata__` must map strings to strings, and the tensors' byte ranges must exactly tile
/// the data section without overlaps, holes or trailing bytes.
#[track_caller]
pub fn load_safetensors_from_bytes(bytes: &[u8]) -> Result<Vec<(String, Tensor)>> {
    let len_bytes: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| bad("missing header length"))?
        .try_into()
        .unwrap();
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_LEN {
        return Err(bad("header is too large"));
    }
    let data_start = 8 + header_len as usize;
    if data_start > bytes.len() {
        return Err(bad("header length exceeds file size"));
    }
    let header =
        std::str::from_utf8(&bytes[8..data_start]).map_err(|_| bad("header is not UTF-8"))?;
    // The JSON parser rejects duplicate keys, so tensor names are unique.
    let header = json::parse(header)?;
    let members = header
        .as_object()
        .ok_or_else(|| bad("header is not an object"))?;
    let data = &bytes[data_start..];

    let mut entries = vec![];
    for (name, info) in members {
        if name == "__metadata__" {
            let valid = info
                .as_object()
                .is_some_and(|m| m.iter().all(|(_, v)| v.as_str().is_some()));
            if !valid {
                return Err(bad("__metadata__ must map strings to strings"));
            }
        } else {
            entries.push(parse_entry(name, info)?);
        }
    }
    let mut ranges: Vec<(usize, usize)> = entries.iter().map(|e| (e.begin, e.end)).collect();
    ranges.sort_unstable();
    let mut covered = 0;
    for (begin, end) in ranges {
        if begin != covered {
            return Err(bad("tensor data overlaps or leaves a gap"));
        }
        covered = end;
    }
    if covered != data.len() {
        return Err(bad("data section doesn't match the tensors in the header"));
    }

    let mut out = Vec::with_capacity(entries.len());
    for e in entries {
        let mut raw = data[e.begin..e.end].to_vec();
        if e.dtype == DType::Bool && raw.iter().any(|&b| b > 1) {
            return Err(bad(&format!(
                "{}: BOOL data must consist of the bytes 0 and 1",
                e.name
            )));
        }
        to_from_le(&mut raw, e.dtype.size_in_bytes());
        let t = Tensor::empty_on(e.shape, e.dtype, Device::Cpu);
        cpu::write_bytes(&t.data(), &raw);
        out.push((e.name.to_string(), t));
    }
    Ok(out)
}

/// Loads every tensor of a safetensors file onto the CPU, in header order.
///
/// The whole file is read into memory; memory mapping and sharded checkpoints come later.
#[track_caller]
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Vec<(String, Tensor)>> {
    load_safetensors_from_bytes(&std::fs::read(path)?)
}

/// Serializes tensors in the safetensors format.
///
/// # Errors
///
/// Returns an error for meta tensors (they have no data), duplicate names and the name
/// `__metadata__`, which the format reserves.
#[track_caller]
pub fn save_safetensors_to_bytes<'a, K: AsRef<str>>(
    tensors: impl IntoIterator<Item = (K, &'a Tensor)>,
) -> Result<Vec<u8>> {
    let mut header = String::from("{");
    let mut data: Vec<u8> = vec![];
    let mut names = std::collections::HashSet::new();
    for (i, (name, t)) in tensors.into_iter().enumerate() {
        let name = name.as_ref();
        if name == "__metadata__" {
            return Err(Error::Format(
                "the tensor name \"__metadata__\" is reserved by the safetensors format".into(),
            ));
        }
        if !names.insert(name.to_string()) {
            return Err(Error::Format(format!("duplicate tensor name {name:?}")));
        }
        if t.device() == Device::Meta {
            return Err(Error::Format(format!(
                "can't save meta tensor {name:?}: it has no data"
            )));
        }
        // Made contiguous at the data level, so that saving records nothing for autograd.
        let c = t.data();
        let c = if c.layout.is_contiguous() {
            c
        } else {
            crate::ops::clone_data(&c)
        };
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
#[track_caller]
pub fn save_safetensors<'a, K: AsRef<str>>(
    tensors: impl IntoIterator<Item = (K, &'a Tensor)>,
    path: impl AsRef<Path>,
) -> Result<()> {
    std::fs::write(path, save_safetensors_to_bytes(tensors)?)?;
    Ok(())
}
