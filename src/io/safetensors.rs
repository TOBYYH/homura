//! Reading and writing safetensors files.
//!
//! The format is an 8-byte little-endian header length, a JSON header describing each tensor
//! (`dtype`, `shape` and `data_offsets` into the data section) and the raw little-endian data.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use super::json::{self, Json};
use crate::backend::cpu;
use crate::nn::StateDict;
use crate::storage::{FileMap, Storage};
use crate::tensor::{Layout, TensorData};
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
struct Entry {
    name: String,
    dtype: DType,
    shape: Shape,
    begin: usize,
    end: usize,
}

/// A validated safetensors header.
struct Header {
    metadata: Vec<(String, String)>,
    /// In header order.
    entries: Vec<Entry>,
    /// Where the data section starts in the file.
    data_start: u64,
    /// The size of the file, where the data section ends.
    len: u64,
}

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid safetensors data: {msg}"))
}

/// Parses one tensor entry, checking that its byte range matches its shape and dtype.
fn parse_entry(name: &str, info: &Json) -> Result<Entry> {
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
        name: name.to_string(),
        dtype,
        shape: Shape::new(&shape),
        begin,
        end,
    })
}

/// Reads and validates the header of the safetensors data in `r`, leaving the data alone.
///
/// Every length is compared with the size of the data before anything is allocated: the header
/// must fit, names must be unique, `__metadata__` must map strings to strings, and the tensors'
/// byte ranges must exactly tile the data section without overlaps, holes or trailing bytes.
fn read_header<R: Read + Seek>(r: &mut R) -> Result<Header> {
    let len = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    if len < 8 {
        return Err(bad("missing header length"));
    }
    let mut len_bytes = [0; 8];
    r.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_LEN {
        return Err(bad("header is too large"));
    }
    let data_start = 8 + header_len;
    if data_start > len {
        return Err(bad("header length exceeds file size"));
    }
    let mut header = vec![0; header_len as usize];
    r.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header).map_err(|_| bad("header is not UTF-8"))?;
    // The JSON parser rejects duplicate keys, so tensor names are unique.
    let header = json::parse(header)?;
    let members = header
        .as_object()
        .ok_or_else(|| bad("header is not an object"))?;

    let mut metadata = vec![];
    let mut entries = vec![];
    for (name, info) in members {
        if name == "__metadata__" {
            let pairs = info.as_object().and_then(|m| {
                m.iter()
                    .map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                    .collect::<Option<Vec<_>>>()
            });
            metadata = pairs.ok_or_else(|| bad("__metadata__ must map strings to strings"))?;
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
    if covered as u64 != len - data_start {
        return Err(bad("data section doesn't match the tensors in the header"));
    }
    Ok(Header {
        metadata,
        entries,
        data_start,
        len,
    })
}

/// Checks the BOOL invariant (see `Storage::cpu_bytes_mut`) before a tensor gets the data.
fn check_bool(e: &Entry, bytes: &[u8]) -> Result<()> {
    if e.dtype == DType::Bool && bytes.iter().any(|&b| b > 1) {
        return Err(bad(&format!(
            "{}: BOOL data must consist of the bytes 0 and 1",
            e.name
        )));
    }
    Ok(())
}

fn tensor(e: Entry, storage: Storage) -> (String, Tensor) {
    let t = Tensor::from_data(TensorData {
        storage: Arc::new(storage),
        layout: Layout::contiguous(e.shape),
        dtype: e.dtype,
    });
    (e.name, t)
}

/// Reads the tensors of the safetensors data in `r` onto the CPU, in header order.
///
/// Each tensor's data is read straight into its new storage, so the peak memory is about the
/// total size of the tensors. BOOL data is checked to consist of the bytes 0 and 1 before the
/// tensor is created.
pub(super) fn read<R: Read + Seek>(r: &mut R) -> Result<Vec<(String, Tensor)>> {
    let header = read_header(r)?;
    let mut pos = None;
    let mut out = Vec::with_capacity(header.entries.len());
    for e in header.entries {
        let start = header.data_start + e.begin as u64;
        if pos != Some(start) {
            r.seek(SeekFrom::Start(start))?;
        }
        let mut storage = Storage::new(e.end - e.begin, Device::Cpu);
        let bytes = storage.cpu_bytes_mut();
        r.read_exact(bytes)?;
        pos = Some(header.data_start + e.end as u64);
        check_bool(&e, bytes)?;
        to_from_le(bytes, e.dtype.size_in_bytes());
        out.push(tensor(e, storage));
    }
    Ok(out)
}

/// Like [`read`], but the tensors use the pages of a private copy-on-write mapping of `file`
/// instead of copies, where they can: tensors whose data starts at a multiple of their element
/// size, all of them for one-byte dtypes. Other tensors, multi-byte tensors on big-endian hosts
/// (the file is little-endian) and empty tensors get their own memory, copied from the mapping.
/// BOOL data is checked on the mapping, touching only its pages, before the tensor is created.
///
/// # Safety
///
/// As for [`load_mmap`](super::load_mmap): `file` must not be truncated or modified while any
/// tensor that maps it is alive.
pub(super) unsafe fn read_mmap(file: &mut File) -> Result<Vec<(String, Tensor)>> {
    let header = read_header(file)?;
    // SAFETY: the caller keeps the file unchanged while the mapping lives.
    let map = unsafe { FileMap::new(file)? };
    if map.len() as u64 != header.len {
        return Err(Error::Io(std::io::Error::other(
            "the file changed while it was being loaded",
        )));
    }
    // The file fits the address space now that it is mapped, so its offsets fit `usize`.
    let data_start = header.data_start as usize;
    let mut out = Vec::with_capacity(header.entries.len());
    for e in header.entries {
        let offset = data_start + e.begin;
        let nbytes = e.end - e.begin;
        let size = e.dtype.size_in_bytes();
        if nbytes == 0 {
            out.push(tensor(e, Storage::new(0, Device::Cpu)));
            continue;
        }
        // SAFETY: the header check makes the entries' byte ranges disjoint, and each one gets
        // a single storage.
        let mut mapped = unsafe { Storage::mapped(map.clone(), offset, nbytes) };
        check_bool(&e, mapped.cpu_bytes_mut())?;
        let storage = if size > 1 && (cfg!(target_endian = "big") || !offset.is_multiple_of(size)) {
            let mut copy = Storage::new(nbytes, Device::Cpu);
            let bytes = copy.cpu_bytes_mut();
            bytes.copy_from_slice(mapped.cpu_bytes_mut());
            to_from_le(bytes, size);
            copy
        } else {
            mapped
        };
        out.push(tensor(e, storage));
    }
    Ok(out)
}

/// Parses safetensors data held in memory. Tensors are created on the CPU, in header order.
///
/// The whole header is validated before any tensor is created: names must be unique,
/// `__metadata__` must map strings to strings, and the tensors' byte ranges must exactly tile
/// the data section without overlaps, holes or trailing bytes. BOOL data must consist of the
/// bytes 0 and 1 (Python's `safetensors` accepts other bytes).
#[track_caller]
pub fn load_safetensors_from_bytes(bytes: &[u8]) -> Result<Vec<(String, Tensor)>> {
    read(&mut Cursor::new(bytes))
}

/// Loads every tensor of a safetensors file onto the CPU, in header order.
///
/// The file is validated as [`load_safetensors_from_bytes`] describes. Each tensor's data is
/// read from the file straight into its storage, without reading the whole file into memory
/// first. [`homura::load`](crate::load) does the same and returns a
/// [`StateDict`](crate::nn::StateDict).
#[track_caller]
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Vec<(String, Tensor)>> {
    read(&mut File::open(path)?)
}

/// The `__metadata__` of a safetensors file: string keys and values, empty if the file has none.
///
/// Only the header is read, but all of it is validated as for loading.
///
/// ```no_run
/// let metadata = homura::io::safetensors_metadata("model.safetensors")?;
/// if metadata.get("format").is_some_and(|f| f == "pt") {
///     println!("saved from PyTorch");
/// }
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub fn safetensors_metadata(path: impl AsRef<Path>) -> Result<BTreeMap<String, String>> {
    Ok(read_header(&mut File::open(path)?)?
        .metadata
        .into_iter()
        .collect())
}

/// Takes a snapshot of the tensors to save, checking their names and devices first.
pub(super) fn snapshot<'a, K: AsRef<str>>(
    tensors: impl IntoIterator<Item = (K, &'a Tensor)>,
) -> Result<Vec<(String, TensorData)>> {
    let mut names = std::collections::HashSet::new();
    let mut out = vec![];
    for (name, t) in tensors {
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
        out.push((name.to_string(), t.data()));
    }
    Ok(out)
}

/// The size of the data of a tensor in a safetensors file.
pub(super) fn data_size(data: &TensorData) -> usize {
    data.layout.num_elements() * data.dtype.size_in_bytes()
}

/// Writes `tensors` (from [`snapshot`]) in the safetensors format, with `__metadata__` if
/// `metadata` is given. Tensors are converted and written one at a time.
pub(super) fn write<W: Write>(
    w: &mut W,
    tensors: &[(String, TensorData)],
    metadata: Option<&BTreeMap<String, String>>,
) -> Result<()> {
    let mut header = String::from("{");
    if let Some(metadata) = metadata {
        header.push_str("\"__metadata__\":{");
        for (i, (k, v)) in metadata.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            json::write_string(&mut header, k);
            header.push(':');
            json::write_string(&mut header, v);
        }
        header.push('}');
    }
    let mut offset = 0;
    for (name, data) in tensors {
        if header.len() > 1 {
            header.push(',');
        }
        json::write_string(&mut header, name);
        let shape: Vec<String> = data.layout.shape.iter().map(|d| d.to_string()).collect();
        let end = offset + data_size(data);
        header.push_str(&format!(
            ":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            dtype_name(data.dtype),
            shape.join(","),
        ));
        offset = end;
    }
    header.push('}');
    // Pad the header with spaces so the data starts 8-byte aligned.
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for (_, data) in tensors {
        // Made contiguous at the data level, so that saving records nothing for autograd.
        let c = if data.layout.is_contiguous() {
            data.clone()
        } else {
            crate::ops::clone_data(data)
        };
        let mut raw = cpu::read_bytes(&c);
        to_from_le(&mut raw, c.dtype.size_in_bytes());
        w.write_all(&raw)?;
    }
    Ok(())
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
    let mut out = vec![];
    write(&mut out, &snapshot(tensors)?, None)?;
    Ok(out)
}

/// Writes tensors to a safetensors file.
///
/// The tensors are checked as for [`save_safetensors_to_bytes`] before the file is created, and
/// written to it one at a time.
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
    let tensors = snapshot(tensors)?;
    write(&mut File::create(path)?, &tensors, None)
}

/// Writes a state dict to a safetensors file with `metadata` as its `__metadata__`, like
/// `safetensors.torch.save_file(state_dict, path, metadata)`.
///
/// The `__metadata__` entry is written even when `metadata` is empty, as Python's `safetensors`
/// does for an empty dict. The format has no place for the state dict's
/// [versions](StateDict::versions), and each key gets its own copy of the data, even for tensors
/// that share storage. Errors as for [`save_safetensors_to_bytes`].
///
/// ```no_run
/// use std::collections::BTreeMap;
///
/// let model = homura::nn::Linear::new(4, 2);
/// # use homura::nn::Module;
/// let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
/// homura::io::save_safetensors_with_metadata(&model.state_dict(), "model.safetensors", &metadata)?;
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub fn save_safetensors_with_metadata(
    state_dict: &StateDict,
    path: impl AsRef<Path>,
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let tensors = snapshot(state_dict)?;
    write(&mut File::create(path)?, &tensors, Some(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_used_where_the_data_is_aligned() {
        // The data section starts 8-byte aligned; "z" starts at 14, which isn't a multiple of 8.
        let header = concat!(
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"#,
            r#""y":{"dtype":"I16","shape":[3],"data_offsets":[8,14]},"#,
            r#""z":{"dtype":"F64","shape":[1],"data_offsets":[14,22]},"#,
            r#""b":{"dtype":"BOOL","shape":[2],"data_offsets":[22,24]},"#,
            r#""e":{"dtype":"F32","shape":[0],"data_offsets":[24,24]}}"#,
        );
        let mut raw = header.to_string();
        while !raw.len().is_multiple_of(8) {
            raw.push(' ');
        }
        let mut bytes = (raw.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(raw.as_bytes());
        for x in [1.5f32, -2.0] {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        for y in [7i16, -300, 4] {
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        bytes.extend_from_slice(&0.25f64.to_le_bytes());
        bytes.extend_from_slice(&[1, 0]);
        let path = std::env::temp_dir().join(format!("homura-mmap-{}", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        // SAFETY: only this test uses the file, and it is removed after the tensors are dropped.
        let loaded = unsafe { read_mmap(&mut File::open(&path).unwrap()) }.unwrap();
        let little = cfg!(target_endian = "little");
        let expected = [
            ("x", little),
            ("y", little),
            ("z", false),
            ("b", true),
            ("e", false),
        ];
        for ((name, t), (expected_name, mapped)) in loaded.iter().zip(expected) {
            assert_eq!(name, expected_name);
            assert_eq!(t.data().storage.is_mapped(), mapped, "{name}");
        }
        assert_eq!(loaded[0].1.to_vec::<f32>(), [1.5, -2.0]);
        assert_eq!(loaded[1].1.to_vec::<i16>(), [7, -300, 4]);
        assert_eq!(loaded[2].1.to_vec::<f64>(), [0.25]);
        assert_eq!(loaded[3].1.to_vec::<bool>(), [true, false]);
        assert_eq!(loaded[4].1.shape(), [0]);
        drop(loaded);
        std::fs::remove_file(&path).unwrap();
    }
}
