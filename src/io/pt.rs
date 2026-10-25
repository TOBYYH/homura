//! PyTorch's `.pt` format, as `torch.save` writes it since PyTorch 1.6: a zip archive with a
//! pickle (`<archive>/data.pkl`) whose persistent ids name storage records (`<archive>/data/<key>`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use super::pickle::{Global, Id, Node, Pickle, Tensors, over_budget};
use super::value::Value;
use super::zip::{self, Archive, Entry};
use crate::nn::StateDict;
use crate::shape::checked_num_elements;
use crate::storage::{FileMap, Storage};
use crate::tensor::{Layout, TensorData};
use crate::{DType, Device, Error, Result, Shape, Strides, Tensor};

/// The versions of the format that PyTorch 2.14 reads.
const VERSIONS: std::ops::RangeInclusive<u64> = 1..=10;

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid PyTorch file: {msg}"))
}

fn unsupported(msg: &str) -> Error {
    Error::Format(format!("unsupported PyTorch file: {msg}"))
}

/// Parses the `version` record as PyTorch does: a decimal number, with optional whitespace
/// around it.
fn parse_version(bytes: &[u8]) -> Result<u64> {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim_ascii().parse().ok())
        .ok_or_else(|| bad("the version record is not a number"))
}

/// A checked archive: its entries, the directory they are in and what its records say.
struct PtArchive {
    archive: Archive,
    /// The archive's directory with a trailing `/`.
    prefix: String,
    /// Whether the storages are big-endian (the `byteorder` record).
    big_endian: bool,
    pickle: Vec<u8>,
}

impl PtArchive {
    /// The record of the storage `key`.
    fn storage(&self, key: &str) -> Option<&Entry> {
        self.archive.get(&format!("{}data/{key}", self.prefix))
    }
}

/// Checks the structure of the archive in `r` and reads its pickle.
///
/// The archive's directory is the one holding `data.pkl`, and every entry must be in it.
/// `version` must be one that PyTorch reads and `byteorder`, if there is one, `little` or `big`;
/// the other records are ignored. TorchScript archives (`torch.jit.save`) are rejected.
fn read_archive<R: Read + Seek>(r: &mut R) -> Result<PtArchive> {
    let archive = Archive::read(r)?;
    let mut pickles = archive.entries().iter().filter_map(|e| {
        let dir = e.name.strip_suffix("/data.pkl")?;
        (!dir.is_empty() && !dir.contains('/')).then_some(dir)
    });
    let dir = pickles
        .next()
        .ok_or_else(|| bad("there is no data.pkl, so this is not a file that torch.save wrote"))?;
    if pickles.next().is_some() {
        return Err(bad("there are several data.pkl records"));
    }
    let prefix = format!("{dir}/");
    if let Some(e) = archive
        .entries()
        .iter()
        .find(|e| !e.name.starts_with(&prefix))
    {
        return Err(bad(&format!(
            "{:?} is outside the archive's directory {dir:?}",
            e.name
        )));
    }
    let record = |name: &str| archive.get(&format!("{prefix}{name}"));
    if record("constants.pkl").is_some()
        || archive
            .entries()
            .iter()
            .any(|e| e.name.starts_with(&format!("{prefix}code/")))
    {
        return Err(unsupported(
            "this is a TorchScript archive (torch.jit.save); save a state dict with torch.save \
             instead",
        ));
    }
    let version = record("version").ok_or_else(|| bad("there is no version record"))?;
    let version = parse_version(&zip::read_entry(r, version)?)?;
    if !VERSIONS.contains(&version) {
        return Err(unsupported(&format!(
            "version {version} (versions {} to {} are supported)",
            VERSIONS.start(),
            VERSIONS.end()
        )));
    }
    let big_endian = match record("byteorder") {
        None => false,
        Some(byteorder) => match &zip::read_entry(r, byteorder)?[..] {
            b"little" => false,
            b"big" => true,
            _ => return Err(bad("the byteorder record is neither little nor big")),
        },
    };
    let pickle = zip::read_entry(r, record("data.pkl").unwrap())?;
    Ok(PtArchive {
        archive,
        prefix,
        big_endian,
        pickle,
    })
}

/// Makes the bytes of a storage hold the values of the file on this host: swaps the bytes of
/// each element if the file's byte order is not the host's, and makes every nonzero BOOL byte
/// 1 (PyTorch writes other bytes too, which are `True` there; see `Storage::cpu_bytes_mut`).
fn fix_bytes(bytes: &mut [u8], dtype: DType, big_endian: bool) {
    let size = dtype.size_in_bytes();
    if size > 1 && big_endian != cfg!(target_endian = "big") {
        for element in bytes.chunks_mut(size) {
            element.reverse();
        }
    }
    if dtype == DType::Bool {
        for b in bytes {
            *b = (*b != 0) as u8;
        }
    }
}

/// A storage that a persistent id declared, with its declared type and size.
struct Declared {
    dtype: DType,
    numel: u64,
    storage: Arc<Storage>,
}

/// Rebuilds the tensors of a `.pt` file from its storage records, read from `r` or, with
/// `map`, mapped.
///
/// Each storage is read once, when a tensor first uses it, and shared by every tensor that uses
/// it. A call node is rebuilt once too: a tensor referenced several times is the same handle,
/// as it is the same object in Python.
struct Storages<'a, R> {
    r: &'a mut R,
    archive: &'a PtArchive,
    /// The mapping of the whole file, for `load_mmap`.
    map: Option<Arc<FileMap>>,
    storages: HashMap<String, Declared>,
    tensors: HashMap<Id, Tensor>,
    /// What is left of the budget for the sizes and strides read, which shared references can
    /// make many calls use.
    dims_left: u64,
}

/// The items of a tuple node.
fn tuple<'p>(pickle: &'p Pickle, id: Id, what: &str) -> Result<&'p [Id]> {
    match pickle.node(id) {
        Node::Tuple(items) => Ok(items),
        other => Err(bad(&format!(
            "{what} is a {}, not a tuple",
            other.type_name()
        ))),
    }
}

fn int(pickle: &Pickle, id: Id, what: &str) -> Result<i64> {
    match pickle.node(id) {
        Node::Int(i) => Ok(*i),
        other => Err(bad(&format!(
            "{what} is a {}, not an int",
            other.type_name()
        ))),
    }
}

/// A nonnegative int that fits `usize`.
fn index(pickle: &Pickle, id: Id, what: &str) -> Result<usize> {
    let i = int(pickle, id, what)?;
    usize::try_from(i).map_err(|_| bad(&format!("{what} is {i}")))
}

fn string<'p>(pickle: &'p Pickle, id: Id, what: &str) -> Result<&'p str> {
    match pickle.node(id) {
        Node::Str(s) => Ok(s),
        other => Err(bad(&format!(
            "{what} is a {}, not a str",
            other.type_name()
        ))),
    }
}

fn boolean(pickle: &Pickle, id: Id, what: &str) -> Result<bool> {
    match pickle.node(id) {
        Node::Bool(b) => Ok(*b),
        other => Err(bad(&format!(
            "{what} is a {}, not a bool",
            other.type_name()
        ))),
    }
}

/// Checks that the `backward_hooks` argument is an empty `OrderedDict`, as `torch.save` writes
/// it: hooks are never saved.
fn no_hooks(pickle: &Pickle, id: Id, func: &str) -> Result<()> {
    match pickle.node(id) {
        Node::Dict(d) if d.ordered && d.items.is_empty() => Ok(()),
        other => Err(unsupported(&format!(
            "the backward hooks of {func} are a {}; only an empty OrderedDict is supported",
            other.type_name()
        ))),
    }
}

impl<R: Read + Seek> Storages<'_, R> {
    /// The sizes or strides of a tensor: a tuple of nonnegative ints.
    fn dims(&mut self, pickle: &Pickle, id: Id, what: &str) -> Result<Vec<usize>> {
        let items = tuple(pickle, id, what)?;
        self.dims_left = self
            .dims_left
            .checked_sub(items.len() as u64)
            .ok_or_else(over_budget)?;
        items
            .iter()
            .map(|&item| match pickle.node(item) {
                Node::Int(i) => usize::try_from(*i)
                    .map_err(|_| bad(&format!("an element of the {what} is {i}"))),
                other => Err(bad(&format!(
                    "an element of the {what} is a {}, not an int",
                    other.type_name()
                ))),
            })
            .collect()
    }

    /// The storage that the persistent id node `id` declares:
    /// `('storage', storage_type, key, location, numel)`. The location is ignored: everything is
    /// loaded onto the CPU.
    fn storage(&mut self, pickle: &Pickle, id: Id) -> Result<(DType, Arc<Storage>)> {
        let Node::Persistent(pid) = pickle.node(id) else {
            return Err(bad(&format!(
                "the storage of a tensor is a {}, not a persistent id",
                pickle.node(id).type_name()
            )));
        };
        let items = tuple(pickle, *pid, "a persistent id")?;
        let [kind, storage_type, key, location, numel] = *items else {
            return Err(bad(&format!(
                "a persistent id has {} items instead of 5",
                items.len()
            )));
        };
        if string(pickle, kind, "the kind of a persistent id")? != "storage" {
            return Err(unsupported("a persistent id that is not a storage"));
        }
        let dtype = match pickle.node(storage_type) {
            Node::Global(Global::Storage(dtype)) => *dtype,
            other => {
                return Err(bad(&format!(
                    "the storage type of a persistent id is a {}",
                    other.type_name()
                )));
            }
        };
        let key = string(pickle, key, "a storage key")?;
        string(pickle, location, "a storage location")?;
        let numel = int(pickle, numel, "a storage size")?;
        let numel =
            u64::try_from(numel).map_err(|_| bad(&format!("storage {key:?} has size {numel}")))?;
        if let Some(d) = self.storages.get(key) {
            if (d.dtype, d.numel) != (dtype, numel) {
                return Err(bad(&format!(
                    "storage {key:?} is declared as {numel} {} elements and as {} {} elements",
                    dtype.name(),
                    d.numel,
                    d.dtype.name()
                )));
            }
            return Ok((dtype, d.storage.clone()));
        }
        let entry = self
            .archive
            .storage(key)
            .ok_or_else(|| bad(&format!("there is no record for storage {key:?}")))?;
        let size = dtype.size_in_bytes();
        if numel.checked_mul(size as u64) != Some(entry.size) {
            return Err(bad(&format!(
                "the record of storage {key:?} has {} bytes, not the {numel} {} elements it is \
                 declared with",
                entry.size,
                dtype.name()
            )));
        }
        let storage = Arc::new(self.read(entry, dtype)?);
        self.storages.insert(
            key.to_string(),
            Declared {
                dtype,
                numel,
                storage: storage.clone(),
            },
        );
        Ok((dtype, storage))
    }

    /// Reads a storage record: copied into memory of its own or, with a mapping, mapped where
    /// the data needs no change and is aligned for its elements.
    fn read(&mut self, entry: &Entry, dtype: DType) -> Result<Storage> {
        zip::check_stored(entry)?;
        // The archive check put the record inside the file.
        let nbytes = usize::try_from(entry.size).map_err(|_| bad("a storage is too large"))?;
        let big_endian = self.archive.big_endian;
        let Some(map) = &self.map else {
            let mut storage = Storage::new(nbytes, Device::Cpu);
            let bytes = storage.cpu_bytes_mut();
            zip::read_entry_into(self.r, entry, bytes)?;
            fix_bytes(bytes, dtype, big_endian);
            return Ok(storage);
        };
        if nbytes == 0 {
            return Ok(Storage::new(0, Device::Cpu));
        }
        // The archive was checked against the file's length when it was read; the mapping must
        // still cover the record.
        let offset = usize::try_from(entry.start)
            .ok()
            .filter(|&o| o.checked_add(nbytes).is_some_and(|end| end <= map.len()))
            .ok_or_else(|| {
                Error::Io(std::io::Error::other(
                    "the file changed while it was being loaded",
                ))
            })?;
        // SAFETY: the archive check makes the entries' byte ranges disjoint, a record is the
        // storage of one key, and each key gets a single storage.
        let mut mapped = unsafe { Storage::mapped(map.clone(), offset, nbytes) };
        let size = dtype.size_in_bytes();
        let swap = size > 1 && big_endian != cfg!(target_endian = "big");
        if dtype != DType::Bool && !swap && offset.is_multiple_of(size) {
            return Ok(mapped);
        }
        let mut copy = Storage::new(nbytes, Device::Cpu);
        let bytes = copy.cpu_bytes_mut();
        bytes.copy_from_slice(mapped.cpu_bytes_mut());
        fix_bytes(bytes, dtype, big_endian);
        Ok(copy)
    }

    /// `_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad,
    /// backward_hooks[, metadata])`. Every argument is checked before the tensor is made.
    fn rebuild_tensor(&mut self, pickle: &Pickle, args: Id) -> Result<Tensor> {
        const FUNC: &str = "_rebuild_tensor_v2";
        let items = tuple(pickle, args, "the arguments of _rebuild_tensor_v2")?;
        if !(6..=7).contains(&items.len()) {
            return Err(bad(&format!(
                "{FUNC} is called with {} arguments instead of 6 or 7",
                items.len()
            )));
        }
        let (dtype, storage) = self.storage(pickle, items[0])?;
        let offset = index(pickle, items[1], "the storage offset of a tensor")?;
        let shape = self.dims(pickle, items[2], "size of a tensor")?;
        let strides = self.dims(pickle, items[3], "stride of a tensor")?;
        let requires_grad = boolean(pickle, items[4], "requires_grad of a tensor")?;
        no_hooks(pickle, items[5], FUNC)?;
        if let Some(&metadata) = items.get(6) {
            match pickle.node(metadata) {
                Node::Dict(d) if d.items.is_empty() => {}
                Node::Dict(_) => {
                    return Err(unsupported(
                        "tensor metadata (such as the conjugate and negative bits of complex \
                         tensors)",
                    ));
                }
                other => {
                    return Err(bad(&format!(
                        "the metadata of a tensor is a {}",
                        other.type_name()
                    )));
                }
            }
        }
        if shape.len() != strides.len() {
            return Err(bad(&format!(
                "a tensor has {} sizes and {} strides",
                shape.len(),
                strides.len()
            )));
        }
        if checked_num_elements(&shape).is_none() {
            return Err(bad(&format!("the size {shape:?} has too many elements")));
        }
        let layout = Layout {
            shape: Shape::new(&shape),
            strides: Strides::new(&strides),
            offset,
        };
        let end = layout
            .checked_end()
            .and_then(|end| end.checked_mul(dtype.size_in_bytes()));
        if end.is_none_or(|end| end > storage.nbytes()) {
            return Err(bad(&format!(
                "a tensor with size {shape:?}, stride {strides:?} and storage offset {offset} \
                 reaches outside its storage of {} bytes",
                storage.nbytes()
            )));
        }
        if requires_grad && !dtype.is_floating_point() {
            return Err(bad(&format!(
                "a tensor of dtype {} requires grad; only floating point tensors can",
                dtype.name()
            )));
        }
        let t = Tensor::from_data(TensorData {
            storage,
            layout,
            dtype,
        });
        if requires_grad {
            t.requires_grad_(true);
        }
        Ok(t)
    }

    /// `_rebuild_parameter(data, requires_grad, backward_hooks)`: a new tensor sharing the data
    /// of `data`, which must be a `_rebuild_tensor_v2` call, with its own `requires_grad`, as
    /// `torch.nn.Parameter(data, requires_grad)` makes. It is a plain tensor in homura.
    fn rebuild_parameter(&mut self, pickle: &Pickle, args: Id) -> Result<Tensor> {
        const FUNC: &str = "_rebuild_parameter";
        let items = tuple(pickle, args, "the arguments of _rebuild_parameter")?;
        let [data, requires_grad, hooks] = *items else {
            return Err(bad(&format!(
                "{FUNC} is called with {} arguments instead of 3",
                items.len()
            )));
        };
        let data = match pickle.node(data) {
            Node::Call {
                func: Global::RebuildTensorV2,
                args,
            } => self.tensor(pickle, data, Global::RebuildTensorV2, *args)?,
            other => {
                return Err(unsupported(&format!(
                    "the data of a parameter is a {}; only a tensor that _rebuild_tensor_v2 \
                     makes is supported",
                    match other {
                        Node::Call { .. } => "parameter",
                        other => other.type_name(),
                    }
                )));
            }
        };
        let requires_grad = boolean(pickle, requires_grad, "requires_grad of a parameter")?;
        no_hooks(pickle, hooks, FUNC)?;
        let data = data.data();
        if requires_grad && !data.dtype.is_floating_point() {
            return Err(bad(&format!(
                "a parameter of dtype {} requires grad; only floating point tensors can",
                data.dtype.name()
            )));
        }
        let t = Tensor::from_data(data);
        if requires_grad {
            t.requires_grad_(true);
        }
        Ok(t)
    }
}

impl<R: Read + Seek> Tensors for Storages<'_, R> {
    fn tensor(&mut self, pickle: &Pickle, call: Id, func: Global, args: Id) -> Result<Tensor> {
        if let Some(t) = self.tensors.get(&call) {
            return Ok(t.clone());
        }
        let t = match func {
            Global::RebuildTensorV2 => self.rebuild_tensor(pickle, args)?,
            Global::RebuildParameter => self.rebuild_parameter(pickle, args)?,
            other => unreachable!("only tensor rebuild functions are called, not {other:?}"),
        };
        self.tensors.insert(call, t.clone());
        Ok(t)
    }
}

/// Reads the object that the `.pt` file in `r` holds, with its tensors read from `r` or, with
/// `map` (a mapping of the file of `r`), mapped.
fn read_with<R: Read + Seek>(r: &mut R, map: Option<Arc<FileMap>>) -> Result<Value> {
    let archive = read_archive(r)?;
    let pickle = Pickle::parse(&archive.pickle)?;
    let mut storages = Storages {
        r,
        archive: &archive,
        map,
        storages: HashMap::new(),
        tensors: HashMap::new(),
        dims_left: pickle.value_budget(),
    };
    pickle.to_value(&mut storages)
}

/// Reads the object that the `.pt` file in `r` holds, copying the tensors' data into memory.
pub(super) fn read_value<R: Read + Seek>(r: &mut R) -> Result<Value> {
    read_with(r, None)
}

/// Reads the object that the `.pt` file `file` holds, mapping the tensors' data where it can
/// (see [`load_mmap`](super::load_mmap)).
///
/// # Safety
///
/// As for [`load_mmap`](super::load_mmap): `file` must not be truncated or modified while any
/// tensor that maps it is alive.
pub(super) unsafe fn read_value_mmap(file: &mut File) -> Result<Value> {
    let len = file.seek(SeekFrom::End(0))?;
    // SAFETY: the caller keeps the file unchanged while the mapping lives.
    let map = unsafe { FileMap::new(file)? };
    if map.len() as u64 != len {
        return Err(Error::Io(std::io::Error::other(
            "the file changed while it was being loaded",
        )));
    }
    read_with(file, Some(map))
}

/// The state dict in `value`: a dict or `OrderedDict` of tensors.
pub(super) fn to_state_dict(value: &Value) -> Result<StateDict> {
    StateDict::try_from(value).map_err(|e| match e {
        Error::Format(msg) => Error::Format(format!(
            "{msg}; homura::io::load_value reads other objects from .pt files"
        )),
        e => e,
    })
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{self, pt};
    use super::super::json::{self, Json};
    use super::super::pickle::tests::{Placeholders, canonical};
    use super::super::zip::testing::{Options, Writer};
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;

    fn value(bytes: &[u8]) -> Result<Value> {
        read_value(&mut Cursor::new(bytes))
    }

    fn err(result: Result<impl std::fmt::Debug>, message: &str) {
        match result {
            Err(Error::Format(m)) if m.contains(message) => {}
            other => panic!("expected a format error with {message:?}, got {other:?}"),
        }
    }

    fn path(name: &str) -> PathBuf {
        PathBuf::from(format!(
            "{}/tests/golden/m3_pt/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    /// Reads a fixture mapped.
    fn mapped(name: &str) -> Result<Value> {
        // SAFETY: nothing modifies the fixtures.
        unsafe { read_value_mmap(&mut File::open(path(name))?) }
    }

    /// The tensors of `v`, depth first, as the generator lists them.
    fn tensors(v: &Value, out: &mut Vec<Tensor>) {
        match v {
            Value::Tensor(t) => out.push(t.clone()),
            Value::List(items) | Value::Tuple(items) => {
                items.iter().for_each(|x| tensors(x, out));
            }
            Value::Dict(d) => d.iter().for_each(|(_, x)| tensors(x, out)),
            _ => {}
        }
    }

    /// The dtype, shape and data of a tensor, as safetensors writes them.
    fn content(t: &Tensor) -> Vec<u8> {
        crate::save_safetensors_to_bytes([("t", t)]).unwrap()
    }

    /// Compares a value read from `<name>.pt` with `<name>.expected` and `<name>.safetensors`.
    fn check(name: &str, v: &Value) {
        let expected = String::from_utf8(pt(&format!("{name}.expected"))).unwrap();
        assert_eq!(canonical(v), expected.trim_end(), "{name}");
        let mut found = vec![];
        tensors(v, &mut found);
        let st = path(&format!("{name}.safetensors"));
        let reference = crate::load_safetensors(&st).unwrap();
        let metadata = crate::io::safetensors_metadata(&st).unwrap();
        assert_eq!(found.len(), reference.len(), "{name}");
        for (i, t) in found.iter().enumerate() {
            let what = format!("{name} tensor {i}");
            let (key, e) = reference.iter().find(|(k, _)| *k == i.to_string()).unwrap();
            assert_eq!(content(t), content(e), "{what}");
            let m = json::parse(&metadata[key]).unwrap();
            let number = |field: &str| m.get(field).and_then(Json::as_u64).unwrap() as usize;
            let stride: Vec<usize> = m
                .get("stride")
                .and_then(Json::as_array)
                .unwrap()
                .iter()
                .map(|s| s.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(t.storage_offset(), number("offset"), "{what}");
            assert_eq!(t.stride().as_slice(), &stride[..], "{what}");
            let data = t.data();
            assert_eq!(data.storage.nbytes(), number("storage_nbytes"), "{what}");
            assert_eq!(
                Some(&Json::Bool(t.requires_grad())),
                m.get("requires_grad"),
                "{what}"
            );
            // Identity and storage sharing, with the first tensor that has them.
            let object = number("object");
            let first_same = found.iter().position(|u| u.is_same(t)).unwrap();
            assert_eq!(first_same, object, "{what}");
            let storage = number("storage");
            let first_sharing = found
                .iter()
                .position(|u| u.is_same(t) || Arc::ptr_eq(&u.data().storage, &data.storage))
                .unwrap();
            assert_eq!(first_sharing, storage, "{what}");
        }
    }

    const TENSOR_FILES: [&str; 10] = [
        "dtypes",
        "big_endian",
        "layouts",
        "layouts_p4",
        "layouts_p5",
        "layouts_repacked",
        "bool_bytes",
        "optimizer",
        "state_dict",
        "checkpoint",
    ];

    #[test]
    fn values_match_the_reference() {
        if !fixtures::available() {
            return;
        }
        for name in [
            "values_p2",
            "values_p4",
            "values_p5",
            "repacked",
            "zip64",
            "crc0",
            "empty_state_dict",
            "build_once",
            "build_state_rebound",
            "build_state_extended",
            "build_empty_state",
            "build_replaced",
            "build_metadata_changed",
        ] {
            let expected = String::from_utf8(pt(&format!("{name}.expected"))).unwrap();
            let v = value(&pt(&format!("{name}.pt"))).unwrap();
            assert_eq!(canonical(&v), expected.trim_end(), "{name}");
        }
    }

    /// Copied from a file, from bytes and mapped, the tensors have the reference's values,
    /// layouts, identities and storage sharing.
    #[test]
    fn tensors_match_the_reference() {
        if !fixtures::available() {
            return;
        }
        for name in TENSOR_FILES {
            let file = format!("{name}.pt");
            check(
                name,
                &read_value(&mut File::open(path(&file)).unwrap()).unwrap(),
            );
            check(name, &value(&pt(&file)).unwrap());
            check(name, &mapped(&file).unwrap());
        }
    }

    /// Which storages a mapped read maps: those that need no change and are aligned.
    #[test]
    fn mapping_is_used_where_the_data_allows() {
        if !fixtures::available() {
            return;
        }
        let storages = |name: &str| -> Vec<(DType, usize, bool)> {
            let mut found = vec![];
            tensors(&mapped(&format!("{name}.pt")).unwrap(), &mut found);
            found
                .iter()
                .map(|t| {
                    let d = t.data();
                    (d.dtype, d.storage.nbytes(), d.storage.is_mapped())
                })
                .collect()
        };
        let little = cfg!(target_endian = "little");
        // torch.save aligns every record to 64 bytes: all but BOOL and empty storages map.
        for (dtype, nbytes, is_mapped) in storages("layouts").into_iter().chain(storages("dtypes"))
        {
            let expected =
                nbytes > 0 && dtype != DType::Bool && (little || dtype.size_in_bytes() == 1);
            assert_eq!(is_mapped, expected, "{dtype:?} {nbytes}");
        }
        // Big-endian data is swapped into memory of its own on little-endian hosts.
        for (dtype, _, is_mapped) in storages("big_endian") {
            let expected = dtype != DType::Bool && (!little || dtype.size_in_bytes() == 1);
            assert_eq!(is_mapped, expected, "{dtype:?}");
        }
        assert!(storages("bool_bytes").iter().all(|s| !s.2));
        // zipfile doesn't align: some records of the repacked file are copied.
        let repacked = storages("layouts_repacked");
        assert!(repacked.iter().any(|s| s.2) && repacked.iter().any(|s| !s.2 && s.1 > 0));
    }

    /// BOOL bytes other than 0 and 1 become 1 on every path.
    #[test]
    fn bool_bytes_are_normalized() {
        if !fixtures::available() {
            return;
        }
        for v in [
            value(&pt("bool_bytes.pt")).unwrap(),
            mapped("bool_bytes.pt").unwrap(),
        ] {
            let mut found = vec![];
            tensors(&v, &mut found);
            assert_eq!(found[0].to_vec::<bool>(), [true, true, false, true]);
            assert_eq!(found[1].to_vec::<bool>(), [true, false, true]);
            let d = found[0].data();
            // SAFETY: the storage has these 4 bytes, and nothing writes them meanwhile.
            let bytes = unsafe { std::slice::from_raw_parts(d.storage.cpu_ptr(), 4) };
            assert_eq!(bytes, [1, 1, 0, 1]);
        }
    }

    /// Copying checks each storage record's CRC; mapping doesn't read the records to check it.
    #[test]
    fn storage_crcs() {
        if !fixtures::available() {
            return;
        }
        let mut bytes = pt("layouts.pt");
        let archive = Archive::read(&mut Cursor::new(&bytes)).unwrap();
        let entry = archive.get("archive/data/0").unwrap();
        bytes[entry.start as usize] ^= 1;
        err(
            value(&bytes),
            "the CRC of \"archive/data/0\" doesn't match its data",
        );
        let dir = std::env::temp_dir().join(format!("homura-pt-crc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("changed.pt");
        std::fs::write(&file, &bytes).unwrap();
        // SAFETY: only this test uses the file, which is removed after the tensors are dropped.
        let v = unsafe { read_value_mmap(&mut File::open(&file).unwrap()) }.unwrap();
        let Value::Dict(d) = &v else { panic!() };
        let Some(Value::Tensor(base)) = d.get("base") else {
            panic!()
        };
        assert_ne!(base.to_vec::<f32>()[0], 0.0);
        drop(v);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A record that the mapping doesn't cover (the file shrank after its archive was read) is an
    /// IO error, not a storage outside the mapping.
    #[test]
    fn records_outside_the_mapping() {
        if !fixtures::available() {
            return;
        }
        let bytes = pt("layouts.pt");
        let archive = read_archive(&mut Cursor::new(&bytes)).unwrap();
        let entry = archive.storage("0").unwrap().clone();
        let dir = std::env::temp_dir().join(format!("homura-pt-shrunk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("shrunk.pt");
        std::fs::write(&file, &bytes[..entry.start as usize + 1]).unwrap();
        // SAFETY: only this test uses the file, which is removed after the mapping is dropped.
        let map = unsafe { FileMap::new(&File::open(&file).unwrap()) }.unwrap();
        let mut r = Cursor::new(&bytes);
        let mut storages = Storages {
            r: &mut r,
            archive: &archive,
            map: Some(map),
            storages: HashMap::new(),
            tensors: HashMap::new(),
            dims_left: 0,
        };
        match storages.read(&entry, DType::F32) {
            Err(Error::Io(e)) => assert!(e.to_string().contains("the file changed"), "{e}"),
            other => panic!("{other:?}"),
        }
        drop(storages);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn state_dicts_keep_their_versions() {
        if !fixtures::available() {
            return;
        }
        let expected: crate::nn::ModuleVersions = [("", 1), ("0", 1), ("1", 2), ("2", 1)]
            .into_iter()
            .collect();
        let sd = to_state_dict(&value(&pt("state_dict.pt")).unwrap()).unwrap();
        assert_eq!(sd.len(), 9);
        assert_eq!(sd.versions(), Some(&expected));
        // Inside a checkpoint too.
        let Value::Dict(checkpoint) = value(&pt("checkpoint.pt")).unwrap() else {
            panic!()
        };
        let model = StateDict::try_from(checkpoint.get("model").unwrap()).unwrap();
        assert_eq!(model.versions(), Some(&expected));
        assert!(matches!(checkpoint.get("epoch"), Some(Value::Int(3))));
        err(
            to_state_dict(&value(&pt("values_p2.pt")).unwrap()),
            "homura::io::load_value reads other objects",
        );
    }

    #[test]
    fn expansion_and_cycles_are_errors() {
        if !fixtures::available() {
            return;
        }
        // 2**30 and 64**8 leaves from about a kilobyte of pickle: the budget stops them.
        err(
            value(&pt("deep.pt")),
            "expand to much more than the size of the file",
        );
        err(
            value(&pt("wide.pt")),
            "expand to much more than the size of the file",
        );
        err(value(&pt("cycle_list.pt")), "recursive pickle structures");
        err(value(&pt("cycle_dict.pt")), "recursive pickle structures");
        err(
            value(&pt("module.pt")),
            "the global torch.nn.modules.linear.Linear is not supported",
        );
    }

    /// A pickle of `{}`.
    const EMPTY: &[u8] = b"\x80\x02}q\x00.";

    /// A record of an archive: its name and data.
    type Record<'a> = (&'a str, &'a [u8]);

    fn archive(entries: &[Record]) -> Vec<u8> {
        let mut w = Writer::default();
        for (name, data) in entries {
            w.add(name, data, Options::default());
        }
        w.finish(false, b"")
    }

    #[test]
    fn archive_structure() {
        let ok = |entries: &[Record]| value(&archive(entries)).unwrap();
        ok(&[("a/data.pkl", EMPTY), ("a/version", b"3\n")]);
        // As PyTorch parses the version, from 1 to 10; byteorder is optional.
        for version in [&b"1"[..], b" 3\n", b"10\n"] {
            ok(&[("a/data.pkl", EMPTY), ("a/version", version)]);
        }
        ok(&[
            ("x/data.pkl", EMPTY),
            ("x/version", b"3"),
            ("x/byteorder", b"big"),
        ]);
        ok(&[
            ("x/data.pkl", EMPTY),
            ("x/version", b"3"),
            ("x/byteorder", b"little"),
        ]);
        let cases: [(&[Record], &str); 12] = [
            (&[("a/version", b"3")], "there is no data.pkl"),
            (
                &[("data.pkl", EMPTY), ("version", b"3")],
                "there is no data.pkl",
            ),
            (
                &[("a/b/data.pkl", EMPTY), ("a/b/version", b"3")],
                "there is no data.pkl",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("b/data.pkl", EMPTY),
                    ("a/version", b"3"),
                ],
                "several data.pkl",
            ),
            (
                &[("a/data.pkl", EMPTY), ("a/version", b"3"), ("b/x", b"")],
                "\"b/x\" is outside the archive's directory \"a\"",
            ),
            (&[("a/data.pkl", EMPTY)], "there is no version record"),
            (&[("a/data.pkl", EMPTY), ("a/version", b"0\n")], "version 0"),
            (
                &[("a/data.pkl", EMPTY), ("a/version", b"11\n")],
                "version 11",
            ),
            (
                &[("a/data.pkl", EMPTY), ("a/version", b"x\n")],
                "not a number",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("a/version", b"3"),
                    ("a/byteorder", b"middle"),
                ],
                "neither little nor big",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("a/version", b"3"),
                    ("a/constants.pkl", EMPTY),
                ],
                "TorchScript",
            ),
            (
                &[
                    ("a/data.pkl", EMPTY),
                    ("a/version", b"3"),
                    ("a/code/m.py", b""),
                ],
                "TorchScript",
            ),
        ];
        for (entries, message) in cases {
            err(value(&archive(entries)), message);
        }
    }

    #[test]
    fn torchscript_archives_are_rejected() {
        if !fixtures::available() {
            return;
        }
        err(
            value(&pt("torchscript.pt")),
            "TorchScript archive (torch.jit.save)",
        );
    }

    /// Writes pickles of tensors by hand, for the cases `torch.save` doesn't write.
    mod build {
        pub fn str(s: &str) -> Vec<u8> {
            let mut out = vec![b'X'];
            out.extend((s.len() as u32).to_le_bytes());
            out.extend(s.as_bytes());
            out
        }

        pub fn int(i: i64) -> Vec<u8> {
            let mut out = vec![0x8a, 8];
            out.extend(i.to_le_bytes());
            out
        }

        pub fn global(module: &str, name: &str) -> Vec<u8> {
            format!("c{module}\n{name}\n").into_bytes()
        }

        pub fn tuple(items: &[Vec<u8>]) -> Vec<u8> {
            let mut out = vec![b'('];
            items.iter().for_each(|i| out.extend(i));
            out.push(b't');
            out
        }

        pub fn ints(items: &[i64]) -> Vec<u8> {
            tuple(&items.iter().map(|&i| int(i)).collect::<Vec<_>>())
        }

        pub fn ordered_dict() -> Vec<u8> {
            let mut out = global("collections", "OrderedDict");
            out.extend(b")R");
            out
        }

        pub fn storage(class: &str, key: &str, numel: i64) -> Vec<u8> {
            let mut out = tuple(&[
                str("storage"),
                global("torch", class),
                str(key),
                str("cpu"),
                int(numel),
            ]);
            out.push(b'Q');
            out
        }

        /// `func(*args)`.
        pub fn call(func: &str, args: &[Vec<u8>]) -> Vec<u8> {
            let mut out = global("torch._utils", func);
            out.extend(tuple(args));
            out.push(b'R');
            out
        }

        pub fn bool(b: bool) -> Vec<u8> {
            vec![if b { 0x88 } else { 0x89 }]
        }

        /// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, OrderedDict())`.
        pub fn tensor(
            storage: Vec<u8>,
            offset: i64,
            size: &[i64],
            stride: &[i64],
            grad: bool,
        ) -> Vec<u8> {
            call(
                "_rebuild_tensor_v2",
                &[
                    storage,
                    int(offset),
                    ints(size),
                    ints(stride),
                    bool(grad),
                    ordered_dict(),
                ],
            )
        }

        /// A pickle of the list of `items`.
        pub fn list(items: &[Vec<u8>]) -> Vec<u8> {
            let mut out = b"\x80\x02](".to_vec();
            items.iter().for_each(|i| out.extend(i));
            out.extend(b"e.");
            out
        }
    }

    /// A `.pt` file with the pickle `pickle` and the storage records `records`.
    fn pt_file(pickle: &[u8], records: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = Writer::default();
        w.add("a/data.pkl", pickle, Options::default());
        w.add("a/version", b"3", Options::default());
        for (key, data) in records {
            w.add(&format!("a/data/{key}"), data, Options::default());
        }
        w.finish(false, b"")
    }

    #[test]
    fn hand_written_tensors() {
        use build::*;
        let four = [0u8, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]; // 1, 2, 3, 4
        let float = |key, numel| storage("FloatStorage", key, numel);
        // A view at an offset, and two declarations of one storage.
        let pickle = list(&[
            tensor(float("0", 4), 1, &[2], &[2], false),
            tensor(float("0", 4), 0, &[2, 2], &[1, 2], true),
        ]);
        let Value::List(items) = value(&pt_file(&pickle, &[("0", &four)])).unwrap() else {
            panic!()
        };
        let [Value::Tensor(a), Value::Tensor(b)] = &items[..] else {
            panic!()
        };
        assert_eq!(a.to_vec::<f32>(), [2.0, 4.0]);
        assert_eq!(b.to_vec::<f32>(), [1.0, 3.0, 2.0, 4.0]);
        assert!(Arc::ptr_eq(&a.data().storage, &b.data().storage));
        assert!(b.requires_grad() && !a.requires_grad());
        // A parameter that shares its data's storage but is a new handle.
        let pickle = list(&[call(
            "_rebuild_parameter",
            &[
                tensor(float("0", 4), 0, &[4], &[1], false),
                bool(true),
                ordered_dict(),
            ],
        )]);
        let Value::List(items) = value(&pt_file(&pickle, &[("0", &four)])).unwrap() else {
            panic!()
        };
        let [Value::Tensor(p)] = &items[..] else {
            panic!()
        };
        assert!(p.requires_grad() && p.is_leaf());

        let cases: Vec<(Vec<u8>, &str)> = vec![
            (
                list(&[
                    tensor(float("0", 4), 0, &[1], &[1], false),
                    tensor(storage("IntStorage", "0", 4), 0, &[1], &[1], false),
                ]),
                "storage \"0\" is declared as 4 i32 elements and as 4 f32 elements",
            ),
            (
                list(&[
                    tensor(float("0", 4), 0, &[1], &[1], false),
                    tensor(float("0", 3), 0, &[1], &[1], false),
                ]),
                "declared as 3 f32 elements and as 4 f32",
            ),
            (
                list(&[tensor(storage("LongStorage", "0", 2), 0, &[2], &[1], true)]),
                "a tensor of dtype i64 requires grad",
            ),
            (
                list(&[call(
                    "_rebuild_parameter",
                    &[
                        tensor(storage("IntStorage", "0", 4), 0, &[4], &[1], false),
                        bool(true),
                        ordered_dict(),
                    ],
                )]),
                "a parameter of dtype i32 requires grad",
            ),
            (
                list(&[tensor(float("0", 4), 1, &[4], &[1], false)]),
                "reaches outside its storage of 16 bytes",
            ),
            (
                list(&[tensor(float("0", 4), 0, &[2, 2], &[2, 2], false)]),
                "reaches outside its storage",
            ),
            (
                list(&[tensor(
                    float("0", 4),
                    0,
                    &[2, i64::MAX],
                    &[i64::MAX, 1],
                    false,
                )]),
                "too many elements",
            ),
            (
                list(&[tensor(
                    float("0", 4),
                    0,
                    &[1 << 40, 1 << 30],
                    &[0, 0],
                    false,
                )]),
                "too many elements",
            ),
            (
                list(&[tensor(float("0", 4), 0, &[2], &[1, 1], false)]),
                "a tensor has 1 sizes and 2 strides",
            ),
            (
                list(&[tensor(float("0", 4), -1, &[1], &[1], false)]),
                "the storage offset of a tensor is -1",
            ),
            (
                list(&[tensor(float("0", 4), 0, &[-1], &[1], false)]),
                "an element of the size of a tensor is -1",
            ),
            (
                list(&[tensor(float("1", 4), 0, &[1], &[1], false)]),
                "there is no record for storage \"1\"",
            ),
            (
                list(&[tensor(float("0", 3), 0, &[1], &[1], false)]),
                "has 16 bytes, not the 3 f32 elements",
            ),
            (
                list(&[tensor(float("0", -1), 0, &[1], &[1], false)]),
                "storage \"0\" has size -1",
            ),
            (
                list(&[tensor(float("0", i64::MAX), 0, &[1], &[1], false)]),
                "not the 9223372036854775807 f32 elements",
            ),
            (
                list(&[call(
                    "_rebuild_tensor_v2",
                    &[float("0", 4), int(0), ints(&[1]), ints(&[1]), bool(false)],
                )]),
                "called with 5 arguments instead of 6 or 7",
            ),
            (
                list(&[call(
                    "_rebuild_tensor_v2",
                    &[
                        float("0", 4),
                        int(0),
                        ints(&[1]),
                        ints(&[1]),
                        bool(false),
                        b"}".to_vec(),
                    ],
                )]),
                "the backward hooks of _rebuild_tensor_v2 are a dict",
            ),
            (
                list(&[call(
                    "_rebuild_tensor_v2",
                    &[
                        float("0", 4),
                        int(0),
                        ints(&[1]),
                        ints(&[1]),
                        bool(false),
                        ordered_dict(),
                        [&b"}"[..], &str("conj"), b"\x88s"].concat(),
                    ],
                )]),
                "tensor metadata",
            ),
            (
                list(&[call(
                    "_rebuild_tensor_v2",
                    &[
                        float("0", 4),
                        int(0),
                        ints(&[1]),
                        ints(&[1]),
                        bool(false),
                        ordered_dict(),
                        b"}".to_vec(),
                    ],
                )]),
                "",
            ),
            (
                list(&[call(
                    "_rebuild_tensor_v2",
                    &[
                        int(0),
                        int(0),
                        ints(&[1]),
                        ints(&[1]),
                        bool(false),
                        ordered_dict(),
                    ],
                )]),
                "the storage of a tensor is a int, not a persistent id",
            ),
            (
                list(&[tensor(
                    [
                        tuple(&[
                            str("module"),
                            global("torch", "FloatStorage"),
                            str("0"),
                            str("cpu"),
                            int(4),
                        ]),
                        b"Q".to_vec(),
                    ]
                    .concat(),
                    0,
                    &[1],
                    &[1],
                    false,
                )]),
                "a persistent id that is not a storage",
            ),
            (
                list(&[tensor(
                    [
                        tuple(&[
                            str("storage"),
                            global("torch", "FloatStorage"),
                            str("0"),
                            int(0),
                            int(4),
                        ]),
                        b"Q".to_vec(),
                    ]
                    .concat(),
                    0,
                    &[1],
                    &[1],
                    false,
                )]),
                "a storage location is a int, not a str",
            ),
            (
                list(&[call(
                    "_rebuild_parameter",
                    &[
                        call(
                            "_rebuild_parameter",
                            &[
                                tensor(float("0", 4), 0, &[4], &[1], false),
                                bool(true),
                                ordered_dict(),
                            ],
                        ),
                        bool(true),
                        ordered_dict(),
                    ],
                )]),
                "the data of a parameter is a parameter",
            ),
            (
                list(&[call(
                    "_rebuild_parameter",
                    &[int(1), bool(true), ordered_dict()],
                )]),
                "the data of a parameter is a int",
            ),
        ];
        for (pickle, message) in cases {
            let result = value(&pt_file(&pickle, &[("0", &four)]));
            if message.is_empty() {
                result.unwrap();
            } else {
                err(result, message);
            }
        }
    }

    /// A tensor referenced many times is rebuilt once; sizes shared by many calls are charged.
    #[test]
    fn shared_tensors_and_sizes() {
        use build::*;
        let four = [0u8; 16];
        let float = storage("FloatStorage", "0", 4);
        // [t, t, ...]: one tensor through the memo.
        let mut pickle = b"\x80\x02](".to_vec();
        pickle.extend(tensor(float.clone(), 0, &[4], &[1], false));
        pickle.extend(b"q\x01");
        for _ in 0..1000 {
            pickle.extend(b"h\x01");
        }
        pickle.extend(b"e.");
        let Value::List(items) = value(&pt_file(&pickle, &[("0", &four)])).unwrap() else {
            panic!()
        };
        let Value::Tensor(first) = &items[0] else {
            panic!()
        };
        assert!(
            items
                .iter()
                .all(|t| matches!(t, Value::Tensor(t) if t.is_same(first)))
        );
        // Many calls sharing one size tuple of 20,000 ones.
        let dims = 20_000;
        let mut pickle = b"\x80\x02](".to_vec();
        pickle.extend(ints(&vec![1; dims]));
        pickle.extend(b"q\x02");
        let shared_call = |pickle: &mut Vec<u8>| {
            pickle.extend(global("torch._utils", "_rebuild_tensor_v2"));
            pickle.push(b'(');
            pickle.extend(&float);
            pickle.extend(int(0));
            pickle.extend(b"h\x02h\x02");
            pickle.extend(bool(false));
            pickle.extend(ordered_dict());
            pickle.extend(b"tR");
        };
        let mut few = pickle.clone();
        few.extend(b"0");
        for _ in 0..2 {
            shared_call(&mut few);
        }
        few.extend(b"e.");
        let Value::List(items) = value(&pt_file(&few, &[("0", &four)])).unwrap() else {
            panic!()
        };
        assert_eq!(items.len(), 2);
        pickle.extend(b"0");
        for _ in 0..200 {
            shared_call(&mut pickle);
        }
        pickle.extend(b"e.");
        err(
            value(&pt_file(&pickle, &[("0", &four)])),
            "expand to much more than the size of the file",
        );
    }

    /// A deterministic sequence of pseudo-random numbers.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as usize
        }
    }

    /// Changes a few bytes of `bytes` or truncates it.
    fn mutate(bytes: &[u8], rng: &mut Lcg) -> Vec<u8> {
        let mut out = bytes.to_vec();
        match rng.next() % 5 {
            0 => out.truncate(rng.next() % bytes.len()),
            1 => {
                let at = rng.next() % out.len();
                out[at] ^= 1 << (rng.next() % 8);
            }
            2 => {
                let at = rng.next() % out.len();
                out[at] = rng.next() as u8;
            }
            3 => {
                // A length or offset field: a small, a large or an extreme value.
                let at = rng.next() % out.len().saturating_sub(4).max(1);
                let value = [1u32, 0xff, 0xffff, 0xffff_ffff][rng.next() % 4];
                let end = (at + 4).min(out.len());
                out[at..end].copy_from_slice(&value.to_le_bytes()[..end - at]);
            }
            _ => {
                let at = rng.next() % out.len();
                out.insert(at, rng.next() as u8);
            }
        }
        out
    }

    /// Mutated archives and pickles give an error or a value, never a panic.
    #[test]
    fn mutated_input_never_panics() {
        if !fixtures::available() {
            return;
        }
        let rounds = if cfg!(debug_assertions) { 300 } else { 3000 };
        let mut rng = Lcg(3);
        for name in [
            "values_p2.pt",
            "values_p4.pt",
            "checkpoint.pt",
            "zip64.pt",
            "repacked.pt",
            "layouts.pt",
            "layouts_p4.pt",
            "big_endian.pt",
        ] {
            let bytes = pt(name);
            let pickle = read_archive(&mut Cursor::new(&bytes)).unwrap().pickle;
            for _ in 0..rounds {
                let _ = value(&mutate(&bytes, &mut rng));
                let mutated = mutate(&pickle, &mut rng);
                if let Ok(p) = Pickle::parse(&mutated) {
                    let _ = p.to_value(&mut Placeholders::default());
                }
            }
        }
    }

    /// Mutated pickles of tensors, rebuilt from the real records.
    #[test]
    fn mutated_tensor_pickles_never_panic() {
        if !fixtures::available() {
            return;
        }
        let rounds = if cfg!(debug_assertions) { 300 } else { 3000 };
        let mut rng = Lcg(5);
        for name in ["layouts.pt", "dtypes.pt", "optimizer.pt"] {
            let bytes = pt(name);
            let archive = read_archive(&mut Cursor::new(&bytes)).unwrap();
            for _ in 0..rounds {
                let mutated = mutate(&archive.pickle, &mut rng);
                let Ok(pickle) = Pickle::parse(&mutated) else {
                    continue;
                };
                let mut r = Cursor::new(&bytes);
                let mut storages = Storages {
                    r: &mut r,
                    archive: &archive,
                    map: None,
                    storages: HashMap::new(),
                    tensors: HashMap::new(),
                    dims_left: pickle.value_budget(),
                };
                let _ = pickle.to_value(&mut storages);
            }
        }
    }
}
