//! [`load`], [`load_mmap`] and [`save`]: state dicts in files whose format is told from their
//! content or name.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use super::{Value, pt, safetensors, shard};
use crate::nn::StateDict;
use crate::{Error, Result, Tensor};

/// The format of a file, from its first bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Safetensors,
    /// A zip archive: PyTorch's `.pt` format.
    Zip,
    /// PyTorch's format before 1.6: a sequence of pickles.
    Legacy,
    /// A JSON index of shards.
    Index,
}

/// How a file in PyTorch's format before 1.6 starts: pickle protocol 2 and the `LONG1` opcode
/// of its magic number, `0x1950a86a20f9469cfc6c`.
const LEGACY_MAGIC: [u8; 8] = [0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46];

/// Tells the format of a file from its first eight bytes (or all of them, if it is shorter). A
/// safetensors file starts with its header length, whose upper bytes are zero for any header
/// size the reader accepts (which also rules out [`LEGACY_MAGIC`]); a JSON text starts with `{`
/// and never contains a zero byte.
fn detect_bytes(start: &[u8]) -> Format {
    let start = &start[..start.len().min(8)];
    if start.starts_with(b"PK\x03\x04") {
        Format::Zip
    } else if start == LEGACY_MAGIC {
        Format::Legacy
    } else if start.first() == Some(&b'{') && !start.contains(&0) {
        Format::Index
    } else {
        Format::Safetensors
    }
}

/// Tells the format of `file` from its first bytes, as [`detect_bytes`] does. The readers seek
/// back to the start themselves.
fn detect(file: &mut File) -> Result<Format> {
    let mut start = Vec::with_capacity(8);
    file.by_ref().take(8).read_to_end(&mut start)?;
    Ok(detect_bytes(&start))
}

/// How the tensor data gets from a file into the tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Access {
    /// Read into memory of their own.
    Read,
    /// Mapped, as [`load_mmap`] describes.
    Map,
}

fn legacy_not_supported(path: &Path) -> Error {
    Error::Format(format!(
        "{} is in the format of PyTorch before 1.6, which is not supported: load it with \
         torch.load and save it again with torch.save in a newer PyTorch",
        path.display()
    ))
}

/// Reads the state dict of a `.pt` file.
///
/// # Safety
///
/// With [`Access::Map`], as for [`load_mmap`].
unsafe fn read_pt(file: &mut File, access: Access) -> Result<StateDict> {
    let value = match access {
        Access::Read => pt::read_value(file)?,
        // SAFETY: the caller upholds the contract of `load_mmap`.
        Access::Map => unsafe { pt::read_value_mmap(file) }?,
    };
    pt::to_state_dict(&value)
}

/// Loads the tensors of one shard of a sharded checkpoint.
///
/// # Safety
///
/// With [`Access::Map`], as for [`load_mmap`].
pub(super) unsafe fn load_shard(path: &Path, access: Access) -> Result<Vec<(String, Tensor)>> {
    let mut file = File::open(path)?;
    match detect(&mut file)? {
        // SAFETY: the caller upholds the contract of `load_mmap` for `Access::Map`.
        Format::Safetensors => unsafe { read_safetensors(&mut file, access) },
        // SAFETY: as above.
        Format::Zip => Ok(unsafe { read_pt(&mut file, access) }?.into_iter().collect()),
        Format::Legacy => Err(legacy_not_supported(path)),
        Format::Index => Err(Error::Format(format!(
            "{} is an index, not a shard",
            path.display()
        ))),
    }
}

/// # Safety
///
/// With [`Access::Map`], as for [`load_mmap`].
unsafe fn read_safetensors(file: &mut File, access: Access) -> Result<Vec<(String, Tensor)>> {
    match access {
        Access::Read => safetensors::read(file),
        // SAFETY: the caller upholds the contract of `load_mmap`.
        Access::Map => unsafe { safetensors::read_mmap(file) },
    }
}

/// # Safety
///
/// With [`Access::Map`], as for [`load_mmap`].
unsafe fn load_with(path: &Path, access: Access) -> Result<StateDict> {
    let mut file = File::open(path)?;
    match detect(&mut file)? {
        // SAFETY: the caller upholds the contract of `load_mmap` for `Access::Map`.
        Format::Safetensors => Ok(unsafe { read_safetensors(&mut file, access) }?
            .into_iter()
            .collect()),
        // SAFETY: as above.
        Format::Zip => unsafe { read_pt(&mut file, access) },
        Format::Legacy => Err(legacy_not_supported(path)),
        // SAFETY: as above.
        Format::Index => unsafe {
            shard::read_index(file, path.parent().unwrap_or(Path::new("")), access)
        },
    }
}

/// # Safety
///
/// With [`Access::Map`], as for [`load_mmap`].
unsafe fn load_index_with(index: &Path, shard_dir: &Path, access: Access) -> Result<StateDict> {
    let mut file = File::open(index)?;
    match detect(&mut file)? {
        // SAFETY: the caller upholds the contract of `load_mmap` for `Access::Map`.
        Format::Index => unsafe { shard::read_index(file, shard_dir, access) },
        _ => Err(Error::Format(format!(
            "{} is not an index of shards (a JSON object)",
            index.display()
        ))),
    }
}

/// Loads a state dict from a file, onto the CPU.
///
/// The format is told from the content, not the name:
///
/// - a safetensors file, read as [`load_safetensors`](crate::load_safetensors) does: in header
///   order, straight into each tensor's storage;
/// - a sharded checkpoint's JSON index (Hugging Face's `model.safetensors.index.json` or
///   `pytorch_model.bin.index.json`): the shards that its `weight_map` lists, relative to the
///   index's directory, in `weight_map` order. Every listed key must be in its shard and every
///   tensor of a shard must be listed; shard names can't leave the directory;
/// - PyTorch's zip-based `.pt` format (`torch.save`, including `pytorch_model.bin`) holding a
///   dict or `OrderedDict` of tensors, as `torch.load(path, map_location="cpu",
///   weights_only=True)` reads it. For other objects, such as a checkpoint with an optimizer's
///   state, use [`io::load_value`](crate::io::load_value). The archive and its pickle are
///   checked as untrusted input, without running any code. Tensors that share storage in the
///   file share it after loading, as views with the file's offsets and strides; each storage
///   is read once, straight into its memory. BOOL bytes other than 0 and 1, which PyTorch
///   writes and reads as `True`, become 1. Tensors keep the `requires_grad` saved with them,
///   and a saved `Parameter` becomes a plain tensor. TorchScript archives (`torch.jit.save`)
///   and files in the format before PyTorch 1.6 are not supported.
///
/// A state dict from safetensors files has no [versions](StateDict::versions), which they can't
/// hold; one from a `.pt` file has the versions saved with it (`_metadata`).
///
/// # Errors
///
/// Returns an error for IO errors and malformed or unsupported files. No input makes it panic.
///
/// ```no_run
/// use homura::nn::Module;
///
/// let model = homura::nn::Linear::new(4, 2);
/// model.load_state_dict(&homura::load("model.safetensors")?)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[track_caller]
pub fn load(path: impl AsRef<Path>) -> Result<StateDict> {
    // SAFETY: reading has no preconditions; only `Access::Map` has.
    unsafe { load_with(path.as_ref(), Access::Read) }
}

/// Loads a state dict like [`load`], but maps the files into memory instead of reading them:
/// tensors use the pages of the file, which are read from disk when the data is first used,
/// and loading needs almost no memory of its own. Like PyTorch's `torch.load(path, mmap=True)`.
///
/// The mapping is private and copy-on-write: tensors can be written in place, and the writes
/// stay in this process and never reach the file. A file is unmapped when the last tensor that
/// uses it is dropped, which can be long after the state dict is.
///
/// Which tensors use the mapping is not part of the contract. Currently, these get memory of
/// their own and a copy of the data: a tensor whose data doesn't start at a multiple of its
/// element size in the file (safetensors doesn't align data, although Python's `safetensors`
/// and homura write 8-byte aligned files; `torch.save` aligns to 64 bytes), multi-byte data in
/// another byte order than the host's (safetensors files are little-endian, `.pt` files say
/// which), an empty tensor, and the BOOL data of `.pt` files, whose bytes [`load`] changes to 0
/// and 1. The BOOL data of safetensors files is checked as [`load`] does, which reads only the
/// pages of BOOL tensors. The CRCs of `.pt` records are not checked, which would read every
/// page; the structure of the archive is.
///
/// # Safety
///
/// Every file that is mapped (the safetensors or `.pt` file, or each shard that an index lists)
/// must not be truncated or modified, by this or any other process, while any tensor whose data
/// it holds is alive. That is not only the returned state dict but every tensor sharing its storage:
/// views and other handles to its tensors, parameters that loading with
/// [`LoadOptions::assign`](crate::nn::LoadOptions::assign) made use them, and optimizers holding
/// those parameters. Tensors that hold a copy of the data, such as parameters loaded without
/// assign, don't depend on the file. Modifying a mapped file is undefined behaviour, as for
/// [`memmap2::Mmap::map`](https://docs.rs/memmap2/0.9/memmap2/struct.Mmap.html#method.map);
/// truncating it can crash the process.
///
/// # Errors
///
/// As for [`load`].
///
/// ```no_run
/// use homura::nn::{LoadOptions, Module};
///
/// let model = homura::nn::Linear::new(4, 2);
/// // SAFETY: nothing modifies model.safetensors while the program runs.
/// let state_dict = unsafe { homura::io::load_mmap("model.safetensors")? };
/// model.load_state_dict_with(&state_dict, LoadOptions::default().assign(true))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[track_caller]
pub unsafe fn load_mmap(path: impl AsRef<Path>) -> Result<StateDict> {
    // SAFETY: the caller upholds the contract above.
    unsafe { load_with(path.as_ref(), Access::Map) }
}

/// Loads a sharded checkpoint like [`load`] loads its index, but with the shard names of the
/// index relative to `shard_dir` rather than to the index's directory: for an index kept apart
/// from its shards, such as a Hugging Face repository's `transformers_weights` index in a
/// subdirectory, whose shards transformers looks for at the repository's root.
///
/// # Errors
///
/// As for [`load`], and for a file that isn't an index.
///
/// ```no_run
/// let sd = homura::io::load_index("repo/sub/model.safetensors.index.json", "repo")?;
/// # let _ = sd;
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub fn load_index(index: impl AsRef<Path>, shard_dir: impl AsRef<Path>) -> Result<StateDict> {
    // SAFETY: reading has no preconditions; only `Access::Map` has.
    unsafe { load_index_with(index.as_ref(), shard_dir.as_ref(), Access::Read) }
}

/// Like [`load_index`], with the shards mapped into memory as [`load_mmap`] maps them.
///
/// # Safety
///
/// As for [`load_mmap`], for every shard.
///
/// # Errors
///
/// As for [`load_index`].
///
/// ```no_run
/// // SAFETY: nothing modifies the shards while the program runs.
/// let sd = unsafe { homura::io::load_index_mmap("repo/sub/model.safetensors.index.json", "repo")? };
/// # let _ = sd;
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub unsafe fn load_index_mmap(
    index: impl AsRef<Path>,
    shard_dir: impl AsRef<Path>,
) -> Result<StateDict> {
    // SAFETY: the caller upholds the contract above, which is `load_mmap`'s.
    unsafe { load_index_with(index.as_ref(), shard_dir.as_ref(), Access::Map) }
}

/// Loads the object that a PyTorch `.pt` file holds, onto the CPU: a state dict, or a
/// checkpoint of other values, such as `{"model": .., "optimizer": .., "epoch": 3}`.
///
/// It reads what `torch.load(path, map_location="cpu", weights_only=True)` reads: `None`,
/// `bool`, `int` (up to 64 bits), `float`, `str`, `list`, `tuple`, `dict` (with `int` and
/// `str` keys), `OrderedDict` (with the versions of a state dict's `_metadata`) and tensors,
/// as [`Value`]s. Tensors are read as [`load`] reads them. The file is checked as untrusted
/// input, without running any code; objects referenced several times are copied into each place,
/// as far as a budget that grows with the file's size allows (a tensor stays one tensor, with
/// the same handle in each place), and objects that contain themselves are an error.
///
/// # Errors
///
/// Returns an error for IO errors, malformed or unsupported files, and files in other formats
/// (safetensors files and indexes hold state dicts: read them with [`load`]). No input makes it
/// panic.
///
/// ```no_run
/// use homura::io::{Value, load_value};
///
/// let Value::Dict(checkpoint) = load_value("checkpoint.pt")? else {
///     panic!("not a checkpoint");
/// };
/// let model = checkpoint.get("model").expect("the checkpoint has a model");
/// let state_dict = homura::nn::StateDict::try_from(model)?;
/// # let _ = state_dict;
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub fn load_value(path: impl AsRef<Path>) -> Result<Value> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    match detect(&mut file)? {
        Format::Zip => pt::read_value(&mut file),
        Format::Legacy => Err(legacy_not_supported(path)),
        Format::Safetensors | Format::Index => Err(not_pt(&path.display().to_string())),
    }
}

/// Like [`load_value`], for the bytes of a `.pt` file held in memory.
///
/// # Errors
///
/// As for [`load_value`].
///
/// ```no_run
/// let bytes = std::fs::read("checkpoint.pt")?;
/// let value = homura::io::load_value_from_bytes(&bytes)?;
/// # let _ = value;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[track_caller]
pub fn load_value_from_bytes(bytes: &[u8]) -> Result<Value> {
    match detect_bytes(bytes) {
        Format::Zip => pt::read_value(&mut Cursor::new(bytes)),
        Format::Legacy => Err(legacy_not_supported(Path::new("the data"))),
        Format::Safetensors | Format::Index => Err(not_pt("the data")),
    }
}

fn not_pt(what: &str) -> Error {
    Error::Format(format!(
        "{what} is not a PyTorch .pt file (a zip archive); read safetensors files and indexes \
         with homura::load"
    ))
}

/// Saves a state dict to a file, in the format its extension names.
///
/// - `.safetensors`: as [`save_safetensors`](crate::save_safetensors) does, in state dict order,
///   without `__metadata__` (see
///   [`save_safetensors_with_metadata`](crate::io::save_safetensors_with_metadata)). The format
///   has no place for the state dict's [versions](StateDict::versions), and each key gets its own
///   copy of the data, even for tensors that share storage;
/// - `.pt`, `.pth` and `.bin` (PyTorch's format) can't be saved yet.
///
/// Extensions are compared ignoring ASCII case. For sharded checkpoints, see
/// [`save_sharded`](crate::io::save_sharded).
///
/// # Errors
///
/// Returns an error for other extensions, meta tensors (they have no data), the key
/// `__metadata__`, which safetensors reserves, and IO errors.
///
/// ```no_run
/// use homura::nn::Module;
///
/// let model = homura::nn::Linear::new(4, 2);
/// homura::save(&model.state_dict(), "model.safetensors")?;
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub fn save(state_dict: &StateDict, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "safetensors" => {
            let tensors = safetensors::snapshot(state_dict)?;
            safetensors::write(&mut File::create(path)?, &tensors, None)
        }
        "pt" | "pth" | "bin" => Err(Error::Format(format!(
            "can't save {}: PyTorch .pt files can't be saved yet",
            path.display()
        ))),
        _ => Err(Error::Format(format!(
            "can't tell the format to save {} in from its extension: use .safetensors",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{ShardOptions, save_sharded};

    #[test]
    fn only_load_mmap_maps_files() {
        let dir = std::env::temp_dir().join(format!("homura-load-mmap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state_dict: StateDict = [
            ("a", crate::ones([4])),
            ("b", crate::zeros_with([3], crate::DType::I64)),
            ("c", crate::ones([0])),
        ]
        .into_iter()
        .collect();
        save_sharded(
            &state_dict,
            &dir,
            ShardOptions::default().max_shard_size(16),
        )
        .unwrap();
        crate::save(&state_dict, dir.join("single.safetensors")).unwrap();
        let mapped = |sd: &StateDict| -> Vec<bool> {
            sd.iter()
                .map(|(_, t)| t.data().storage.is_mapped())
                .collect()
        };
        for path in ["model.safetensors.index.json", "single.safetensors"] {
            let path = dir.join(path);
            assert_eq!(mapped(&load(&path).unwrap()), [false; 3]);
            // SAFETY: only this test uses the files, and they are removed after the tensors are
            // dropped.
            let sd = unsafe { load_mmap(&path) }.unwrap();
            let little = cfg!(target_endian = "little");
            assert_eq!(mapped(&sd), [little, little, false], "{}", path.display());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
