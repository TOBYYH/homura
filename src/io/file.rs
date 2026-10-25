//! [`load`], [`load_mmap`] and [`save`]: state dicts in files whose format is told from their
//! content or name.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::{pt, safetensors, shard};
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

/// Tells the format of `file` from its first eight bytes. A safetensors file starts with its
/// header length, whose upper bytes are zero for any header size the reader accepts (which
/// also rules out [`LEGACY_MAGIC`]); a JSON text starts with `{` and never contains a zero byte.
/// The readers seek back to the start themselves.
fn detect(file: &mut File) -> Result<Format> {
    let mut start = Vec::with_capacity(8);
    file.by_ref().take(8).read_to_end(&mut start)?;
    Ok(if start.starts_with(b"PK\x03\x04") {
        Format::Zip
    } else if start == LEGACY_MAGIC {
        Format::Legacy
    } else if start.first() == Some(&b'{') && !start.contains(&0) {
        Format::Index
    } else {
        Format::Safetensors
    })
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
        Format::Zip => Ok(pt::read_state_dict(&mut file)?.into_iter().collect()),
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
        Format::Zip => pt::read_state_dict(&mut file),
        Format::Legacy => Err(legacy_not_supported(path)),
        // SAFETY: as above.
        Format::Index => unsafe { shard::read_index(path, file, access) },
    }
}

/// Loads a state dict from a file, onto the CPU.
///
/// The format is told from the content, not the name:
///
/// - a safetensors file, read as [`load_safetensors`](crate::load_safetensors) does: in header
///   order, straight into each tensor's storage;
/// - a sharded checkpoint's JSON index (Hugging Face's `model.safetensors.index.json`): the
///   shards that its `weight_map` lists, relative to the index's directory, in `weight_map`
///   order. Every listed key must be in its shard and every tensor of a shard must be listed;
///   shard names can't leave the directory;
/// - PyTorch's zip-based `.pt` format (including `pytorch_model.bin`): the archive and its
///   pickle are checked as untrusted input, without running any code, but tensors can't be
///   loaded from them yet. TorchScript archives (`torch.jit.save`) and files in the format before
///   PyTorch 1.6 are not supported.
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
/// Which tensors use the mapping is not part of the contract. Currently, a tensor whose data
/// doesn't start at a multiple of its element size in the file (safetensors doesn't align data,
/// although Python's `safetensors` and homura write 8-byte aligned files), a multi-byte tensor
/// on a big-endian host and an empty tensor get memory of their own and a copy of the data.
/// BOOL data is checked as [`load`] does, which reads only the pages of BOOL tensors.
///
/// Tensors can't be loaded from PyTorch's `.pt` format yet, mapped or not.
///
/// # Safety
///
/// Every file that is mapped (the safetensors file, or each shard that an index lists) must not
/// be truncated or modified, by this or any other process, while any tensor whose data it holds
/// is alive. That is not only the returned state dict but every tensor sharing its storage:
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
