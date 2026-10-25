//! [`load`] and [`save`]: state dicts in files whose format is told from their content or name.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::{safetensors, shard};
use crate::nn::StateDict;
use crate::{Error, Result, Tensor};

/// The format of a file, from its first bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Safetensors,
    /// A zip archive: PyTorch's `.pt` format.
    Zip,
    /// A JSON index of shards.
    Index,
}

/// Tells the format of `file` from its first eight bytes. A safetensors file starts with its
/// header length, whose upper bytes are zero for any header size the reader accepts; a JSON text
/// starts with `{` and never contains a zero byte. The readers seek back to the start themselves.
fn detect(file: &mut File) -> Result<Format> {
    let mut start = Vec::with_capacity(8);
    file.by_ref().take(8).read_to_end(&mut start)?;
    Ok(if start.starts_with(b"PK\x03\x04") {
        Format::Zip
    } else if start.first() == Some(&b'{') && !start.contains(&0) {
        Format::Index
    } else {
        Format::Safetensors
    })
}

fn pt_not_supported(path: &Path) -> Error {
    Error::Format(format!(
        "{} is a PyTorch .pt file, which can't be loaded yet",
        path.display()
    ))
}

/// Loads the tensors of one shard of a sharded checkpoint.
pub(super) fn load_shard(path: &Path) -> Result<Vec<(String, Tensor)>> {
    let mut file = File::open(path)?;
    match detect(&mut file)? {
        Format::Safetensors => safetensors::read(&mut file),
        Format::Zip => Err(pt_not_supported(path)),
        Format::Index => Err(Error::Format(format!(
            "{} is an index, not a shard",
            path.display()
        ))),
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
/// - PyTorch's zip-based `.pt` format (including `pytorch_model.bin`) can't be loaded yet.
///
/// The state dict has no [versions](StateDict::versions): safetensors files can't hold them.
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
    let path = path.as_ref();
    let mut file = File::open(path)?;
    match detect(&mut file)? {
        Format::Safetensors => Ok(safetensors::read(&mut file)?.into_iter().collect()),
        Format::Zip => Err(pt_not_supported(path)),
        Format::Index => shard::read_index(path, file),
    }
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
