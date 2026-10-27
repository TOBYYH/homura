//! Reading and writing tensor files.
//!
//! [`homura::load`](crate::load) and [`homura::save`](crate::save) read and write
//! [`StateDict`](crate::nn::StateDict)s and cover most uses; this module has the rest:
//!
//! - safetensors `__metadata__`: [`safetensors_metadata`] and
//!   [`save_safetensors_with_metadata`];
//! - sharded checkpoints in the Hugging Face layout: [`save_sharded`] with [`ShardOptions`]
//!   (`load` reads them through their index; [`load_index`] reads shards kept apart from it);
//! - zero-copy loading: [`load_mmap`], which maps the files into memory. It is `unsafe`: the
//!   files must not change while tensors use them. `load` never maps files;
//! - whole PyTorch checkpoints, not only state dicts: [`load_value`] and
//!   [`load_value_from_bytes`] read the Python objects of `.pt` files as a [`Value`], and
//!   [`save_value`] and [`save_value_to_bytes`] write them, with
//!   [`Dict`] and its [`Key`]s, and conversions from and to [`StateDict`](crate::nn::StateDict)
//!   that keep the module versions;
//! - lists of tensors without a state dict: [`load_safetensors`],
//!   [`load_safetensors_from_bytes`], [`save_safetensors`] and [`save_safetensors_to_bytes`],
//!   also at the crate root.
//!
//! Loading treats files as untrusted input: malformed files give an [`Error`](crate::Error),
//! never a panic, and every length is checked against the file size before memory is allocated.
//! PyTorch's `.pt` files are read with a restricted unpickler that never runs code: it accepts
//! only the objects that [`Value`] describes, and limits how far objects referenced several times
//! can expand, so a small file can't make it allocate without bound. Their tensors are checked
//! before they are made: the storage records must have the declared sizes and the views must lie
//! inside them. Their zip archives are checked more strictly than PyTorch checks them: local
//! headers and data descriptors must agree with the central directory, entries can't overlap,
//! and nothing can follow the end of central directory record. The archives that `torch.save`
//! and Python's `zipfile` write pass.
//!
//! | PyTorch / Hugging Face | homura |
//! | --- | --- |
//! | `safetensors.torch.load_file(path)` | `homura::load(path)` |
//! | `safetensors.torch.save_file(sd, path)` | `homura::save(&sd, path)` |
//! | `save_file(sd, path, metadata)` | `io::save_safetensors_with_metadata(&sd, path, &metadata)` |
//! | `safe_open(path, "pt").metadata()` | `io::safetensors_metadata(path)` |
//! | `huggingface_hub.save_torch_state_dict(sd, dir, max_shard_size=n)` | `io::save_sharded(&sd, dir, ShardOptions::default().max_shard_size(n))` |
//! | loading `model.safetensors.index.json` | `homura::load("model.safetensors.index.json")` |
//! | `torch.load(path, mmap=True)` | `unsafe { io::load_mmap(path) }` |
//! | `torch.load(path, map_location="cpu", weights_only=True)` (a state dict) | `homura::load(path)` |
//! | `torch.load(path, map_location="cpu", weights_only=True)` (a checkpoint) | `io::load_value(path)` |
//! | `torch.save(model.state_dict(), "model.pt")` | `homura::save(&model.state_dict(), "model.pt")` |
//! | `torch.save(checkpoint, path)` | `io::save_value(&checkpoint, path)` |
//! | loading `pytorch_model.bin.index.json` | `homura::load("pytorch_model.bin.index.json")` |

mod file;
mod json;
mod pickle;
mod pt;
mod safetensors;
mod shard;
mod value;
mod zip;

pub use file::{
    load, load_index, load_index_mmap, load_mmap, load_value, load_value_from_bytes, save,
    save_value, save_value_to_bytes,
};
pub use safetensors::{
    load_safetensors, load_safetensors_from_bytes, safetensors_metadata, save_safetensors,
    save_safetensors_to_bytes, save_safetensors_with_metadata,
};
pub use shard::{ShardOptions, save_sharded};
pub use value::{Dict, Key, Value};

/// The files of `tests/golden/` for unit tests.
#[cfg(test)]
mod fixtures {
    use std::path::Path;

    /// Whether the files are here: a packaged crate has no `tests/` directory, so its unit tests
    /// skip the cases that read them.
    pub(super) fn available() -> bool {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").is_dir()
    }

    /// A file of `tests/golden/m3_pt/`.
    pub(super) fn pt(name: &str) -> Vec<u8> {
        let path = format!("{}/tests/golden/m3_pt/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }
}
