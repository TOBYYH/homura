//! Reading and writing tensor files.
//!
//! [`homura::load`](crate::load) and [`homura::save`](crate::save) read and write
//! [`StateDict`](crate::nn::StateDict)s and cover most uses; this module has the rest:
//!
//! - safetensors `__metadata__`: [`safetensors_metadata`] and
//!   [`save_safetensors_with_metadata`];
//! - sharded checkpoints in the Hugging Face layout: [`save_sharded`] with [`ShardOptions`]
//!   (`load` reads them through their index);
//! - lists of tensors without a state dict: [`load_safetensors`],
//!   [`load_safetensors_from_bytes`], [`save_safetensors`] and [`save_safetensors_to_bytes`],
//!   also at the crate root.
//!
//! Loading treats files as untrusted input: malformed files give an [`Error`](crate::Error),
//! never a panic, and every length is checked against the file size before memory is allocated.
//!
//! | PyTorch / Hugging Face | homura |
//! | --- | --- |
//! | `safetensors.torch.load_file(path)` | `homura::load(path)` |
//! | `safetensors.torch.save_file(sd, path)` | `homura::save(&sd, path)` |
//! | `save_file(sd, path, metadata)` | `io::save_safetensors_with_metadata(&sd, path, &metadata)` |
//! | `safe_open(path, "pt").metadata()` | `io::safetensors_metadata(path)` |
//! | `huggingface_hub.save_torch_state_dict(sd, dir, max_shard_size=n)` | `io::save_sharded(&sd, dir, ShardOptions::default().max_shard_size(n))` |
//! | loading `model.safetensors.index.json` | `homura::load("model.safetensors.index.json")` |

mod file;
mod json;
mod safetensors;
mod shard;

pub use file::{load, save};
pub use safetensors::{
    load_safetensors, load_safetensors_from_bytes, safetensors_metadata, save_safetensors,
    save_safetensors_to_bytes, save_safetensors_with_metadata,
};
pub use shard::{ShardOptions, save_sharded};
