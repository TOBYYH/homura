//! Sharded safetensors checkpoints in the Hugging Face layout: shard files and a JSON index with
//! a `weight_map` from tensor names to shard files.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::Path;

use super::file::{Access, load_shard};
use super::json::{self, Json};
use super::safetensors::{self, data_size, snapshot};
use crate::nn::StateDict;
use crate::{Error, Result, Tensor};

/// Upper limit on the size of an index file.
const MAX_INDEX_LEN: u64 = 100_000_000;

/// How [`save_sharded`] splits a state dict into files.
///
/// ```
/// use homura::io::ShardOptions;
///
/// let options = ShardOptions::default().max_shard_size(2_000_000_000).prefix("model");
/// # let _ = options;
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ShardOptions {
    max_shard_size: u64,
    prefix: String,
    metadata: Option<BTreeMap<String, String>>,
}

impl Default for ShardOptions {
    /// Shards of at most 5 GB (5×10⁹ bytes, Hugging Face's default), named after `model`, without
    /// `__metadata__`.
    #[track_caller]
    fn default() -> Self {
        ShardOptions {
            max_shard_size: 5_000_000_000,
            prefix: "model".into(),
            metadata: None,
        }
    }
}

impl ShardOptions {
    /// The largest size of the tensor data of a shard, in bytes. A tensor larger than this gets
    /// a shard of its own.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is 0.
    #[track_caller]
    pub fn max_shard_size(mut self, bytes: u64) -> Self {
        assert!(bytes > 0, "the maximum shard size must be positive");
        self.max_shard_size = bytes;
        self
    }

    /// The start of the file names: `<prefix>.safetensors` for a single shard, otherwise
    /// `<prefix>-00001-of-00003.safetensors` and so on, with the index
    /// `<prefix>.safetensors.index.json`.
    ///
    /// Names ending like a numbered shard (`-<digits>-of-<digits>`, as in `model-00001-of-00002`)
    /// are reserved for the shards of other prefixes, so the files of different prefixes never
    /// have the same name.
    ///
    /// # Panics
    ///
    /// Panics if `prefix` is empty, `.` or `..`, contains `/`, `\`, `:` or NUL, or ends like a
    /// numbered shard.
    #[track_caller]
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        assert!(
            is_file_name(&prefix),
            "the shard prefix must be a file name, not {prefix:?}"
        );
        assert!(
            numbered_base(&prefix).is_none(),
            "the shard prefix {prefix:?} ends like a numbered shard (`-<digits>-of-<digits>`), \
             which names the shards of another prefix"
        );
        self.prefix = prefix;
        self
    }

    /// The `__metadata__` written into every shard, as Hugging Face writes `{"format": "pt"}`.
    #[track_caller]
    pub fn metadata(mut self, metadata: BTreeMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Whether `name` names a file in a directory, without leaving it on any platform: not empty, `.`
/// or `..`, without `/`, `\\`, `:` (a Windows drive or prefix, such as `C:x`, replaces the
/// directory it is joined to) or NUL.
fn is_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

/// Saves a state dict as safetensors shards in the directory `dir`, which is created if needed,
/// as Hugging Face's `save_pretrained` does.
///
/// The entries are split greedily in state dict order: a new shard starts when the current one
/// is not empty and the next tensor would make its data larger than the maximum size. With a
/// single shard, only `<prefix>.safetensors` is written. Otherwise the shards
/// `<prefix>-0000i-of-0000n.safetensors` and the index `<prefix>.safetensors.index.json` are
/// written; the index has `metadata.total_size` (the size of all tensor data) and a
/// `weight_map` in state dict order. Python reads the result, but the split is not guaranteed to
/// match `huggingface_hub` byte for byte.
///
/// Saving replaces an earlier checkpoint with the same prefix in `dir`. Files with the names of
/// the new ones are overwritten. After the new files are written, the entry point of the other
/// layout, which a loader could pick instead, is removed (the index after saving a single file,
/// `<prefix>.safetensors` after saving shards, which loaders such as transformers look for
/// first), and so are the shards that the earlier index (read before it is overwritten) lists,
/// named like numbered shards of this prefix, that the new checkpoint doesn't use. Other files
/// are left alone, including numbered files that no readable earlier index lists, and
/// directories; a symbolic link is removed, not its target. The names of different prefixes
/// never overlap (see [`ShardOptions::prefix`]).
///
/// If the state dict fails the checks below, nothing is written or removed. An IO error while
/// writing or removing can leave new files next to old ones: saving is not atomic.
///
/// The format has no place for the state dict's [versions](StateDict::versions), and each key
/// gets its own copy of the data, even for tensors that share storage.
///
/// # Errors
///
/// Returns an error for meta tensors (they have no data), the key `__metadata__`, which the
/// format reserves, and IO errors.
///
/// ```no_run
/// use homura::io::{ShardOptions, save_sharded};
/// use homura::nn::Module;
///
/// let model = homura::nn::Linear::new(1024, 1024);
/// save_sharded(&model.state_dict(), "checkpoint", ShardOptions::default().max_shard_size(1 << 20))?;
/// let state_dict = homura::load("checkpoint/model.safetensors.index.json")?;
/// # Ok::<(), homura::Error>(())
/// ```
#[track_caller]
pub fn save_sharded(
    state_dict: &StateDict,
    dir: impl AsRef<Path>,
    options: ShardOptions,
) -> Result<()> {
    let tensors = snapshot(state_dict)?;
    let mut shards: Vec<&[_]> = vec![];
    let (mut start, mut size) = (0, 0);
    for (i, (_, data)) in tensors.iter().enumerate() {
        let n = data_size(data) as u64;
        if i > start && size + n > options.max_shard_size {
            shards.push(&tensors[start..i]);
            (start, size) = (i, 0);
        }
        size += n;
    }
    shards.push(&tensors[start..]);

    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let metadata = options.metadata.as_ref();
    let prefix = &options.prefix;
    let single = format!("{prefix}.safetensors");
    let index_name = format!("{prefix}.safetensors.index.json");
    let old_shards = indexed_shards(dir, prefix);
    if let [shard] = shards[..] {
        safetensors::write(&mut File::create(dir.join(&single))?, shard, metadata)?;
        return remove_stale(dir, prefix, old_shards, &[single]);
    }
    let n = shards.len();
    let mut written = vec![index_name.clone()];
    let mut weight_map = String::new();
    let mut total_size = 0;
    for (i, shard) in shards.into_iter().enumerate() {
        let file = format!("{prefix}-{:05}-of-{n:05}.safetensors", i + 1);
        safetensors::write(&mut File::create(dir.join(&file))?, shard, metadata)?;
        written.push(file.clone());
        for (name, data) in shard {
            total_size += data_size(data) as u64;
            if !weight_map.is_empty() {
                weight_map.push_str(",\n");
            }
            weight_map.push_str("    ");
            json::write_string(&mut weight_map, name);
            weight_map.push_str(": ");
            json::write_string(&mut weight_map, &file);
        }
    }
    let index = format!(
        "{{\n  \"metadata\": {{\n    \"total_size\": {total_size}\n  }},\n  \"weight_map\": {{\n{weight_map}\n  }}\n}}\n"
    );
    std::fs::write(dir.join(index_name), index)?;
    remove_stale(dir, prefix, old_shards, &written)
}

/// The `<base>` of `<base>-<digits>-of-<digits>`, if `stem` has that form.
fn numbered_base(stem: &str) -> Option<&str> {
    let trim = |s: &str| s.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let n = trim(stem);
    let rest = stem[..n].strip_suffix("-of-").filter(|_| n < stem.len())?;
    let i = trim(rest);
    rest[..i].strip_suffix('-').filter(|_| i < rest.len())
}

/// Whether `name` is a numbered shard of `prefix`: `<prefix>-<digits>-of-<digits>.safetensors`.
fn is_numbered_shard(name: &str, prefix: &str) -> bool {
    name.strip_suffix(".safetensors")
        .and_then(numbered_base)
        .is_some_and(|base| base == prefix)
}

/// The shards that the index of `prefix` in `dir` lists, as far as they are named like numbered
/// shards of `prefix`: the files known to belong to the earlier checkpoint. None if there is no
/// index or it can't be read.
fn indexed_shards(dir: &Path, prefix: &str) -> Vec<String> {
    let path = dir.join(format!("{prefix}.safetensors.index.json"));
    let Ok(file) = File::open(path) else {
        return vec![];
    };
    if !file
        .metadata()
        .is_ok_and(|m| m.is_file() && m.len() <= MAX_INDEX_LEN)
    {
        return vec![];
    }
    let Ok(text) = std::io::read_to_string(file) else {
        return vec![];
    };
    let Ok(index) = json::parse(&text) else {
        return vec![];
    };
    let Some(weight_map) = index.get("weight_map").and_then(Json::as_object) else {
        return vec![];
    };
    weight_map
        .iter()
        .filter_map(|(_, file)| file.as_str())
        .filter(|file| is_numbered_shard(file, prefix))
        .map(String::from)
        .collect()
}

/// Removes the files of the earlier checkpoint with `prefix` in `dir` that are not among
/// `written` (see [`save_sharded`]): first the entry points (the single file and the index), then
/// `old_shards`, so that no remaining entry point names a removed shard. Missing files and
/// directories are skipped; symbolic links are removed themselves.
fn remove_stale(
    dir: &Path,
    prefix: &str,
    old_shards: Vec<String>,
    written: &[String],
) -> Result<()> {
    let mut stale = vec![
        format!("{prefix}.safetensors"),
        format!("{prefix}.safetensors.index.json"),
    ];
    stale.extend(old_shards);
    for name in stale {
        if written.contains(&name) {
            continue;
        }
        let path = dir.join(name);
        match std::fs::symlink_metadata(&path) {
            Ok(m) if !m.is_dir() => std::fs::remove_file(path)?,
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid sharded checkpoint index: {msg}"))
}

/// Loads the shards that the index open as `file` lists, in `weight_map` order. Shard names are
/// relative to `dir`; each shard is opened and read once.
///
/// # Safety
///
/// With [`Access::Map`], as for [`load_mmap`](super::load_mmap).
pub(super) unsafe fn read_index(mut file: File, dir: &Path, access: Access) -> Result<StateDict> {
    if file.metadata()?.len() > MAX_INDEX_LEN {
        return Err(bad("the file is too large"));
    }
    file.seek(SeekFrom::Start(0))?;
    let text = std::io::read_to_string(file).map_err(|_| bad("the file is not UTF-8"))?;
    let index = json::parse(&text).map_err(|e| match e {
        Error::Format(msg) => bad(&msg),
        e => e,
    })?;
    let weight_map = index
        .get("weight_map")
        .ok_or_else(|| bad("missing weight_map"))?
        .as_object()
        .ok_or_else(|| bad("weight_map is not an object"))?;

    // Shard files in order of first use, with the keys each one should hold.
    let mut shards: Vec<(&str, Vec<&str>)> = vec![];
    let mut position = HashMap::new();
    for (key, file) in weight_map {
        let file = Json::as_str(file).ok_or_else(|| bad(&format!("{key:?}: not a file name")))?;
        if !is_file_name(file) {
            return Err(bad(&format!(
                "{key:?}: {file:?} is not a file in the shards' directory"
            )));
        }
        let i = *position.entry(file).or_insert_with(|| {
            shards.push((file, vec![]));
            shards.len() - 1
        });
        shards[i].1.push(key);
    }

    let mut loaded: HashMap<String, Tensor> = HashMap::new();
    for (file, keys) in shards {
        // SAFETY: the caller upholds the contract of `load_mmap` for `Access::Map`.
        let mut tensors: HashMap<String, Tensor> = unsafe { load_shard(&dir.join(file), access) }?
            .into_iter()
            .collect();
        for key in keys {
            let t = tensors
                .remove(key)
                .ok_or_else(|| bad(&format!("{file:?} has no tensor {key:?}")))?;
            loaded.insert(key.to_string(), t);
        }
        if !tensors.is_empty() {
            let mut extra: Vec<_> = tensors.into_keys().collect();
            extra.sort();
            return Err(bad(&format!(
                "{file:?} has tensors that the index doesn't list: {extra:?}"
            )));
        }
    }
    Ok(weight_map
        .iter()
        .map(|(key, _)| {
            let t = loaded.remove(key).expect("every listed key was loaded");
            (key.clone(), t)
        })
        .collect())
}
