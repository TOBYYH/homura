//! Loading models from Hugging Face repositories, transformers' `from_pretrained`:
//! [`FromPretrained`], which the models implement, and its options, results and errors.
//!
//! `from_pretrained` reads the repository's `config.json`, picks its weight files
//! ([`Repo::weight_files`]), builds the model of the configuration on the meta device, reads the
//! weights, renames them for the model (the base model's prefix, see below), assigns them to the
//! model's parameters and buffers and converts them to the dtype (see
//! [`PretrainedOptions::dtype`]). The model is returned in eval mode, as transformers returns it.
//! It only uses homura's public API; with homura's core alone, the same is:
//!
//! ```no_run
//! use homura::nn::{LoadOptions, Module};
//! use homura::Device;
//! use homura_models::resnet::{ResNetConfig, ResNetForImageClassification};
//!
//! let config = ResNetConfig::from_file("resnet-50/config.json")?;
//! let model = homura::with_device(Device::Meta, || ResNetForImageClassification::new(&config));
//! let state_dict = homura::load("resnet-50/model.safetensors")?;
//! model.load_state_dict_with(&state_dict, LoadOptions::default().assign(true))?;
//! model.eval();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! **The base model's prefix.** The classification model's state dict has the base model's
//! under a prefix ([`FromPretrained::BASE_MODEL_PREFIX`], `resnet.` for ResNet). As in
//! transformers, each key of the weights is renamed if that gives a key of the model: one that
//! starts with the prefix loses it if the rest is the model's key, and otherwise one gets the
//! prefix if that is the model's key. So the base model loads from a classification repository
//! (the classification head's keys are left unused, see [`PretrainedReport::unused_keys`]), and
//! the classification model from a base model's repository, except that its head is missing:
//! transformers initializes the parameters that the weights don't have, and homura returns an
//! error ([`PretrainedError::Load`]).
//!
//! **Differences from transformers.** The weights must have every parameter and buffer of the
//! model (see above). Two keys of the weights that are renamed to the same key are an error
//! (transformers loads one of them). The versions of a `.bin` file's `_metadata` aren't read,
//! as transformers doesn't read them: a BatchNorm's count `num_batches_tracked` may be missing
//! (it is then 0). Variants (`model.fp16.safetensors`), subfolders, GGUF, quantization and
//! device maps aren't supported.
//!
//! **Memory maps.** [`FromPretrained::from_pretrained`] always reads the weights into memory of
//! their own. Only the unsafe [`FromPretrained::from_pretrained_mmap`] maps the weight files, so
//! that the model's tensors use the pages of the files, whose safety contract the caller upholds.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use homura::nn::{LoadError, LoadOptions, Module, StateDict};
use homura::{DType, Device};

use crate::config::{CommonConfig, ConfigError};
use crate::hub::{HubError, Repo, WeightFiles, WeightFormat};

/// The configuration of a model that [`FromPretrained`] loads.
pub trait PretrainedConfig: Clone + fmt::Debug + Sized {
    /// Reads and checks a `config.json` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be read, or if it isn't a valid configuration for
    /// the model.
    fn from_file(path: &Path) -> Result<Self, ConfigError>;

    /// The fields that every configuration has.
    #[track_caller]
    fn common(&self) -> &CommonConfig;

    /// The fields that every configuration has, to change them.
    #[track_caller]
    fn common_mut(&mut self) -> &mut CommonConfig;
}

/// A model that loads from Hugging Face repositories (transformers' `PreTrainedModel`).
///
/// A model gives its configuration type, its base model's prefix and its constructor; the
/// provided methods are the loading flow (see the [module documentation](self)).
///
/// ```no_run
/// use homura::nn::Module;
/// use homura_models::hub::{Repo, Revision};
/// use homura_models::pretrained::{FromPretrained, PretrainedOptions};
/// use homura_models::resnet::ResNetForImageClassification;
///
/// let repo = Repo::cached("microsoft/resnet-50", Revision::main())?;
/// let model = ResNetForImageClassification::from_pretrained(&repo, PretrainedOptions::default())?;
/// let logits = homura::with_no_grad(|| model.forward(&homura::zeros([1, 3, 224, 224])));
/// assert_eq!(logits.shape(), [1, 1000]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait FromPretrained: Module + Sized {
    /// The model's configuration.
    type Config: PretrainedConfig;

    /// The name of the base model in the model with a head, which prefixes its keys
    /// (transformers' `base_model_prefix`): `"resnet"` for ResNet.
    const BASE_MODEL_PREFIX: &'static str;

    /// The model of a configuration that [`PretrainedConfig::from_file`] read; `from_pretrained`
    /// calls it on the meta device.
    fn from_config(config: &Self::Config) -> Self;

    /// The model of the repository, with its weights read into memory (never mapped), in eval
    /// mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the repository has no `config.json` or no weights, the configuration
    /// isn't valid, the weights can't be read, or they don't fit the model ([`PretrainedError`]).
    #[track_caller]
    fn from_pretrained(repo: &Repo, options: PretrainedOptions) -> Result<Self, PretrainedError> {
        Ok(Self::from_pretrained_with_info(repo, options)?.model)
    }

    /// Like [`from_pretrained`](FromPretrained::from_pretrained), with the configuration as read
    /// (and with the dtype of the model) and a report of the loading.
    ///
    /// # Errors
    ///
    /// As for [`from_pretrained`](FromPretrained::from_pretrained).
    #[track_caller]
    fn from_pretrained_with_info(
        repo: &Repo,
        options: PretrainedOptions,
    ) -> Result<LoadedModel<Self>, PretrainedError> {
        // SAFETY: reading has no preconditions; only `Access::Map` has.
        unsafe { load(repo, options, Access::Read) }
    }

    /// Like [`from_pretrained_with_info`](FromPretrained::from_pretrained_with_info), but maps
    /// the weight files into memory ([`homura::io::load_mmap`]) instead of reading them: the
    /// model's tensors use the pages of the files, unless the dtype conversion gives them memory
    /// of their own.
    ///
    /// ```no_run
    /// use homura_models::hub::{Repo, Revision};
    /// use homura_models::pretrained::{FromPretrained, PretrainedOptions};
    /// use homura_models::resnet::ResNetForImageClassification;
    ///
    /// let repo = Repo::cached("microsoft/resnet-50", Revision::main())?;
    /// // SAFETY: nothing modifies or truncates the files of this cache while the program runs
    /// // (huggingface_hub itself writes new blobs rather than changing existing ones).
    /// let loaded = unsafe {
    ///     ResNetForImageClassification::from_pretrained_mmap(&repo, PretrainedOptions::default())
    /// }?;
    /// let x = homura::zeros([1, 3, 224, 224]);
    /// let logits = homura::with_no_grad(|| loaded.model.forward(&x));
    /// assert_eq!(logits.shape(), [1, 1000]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Safety
    ///
    /// From before this call, during it and until every tensor that holds data of the
    /// repository's weight files is dropped, the mapped files (the file that
    /// [`Repo::weight_files`] picks, or each shard of the index, at the repository's root) must
    /// not be modified or truncated, by this or any other process. Those tensors are the model's
    /// parameters and buffers, and every tensor sharing their storage: views and `detach` aliases
    /// of them, handles from `state_dict()` and the like, and parameters assigned to other modules
    /// or held by optimizers. Dropping the [`LoadedModel`] or the model doesn't end the period
    /// while such tensors are alive. Modifying a mapped file is undefined behaviour; truncating it
    /// can crash the process.
    ///
    /// # Errors
    ///
    /// As for [`from_pretrained`](FromPretrained::from_pretrained).
    #[track_caller]
    unsafe fn from_pretrained_mmap(
        repo: &Repo,
        options: PretrainedOptions,
    ) -> Result<LoadedModel<Self>, PretrainedError> {
        // SAFETY: the caller upholds the contract above, which is `load_mmap`'s.
        unsafe { load(repo, options, Access::Map) }
    }
}

/// The options of [`FromPretrained::from_pretrained`]. There are none that map files: only
/// [`FromPretrained::from_pretrained_mmap`] does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PretrainedOptions {
    format: Option<WeightFormat>,
    dtype: Option<DType>,
}

impl PretrainedOptions {
    /// Loads only weights of `format` (transformers' `use_safetensors=True` or `False`); by
    /// default, safetensors files are preferred to PyTorch files ([`Repo::weight_files`]). A
    /// configuration's `transformers_weights` is loaded whatever the format.
    #[must_use]
    #[track_caller]
    pub fn format(mut self, format: WeightFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Converts the model's floating-point parameters and buffers to `dtype` (`dtype=...`).
    /// Without it (`dtype="auto"`), the dtype is the configuration's (`dtype`, or the older name
    /// `torch_dtype`), else the `dtype` of a sharded checkpoint's index metadata, else that of
    /// the weights' first floating-point tensor (transformers reads the first shard's, in file
    /// name order; homura the first in the index's order); nothing is converted when the weights
    /// have it already. Integer buffers, such as the counts of batch normalizations, keep their
    /// dtype.
    ///
    /// # Panics
    ///
    /// Panics if `dtype` isn't floating point.
    #[must_use]
    #[track_caller]
    pub fn dtype(mut self, dtype: DType) -> Self {
        assert!(
            dtype.is_floating_point(),
            "the dtype of a model must be floating point, found {dtype:?}"
        );
        self.dtype = Some(dtype);
        self
    }
}

/// A model that [`FromPretrained`] loaded, with its configuration and a report of the loading.
#[derive(Debug)]
#[non_exhaustive]
pub struct LoadedModel<M: FromPretrained> {
    /// The model, in eval mode.
    pub model: M,
    /// The configuration of the repository, with the dtype of the model (transformers sets
    /// `config.dtype` too).
    pub config: M::Config,
    /// What loading did.
    pub report: PretrainedReport,
}

/// What [`FromPretrained::from_pretrained_with_info`] did, the counterpart of transformers'
/// loading warnings and `output_loading_info`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PretrainedReport {
    /// The weight file read (an index, with its shards).
    pub weights: WeightFiles,
    /// The number of keys of the weights that lost the base model's prefix.
    pub prefix_removed: usize,
    /// The number of keys of the weights that got the base model's prefix.
    pub prefix_added: usize,
    /// The keys of the weights (after renaming) that the model doesn't have, in the weights'
    /// order: they aren't loaded.
    pub unused_keys: Vec<String>,
    /// The dtype of the model's floating-point tensors; `None` if the weights have none.
    pub dtype: Option<DType>,
    /// Whether tensors were converted to `dtype`.
    pub converted: bool,
    /// What reading the configuration warned about
    /// ([`CommonConfig::warnings`](crate::config::CommonConfig::warnings)).
    pub config_warnings: Vec<String>,
}

/// Why [`FromPretrained::from_pretrained`] failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum PretrainedError {
    /// The repository, its `config.json` or its weights aren't there, or a name isn't valid.
    Hub(HubError),
    /// The configuration can't be read or isn't valid.
    Config(ConfigError),
    /// The weights can't be read: an IO error, or a file that is malformed or unsupported.
    Weights {
        /// The file that [`Repo::weight_files`] picked.
        path: PathBuf,
        /// What reading it returned.
        error: homura::Error,
    },
    /// The weights hold something else than the model's checkpoint: keys that are renamed to the
    /// same key, or an index whose metadata has a dtype that isn't floating point.
    Checkpoint(String),
    /// The weights don't fit the model: missing keys (such as a classification head when
    /// loading from a base model's weights) or tensors of other shapes.
    Load(LoadError),
}

impl fmt::Display for PretrainedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PretrainedError::Hub(e) => e.fmt(f),
            PretrainedError::Config(e) => e.fmt(f),
            PretrainedError::Weights { path, error } => {
                write!(f, "can't read the weights {}: {error}", path.display())
            }
            PretrainedError::Checkpoint(message) => f.write_str(message),
            PretrainedError::Load(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for PretrainedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PretrainedError::Hub(e) => Some(e),
            PretrainedError::Config(e) => Some(e),
            PretrainedError::Weights { error, .. } => Some(error),
            PretrainedError::Checkpoint(_) => None,
            PretrainedError::Load(e) => Some(e),
        }
    }
}

impl From<HubError> for PretrainedError {
    fn from(e: HubError) -> Self {
        PretrainedError::Hub(e)
    }
}

impl From<ConfigError> for PretrainedError {
    fn from(e: ConfigError) -> Self {
        PretrainedError::Config(e)
    }
}

impl From<LoadError> for PretrainedError {
    fn from(e: LoadError) -> Self {
        PretrainedError::Load(e)
    }
}

/// How the weights are read.
#[derive(Clone, Copy)]
enum Access {
    Read,
    /// Mapped: the caller of `from_pretrained_mmap` upholds `load_mmap`'s contract.
    Map,
}

/// The loading flow of [`FromPretrained`].
///
/// # Safety
///
/// With `Access::Map`, the contract of [`FromPretrained::from_pretrained_mmap`].
#[track_caller]
unsafe fn load<M: FromPretrained>(
    repo: &Repo,
    options: PretrainedOptions,
    access: Access,
) -> Result<LoadedModel<M>, PretrainedError> {
    let mut config = M::Config::from_file(&repo.path("config.json")?)?;
    let weights = repo.weight_files(
        options.format,
        config.common().transformers_weights.as_deref(),
    )?;
    let model = homura::with_device(Device::Meta, || M::from_config(&config));
    // The shards of an index are files of the repository's root, as transformers finds them,
    // also for a `transformers_weights` index in a subdirectory.
    let read = match (access, weights.sharded) {
        (Access::Read, false) => homura::load(&weights.path),
        (Access::Read, true) => homura::io::load_index(&weights.path, repo.dir()),
        // SAFETY: the caller upholds `load_mmap`'s contract (see `from_pretrained_mmap`).
        (Access::Map, false) => unsafe { homura::io::load_mmap(&weights.path) },
        // SAFETY: as above.
        (Access::Map, true) => unsafe { homura::io::load_index_mmap(&weights.path, repo.dir()) },
    };
    let state_dict = read.map_err(|error| PretrainedError::Weights {
        path: weights.path.clone(),
        error,
    })?;
    let weights_dtype = state_dict
        .values()
        .find(|t| t.is_floating_point())
        .map(|t| t.dtype());

    let model_keys: HashSet<String> = model.state_dict().keys().map(str::to_owned).collect();
    let renamed = rename(state_dict, &model_keys, M::BASE_MODEL_PREFIX)?;
    let mut state_dict = renamed.state_dict;
    let unused_keys: Vec<String> = state_dict
        .keys()
        .filter(|k| !model_keys.contains(*k))
        .map(str::to_owned)
        .collect();
    for key in &unused_keys {
        state_dict.remove(key);
    }
    model.load_state_dict_with(&state_dict, LoadOptions::default().assign(true))?;
    drop(state_dict);

    let auto = options.dtype.is_none() && config.common().dtype.is_none();
    let index_dtype = if weights.sharded && auto {
        index_dtype(&weights.path)?
    } else {
        None
    };
    let dtype = options
        .dtype
        .or(config.common().dtype)
        .or(index_dtype)
        .or(weights_dtype);
    let mut converted = false;
    if let Some(dtype) = dtype {
        converted = model
            .state_dict()
            .values()
            .any(|t| t.is_floating_point() && t.dtype() != dtype);
        if converted {
            model.to(dtype);
        }
        config.common_mut().dtype = Some(dtype);
    }
    model.eval();
    let report = PretrainedReport {
        weights,
        prefix_removed: renamed.removed,
        prefix_added: renamed.added,
        unused_keys,
        dtype,
        converted,
        config_warnings: config.common().warnings.clone(),
    };
    Ok(LoadedModel {
        model,
        config,
        report,
    })
}

struct Renamed {
    state_dict: StateDict,
    removed: usize,
    added: usize,
}

/// The keys of `state_dict` with the base model's prefix removed or added where that gives a key
/// of the model (transformers' renaming for `base_model_prefix`, key by key). The result has no
/// versions, as transformers 5 doesn't read `_metadata`.
fn rename(
    state_dict: StateDict,
    model_keys: &HashSet<String>,
    prefix: &str,
) -> Result<Renamed, PretrainedError> {
    let dotted = format!("{prefix}.");
    let (mut removed, mut added) = (0, 0);
    let mut names = Vec::with_capacity(state_dict.len());
    let mut seen = HashSet::with_capacity(state_dict.len());
    for key in state_dict.keys() {
        let name = match key.strip_prefix(&dotted) {
            Some(rest) if model_keys.contains(rest) => {
                removed += 1;
                rest.to_owned()
            }
            _ if model_keys.contains(&format!("{dotted}{key}")) => {
                added += 1;
                format!("{dotted}{key}")
            }
            _ => key.to_owned(),
        };
        if !seen.insert(name.clone()) {
            return Err(PretrainedError::Checkpoint(format!(
                "the weights have two keys for {name} (with and without the prefix `{dotted}`)"
            )));
        }
        names.push(name);
    }
    let mut names = names.into_iter();
    let state_dict = state_dict.map_keys(|_| names.next());
    Ok(Renamed {
        state_dict,
        removed,
        added,
    })
}

/// The `dtype` of the metadata of a sharded checkpoint's index, if it has one.
fn index_dtype(path: &Path) -> Result<Option<DType>, PretrainedError> {
    let bad = |message: String| PretrainedError::Checkpoint(message);
    let text = std::fs::read_to_string(path)
        .map_err(|e| bad(format!("can't read the index {}: {e}", path.display())))?;
    let index: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| bad(format!("the index {} isn't JSON: {e}", path.display())))?;
    let Some(found) = index.get("metadata").and_then(|m| m.get("dtype")) else {
        return Ok(None);
    };
    match found.as_str().and_then(crate::config::dtype_from_name) {
        Some(dtype) => Ok(Some(dtype)),
        None => Err(bad(format!(
            "the metadata of the index {} has the dtype {found}, which isn't a floating-point \
             dtype of homura",
            path.display()
        ))),
    }
}
