//! What the configurations of all models have: transformers' `PreTrainedConfig` fields that
//! homura uses (the model type, labels, dtype, weight file), the error type, and the reading of
//! `config.json` fields.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use homura::DType;
use serde_json::{Map, Value};

/// Why a configuration can't be read or used.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// The file can't be read.
    Io {
        /// The file.
        path: PathBuf,
        /// What reading it returned.
        error: std::io::Error,
    },
    /// The text isn't a JSON object, a field has a type or value that the model doesn't accept,
    /// or the configuration is for another model type.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io { path, error } => {
                write!(
                    f,
                    "can't read the configuration {}: {error}",
                    path.display()
                )
            }
            ConfigError::Invalid(message) => write!(f, "invalid configuration: {message}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io { error, .. } => Some(error),
            ConfigError::Invalid(_) => None,
        }
    }
}

/// The fields of a configuration that don't depend on the model, as transformers'
/// `PreTrainedConfig` reads them from `config.json`. Each model's configuration has one (for
/// example [`ResNetConfig::common`](crate::resnet::ResNetConfig::common)).
///
/// The labels follow transformers: without `id2label`, `num_labels` (2 if it is missing too)
/// gives the labels `LABEL_0`, `LABEL_1`, ...; with `id2label`, its entries are the labels, unless
/// `num_labels` is also given and differs from their number: then, as transformers does after a
/// warning, `num_labels` wins and the labels become `LABEL_0`, ... (the warning is in
/// [`warnings`](CommonConfig::warnings)). Unlike transformers, a negative `num_labels` is an
/// error rather than no labels, and so is one above 2^20.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CommonConfig {
    /// `model_type`: the model's name in transformers (`"resnet"`). A configuration without it is
    /// read as one for the model that reads it.
    pub model_type: String,
    /// `architectures`: the model classes that the weights are for
    /// (`["ResNetForImageClassification"]`).
    pub architectures: Option<Vec<String>>,
    /// The dtype of the weights: `dtype`, or the older name `torch_dtype` when `dtype` is missing
    /// or `null`.
    pub dtype: Option<DType>,
    /// `id2label`: the name of each label by its index (an empty map for no labels).
    pub id2label: BTreeMap<i64, String>,
    /// `label2id`: the index of each label name.
    pub label2id: BTreeMap<String, i64>,
    /// `transformers_weights`: the weight file to load instead of the usual names, a
    /// `.safetensors` or `.safetensors.index.json` file of the repository
    /// ([`Repo::weight_files`](crate::hub::Repo::weight_files)).
    pub transformers_weights: Option<String>,
    /// `transformers_version`: the version of transformers that wrote the configuration.
    pub transformers_version: Option<String>,
    /// What transformers warns about when it reads the configuration, such as a `num_labels`
    /// that disagrees with `id2label`.
    pub warnings: Vec<String>,
}

impl CommonConfig {
    /// The fields of a new configuration for `model_type`, with 2 labels, as transformers'
    /// defaults.
    #[track_caller]
    pub fn new(model_type: &str) -> CommonConfig {
        let mut config = CommonConfig {
            model_type: model_type.to_owned(),
            architectures: None,
            dtype: None,
            id2label: BTreeMap::new(),
            label2id: BTreeMap::new(),
            transformers_weights: None,
            transformers_version: None,
            warnings: Vec::new(),
        };
        config.set_num_labels(2);
        config
    }

    /// The number of labels: the number of entries of `id2label`.
    #[track_caller]
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    /// Replaces the labels with `n` labels named `LABEL_0`, ..., `LABEL_{n-1}` (transformers'
    /// `config.num_labels = n`), unless there are already `n`.
    #[track_caller]
    pub fn set_num_labels(&mut self, n: usize) {
        if self.num_labels() != n {
            self.name_labels(n);
        }
    }

    fn name_labels(&mut self, n: usize) {
        self.id2label = (0..n as i64).map(|i| (i, format!("LABEL_{i}"))).collect();
        self.label2id = self
            .id2label
            .iter()
            .map(|(&i, name)| (name.clone(), i))
            .collect();
    }

    /// Reads the fields from a `config.json` object for `model_type`. The other fields are left to
    /// the model's configuration.
    pub(crate) fn from_json(fields: &Fields<'_>, model_type: &str) -> Result<Self, ConfigError> {
        let mut config = CommonConfig::new(model_type);
        if let Some(found) = fields.string("model_type")?
            && found != model_type
        {
            return Err(invalid(format!(
                "the configuration is for a model of type `{found}`, not `{model_type}`"
            )));
        }
        config.architectures = fields.strings("architectures")?;
        config.dtype = fields.dtype("dtype")?;
        if config.dtype.is_none() {
            config.dtype = fields.dtype("torch_dtype")?;
        }
        config.transformers_weights = fields.string("transformers_weights")?;
        if let Some(name) = &config.transformers_weights {
            crate::hub::check_transformers_weights(name).map_err(ConfigError::Invalid)?;
        }
        config.transformers_version = fields.string("transformers_version")?;

        let num_labels = fields.integer("num_labels")?;
        if let Some(n) = num_labels {
            if n < 0 {
                return Err(invalid(format!("`num_labels` is negative: {n}")));
            }
            if n > MAX_LABELS as i64 {
                return Err(invalid(format!(
                    "`num_labels` is {n}, more than the {MAX_LABELS} labels that homura reads"
                )));
            }
        }
        match fields.id2label()? {
            None => config.name_labels(num_labels.map_or(2, |n| n as usize)),
            Some(id2label) => {
                let given = id2label.len();
                config.id2label = id2label;
                config.label2id = fields.label2id()?.unwrap_or_else(|| {
                    config
                        .id2label
                        .iter()
                        .map(|(&i, name)| (name.clone(), i))
                        .collect()
                });
                if let Some(n) = num_labels
                    && n as usize != given
                {
                    config.warnings.push(format!(
                        "`num_labels` ({n}) disagrees with the {given} entries of `id2label`: \
                         the labels are LABEL_0, ... as `num_labels` says"
                    ));
                    config.name_labels(n as usize);
                }
            }
        }
        Ok(config)
    }
}

/// The floating-point dtype of homura that a PyTorch dtype name gives (`getattr(torch, name)`).
pub(crate) fn dtype_from_name(name: &str) -> Option<DType> {
    match name {
        "float32" | "float" => Some(DType::F32),
        "float64" | "double" => Some(DType::F64),
        "float16" | "half" => Some(DType::F16),
        "bfloat16" => Some(DType::BF16),
        _ => None,
    }
}

/// The most labels that a configuration file may give with `num_labels` (each gets a name).
const MAX_LABELS: usize = 1 << 20;

pub(crate) fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

/// The JSON object of a configuration file.
pub(crate) fn parse(text: &str) -> Result<Map<String, Value>, ConfigError> {
    match serde_json::from_str(text) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(invalid("the configuration isn't a JSON object")),
        Err(e) => Err(invalid(format!("the configuration isn't valid JSON: {e}"))),
    }
}

/// The text of a configuration file.
pub(crate) fn read(path: &Path) -> Result<String, ConfigError> {
    std::fs::read_to_string(path).map_err(|error| ConfigError::Io {
        path: path.to_owned(),
        error,
    })
}

/// The fields of a `config.json` object. A missing field and `null` are both "not given".
pub(crate) struct Fields<'a>(pub(crate) &'a Map<String, Value>);

impl Fields<'_> {
    fn get(&self, name: &str) -> Option<&Value> {
        self.0.get(name).filter(|v| !v.is_null())
    }

    fn wrong(name: &str, expected: &str, found: &Value) -> ConfigError {
        invalid(format!("`{name}` must be {expected}, found {found}"))
    }

    pub(crate) fn string(&self, name: &str) -> Result<Option<String>, ConfigError> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(v) => Err(Self::wrong(name, "a string", v)),
        }
    }

    pub(crate) fn bool(&self, name: &str) -> Result<Option<bool>, ConfigError> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(v) => Err(Self::wrong(name, "true or false", v)),
        }
    }

    pub(crate) fn integer(&self, name: &str) -> Result<Option<i64>, ConfigError> {
        match self.get(name) {
            None => Ok(None),
            Some(v) => v
                .as_i64()
                .map(Some)
                .ok_or_else(|| Self::wrong(name, "an integer", v)),
        }
    }

    /// A positive integer (sizes of 0 are rejected before a model is built).
    pub(crate) fn size(&self, name: &str) -> Result<Option<usize>, ConfigError> {
        match self.get(name) {
            None => Ok(None),
            Some(v) => match v.as_u64() {
                Some(n) if n > 0 && usize::try_from(n).is_ok() => Ok(Some(n as usize)),
                _ => Err(Self::wrong(name, "a positive integer", v)),
            },
        }
    }

    /// A list of positive integers.
    pub(crate) fn sizes(&self, name: &str) -> Result<Option<Vec<usize>>, ConfigError> {
        let expected = "a list of positive integers";
        match self.get(name) {
            None => Ok(None),
            Some(v @ Value::Array(items)) => items
                .iter()
                .map(|item| match item.as_u64() {
                    Some(n) if n > 0 && usize::try_from(n).is_ok() => Ok(n as usize),
                    _ => Err(Self::wrong(name, expected, v)),
                })
                .collect::<Result<_, _>>()
                .map(Some),
            Some(v) => Err(Self::wrong(name, expected, v)),
        }
    }

    fn strings(&self, name: &str) -> Result<Option<Vec<String>>, ConfigError> {
        let expected = "a list of strings";
        match self.get(name) {
            None => Ok(None),
            Some(v @ Value::Array(items)) => items
                .iter()
                .map(|item| match item {
                    Value::String(s) => Ok(s.clone()),
                    _ => Err(Self::wrong(name, expected, v)),
                })
                .collect::<Result<_, _>>()
                .map(Some),
            Some(v) => Err(Self::wrong(name, expected, v)),
        }
    }

    /// A dtype name, as `getattr(torch, name)` takes it; only floating-point dtypes of homura.
    fn dtype(&self, name: &str) -> Result<Option<DType>, ConfigError> {
        let Some(found) = self.string(name)? else {
            return Ok(None);
        };
        match dtype_from_name(&found) {
            Some(dtype) => Ok(Some(dtype)),
            None => Err(invalid(format!(
                "`{name}` must name a floating-point dtype (float32, float64, float16 or \
                 bfloat16), found \"{found}\""
            ))),
        }
    }

    /// `id2label`, with its keys (strings in JSON) as integers.
    fn id2label(&self) -> Result<Option<BTreeMap<i64, String>>, ConfigError> {
        let expected = "an object from label indices to names";
        let entries = match self.get("id2label") {
            None => return Ok(None),
            Some(Value::Object(entries)) => entries,
            Some(v) => return Err(Self::wrong("id2label", expected, v)),
        };
        let mut id2label = BTreeMap::new();
        for (key, name) in entries {
            let id: i64 = key.trim().parse().map_err(|_| {
                invalid(format!("the key \"{key}\" of `id2label` isn't an integer"))
            })?;
            let Value::String(name) = name else {
                return Err(invalid(format!(
                    "the label {key} of `id2label` must be a string, found {name}"
                )));
            };
            if id2label.insert(id, name.clone()).is_some() {
                return Err(invalid(format!("`id2label` has the label {id} twice")));
            }
        }
        Ok(Some(id2label))
    }

    /// `label2id`, whose indices may be integers or strings of integers.
    fn label2id(&self) -> Result<Option<BTreeMap<String, i64>>, ConfigError> {
        let entries = match self.get("label2id") {
            None => return Ok(None),
            Some(Value::Object(entries)) => entries,
            Some(v) => {
                return Err(Self::wrong(
                    "label2id",
                    "an object from label names to indices",
                    v,
                ));
            }
        };
        let mut label2id = BTreeMap::new();
        for (name, id) in entries {
            let id = match id {
                Value::Number(n) => n.as_i64(),
                Value::String(s) => s.trim().parse().ok(),
                _ => None,
            }
            .ok_or_else(|| {
                invalid(format!(
                    "the index of \"{name}\" in `label2id` must be an integer, found {id}"
                ))
            })?;
            label2id.insert(name.clone(), id);
        }
        Ok(Some(label2id))
    }
}
