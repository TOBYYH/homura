use std::fmt;
use std::path::Path;

use homura::DType;

use crate::activations::Activation;
use crate::config::{self, CommonConfig, ConfigError, Fields, invalid};

/// The layers of a ResNet: `"basic"` (two 3 × 3 convolutions, ResNet-18 and -34) or
/// `"bottleneck"` (1 × 1, 3 × 3 and 1 × 1 convolutions, ResNet-50 and larger).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResNetLayerType {
    /// [`ResNetBasicLayer`](super::ResNetBasicLayer).
    Basic,
    /// [`ResNetBottleNeckLayer`](super::ResNetBottleNeckLayer).
    Bottleneck,
}

impl fmt::Display for ResNetLayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ResNetLayerType::Basic => "basic",
            ResNetLayerType::Bottleneck => "bottleneck",
        })
    }
}

/// The configuration of a ResNet, transformers' `ResNetConfig`: what `config.json` of a
/// repository such as `microsoft/resnet-50` says, from which the models are built.
///
/// [`from_file`](ResNetConfig::from_file) reads `config.json`: the fields have transformers'
/// names and defaults (those of [`ResNetConfig::default`], ResNet-50's), other fields are
/// ignored, and the configuration is checked as the models need it (see
/// [`validate`](ResNetConfig::validate)). The setters change one field; the model constructors
/// panic if the result isn't valid.
///
/// ```
/// use homura_models::resnet::{ResNetConfig, ResNetLayerType};
///
/// let config = ResNetConfig::default()
///     .layer_type(ResNetLayerType::Basic)
///     .hidden_sizes([64, 128, 256, 512])
///     .depths([2, 2, 2, 2])
///     .num_labels(10);
/// assert!(config.validate().is_ok());
/// assert_eq!(config.common.id2label[&9], "LABEL_9");
///
/// let config = ResNetConfig::from_json(r#"{"model_type": "resnet", "hidden_sizes": []}"#);
/// assert!(config.is_err());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResNetConfig {
    /// The fields of every model: labels, dtype, ...
    pub common: CommonConfig,
    /// `num_channels`: the channels of the input images (3).
    pub num_channels: usize,
    /// `embedding_size`: the channels of the stem's output (64).
    pub embedding_size: usize,
    /// `hidden_sizes`: the channels of each stage's output (`[256, 512, 1024, 2048]`).
    pub hidden_sizes: Vec<usize>,
    /// `depths`: the layers of each stage (`[3, 4, 6, 3]`).
    pub depths: Vec<usize>,
    /// `layer_type`: the layers of the stages (bottleneck).
    pub layer_type: ResNetLayerType,
    /// `hidden_act`: the activation of the stem and of each layer's output (`"relu"`), one of
    /// [`Activation::NAMES`].
    pub hidden_act: String,
    /// `downsample_in_first_stage`: whether the first stage has a stride of 2 (false).
    pub downsample_in_first_stage: bool,
    /// `downsample_in_bottleneck`: whether a bottleneck layer's first 1 × 1 convolution has the
    /// stride, rather than its 3 × 3 convolution (false; configurations written by older
    /// transformers lack the field).
    pub downsample_in_bottleneck: bool,
}

/// transformers' defaults, ResNet-50's structure with 2 labels.
impl Default for ResNetConfig {
    fn default() -> Self {
        ResNetConfig {
            common: CommonConfig::new("resnet"),
            num_channels: 3,
            embedding_size: 64,
            hidden_sizes: vec![256, 512, 1024, 2048],
            depths: vec![3, 4, 6, 3],
            layer_type: ResNetLayerType::Bottleneck,
            hidden_act: "relu".to_owned(),
            downsample_in_first_stage: false,
            downsample_in_bottleneck: false,
        }
    }
}

impl ResNetConfig {
    /// The `model_type` of ResNet configurations.
    pub const MODEL_TYPE: &'static str = "resnet";

    /// Reads a `config.json` file (see [`from_json`](ResNetConfig::from_json)).
    #[track_caller]
    pub fn from_file(path: impl AsRef<Path>) -> Result<ResNetConfig, ConfigError> {
        ResNetConfig::from_json(&config::read(path.as_ref())?)
    }

    /// Reads the text of a `config.json` file: a JSON object whose `model_type`, if it has one,
    /// is `"resnet"`. The configuration is [validated](ResNetConfig::validate).
    #[track_caller]
    pub fn from_json(text: &str) -> Result<ResNetConfig, ConfigError> {
        let map = config::parse(text)?;
        let fields = Fields(&map);
        let defaults = ResNetConfig::default();
        let layer_type = match fields.string("layer_type")?.as_deref() {
            None => defaults.layer_type,
            Some("basic") => ResNetLayerType::Basic,
            Some("bottleneck") => ResNetLayerType::Bottleneck,
            Some(other) => {
                return Err(invalid(format!(
                    "`layer_type` must be \"basic\" or \"bottleneck\", found \"{other}\""
                )));
            }
        };
        let config = ResNetConfig {
            common: CommonConfig::from_json(&fields, Self::MODEL_TYPE)?,
            num_channels: fields
                .size("num_channels")?
                .unwrap_or(defaults.num_channels),
            embedding_size: fields
                .size("embedding_size")?
                .unwrap_or(defaults.embedding_size),
            hidden_sizes: fields
                .sizes("hidden_sizes")?
                .unwrap_or(defaults.hidden_sizes),
            depths: fields.sizes("depths")?.unwrap_or(defaults.depths),
            layer_type,
            hidden_act: fields.string("hidden_act")?.unwrap_or(defaults.hidden_act),
            downsample_in_first_stage: fields
                .bool("downsample_in_first_stage")?
                .unwrap_or(defaults.downsample_in_first_stage),
            downsample_in_bottleneck: fields
                .bool("downsample_in_bottleneck")?
                .unwrap_or(defaults.downsample_in_bottleneck),
        };
        config.validate()?;
        Ok(config)
    }

    /// Checks that the models can be built from this configuration: the sizes aren't 0, there
    /// are as many depths as hidden sizes and at least one, the hidden sizes of bottleneck
    /// layers are at least 4 (their inner convolutions have a quarter of the channels) and
    /// `hidden_act` is known.
    ///
    /// transformers builds models from some of these configurations: it uses only as many stages
    /// as the shorter list has, gives a stage of depth 0 one layer, and builds layers without
    /// channels.
    #[track_caller]
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.problem() {
            Some(message) => Err(invalid(message)),
            None => Ok(()),
        }
    }

    fn problem(&self) -> Option<String> {
        if self.num_channels == 0 || self.embedding_size == 0 {
            return Some("`num_channels` and `embedding_size` must be positive".to_owned());
        }
        if self.hidden_sizes.is_empty() || self.hidden_sizes.len() != self.depths.len() {
            return Some(format!(
                "`hidden_sizes` and `depths` must have the same nonzero length, found {:?} and {:?}",
                self.hidden_sizes, self.depths
            ));
        }
        if self.hidden_sizes.contains(&0) || self.depths.contains(&0) {
            return Some(format!(
                "`hidden_sizes` and `depths` must be positive, found {:?} and {:?}",
                self.hidden_sizes, self.depths
            ));
        }
        if self.layer_type == ResNetLayerType::Bottleneck
            && self.hidden_sizes.iter().any(|&h| h < 4)
        {
            return Some(format!(
                "the `hidden_sizes` of bottleneck layers must be at least 4, found {:?}",
                self.hidden_sizes
            ));
        }
        if Activation::from_name(&self.hidden_act).is_none() {
            return Some(format!(
                "`hidden_act` \"{}\" isn't one of {:?}",
                self.hidden_act,
                Activation::NAMES
            ));
        }
        None
    }

    /// The activation that `hidden_act` names.
    ///
    /// # Panics
    ///
    /// Panics if homura doesn't know the name.
    #[track_caller]
    pub fn activation(&self) -> Activation {
        match Activation::from_name(&self.hidden_act) {
            Some(activation) => activation,
            None => panic!(
                "invalid ResNet configuration: `hidden_act` \"{}\" isn't one of {:?}",
                self.hidden_act,
                Activation::NAMES
            ),
        }
    }

    /// The configuration, checked for building a model.
    #[track_caller]
    pub(crate) fn checked(&self) -> &ResNetConfig {
        if let Some(message) = self.problem() {
            panic!("invalid ResNet configuration: {message}");
        }
        self
    }

    /// This configuration with `num_channels`.
    #[must_use]
    #[track_caller]
    pub fn num_channels(mut self, num_channels: usize) -> ResNetConfig {
        self.num_channels = num_channels;
        self
    }

    /// This configuration with `embedding_size`.
    #[must_use]
    #[track_caller]
    pub fn embedding_size(mut self, embedding_size: usize) -> ResNetConfig {
        self.embedding_size = embedding_size;
        self
    }

    /// This configuration with `hidden_sizes`.
    #[must_use]
    #[track_caller]
    pub fn hidden_sizes(mut self, hidden_sizes: impl Into<Vec<usize>>) -> ResNetConfig {
        self.hidden_sizes = hidden_sizes.into();
        self
    }

    /// This configuration with `depths`.
    #[must_use]
    #[track_caller]
    pub fn depths(mut self, depths: impl Into<Vec<usize>>) -> ResNetConfig {
        self.depths = depths.into();
        self
    }

    /// This configuration with `layer_type`.
    #[must_use]
    #[track_caller]
    pub fn layer_type(mut self, layer_type: ResNetLayerType) -> ResNetConfig {
        self.layer_type = layer_type;
        self
    }

    /// This configuration with `hidden_act`.
    #[must_use]
    #[track_caller]
    pub fn hidden_act(mut self, hidden_act: &str) -> ResNetConfig {
        self.hidden_act = hidden_act.to_owned();
        self
    }

    /// This configuration with `downsample_in_first_stage`.
    #[must_use]
    #[track_caller]
    pub fn downsample_in_first_stage(mut self, downsample: bool) -> ResNetConfig {
        self.downsample_in_first_stage = downsample;
        self
    }

    /// This configuration with `downsample_in_bottleneck`.
    #[must_use]
    #[track_caller]
    pub fn downsample_in_bottleneck(mut self, downsample: bool) -> ResNetConfig {
        self.downsample_in_bottleneck = downsample;
        self
    }

    /// This configuration with `n` labels ([`CommonConfig::set_num_labels`]).
    #[must_use]
    #[track_caller]
    pub fn num_labels(mut self, n: usize) -> ResNetConfig {
        self.common.set_num_labels(n);
        self
    }

    /// This configuration with the dtype of the weights.
    #[must_use]
    #[track_caller]
    pub fn dtype(mut self, dtype: impl Into<Option<DType>>) -> ResNetConfig {
        self.common.dtype = dtype.into();
        self
    }
}
