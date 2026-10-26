//! Models for [homura] that use Hugging Face repositories as they are: a model reads the
//! repository's `config.json` and its weights (`model.safetensors` or `pytorch_model.bin`)
//! without converting them, because its module tree, its parameter names and its configuration
//! are those of the reference library's implementation, [transformers].
//!
//! Each architecture has one implementation, which follows the reference's model class over its
//! whole configuration range; checkpoints of other libraries with other module trees (torchvision
//! or timm ResNets, say) aren't supported. A model is built from its configuration at run time,
//! as in transformers, so the layers that a configuration chooses are enums and
//! [`Sequential`](homura::nn::Sequential)s rather than type parameters.
//!
//! - [`resnet`]: transformers' ResNet (`ResNetModel`, `ResNetForImageClassification`).
//! - [`config`]: the fields that every model's configuration has, such as the labels and dtype.
//! - [`activations`]: the activation functions that configurations name, transformers' `ACT2FN`.
//!
//! ```
//! use homura::nn::{LoadOptions, Module};
//! use homura::Device;
//! use homura_models::resnet::{ResNetConfig, ResNetForImageClassification};
//!
//! let config = ResNetConfig::from_json(r#"{
//!     "model_type": "resnet", "layer_type": "basic", "embedding_size": 8,
//!     "hidden_sizes": [8, 16], "depths": [1, 1], "num_labels": 3
//! }"#)?;
//! let model = ResNetForImageClassification::new(&config);
//! assert_eq!(model.forward(&homura::zeros([2, 3, 32, 32])).shape(), [2, 3]);
//!
//! // Built on the meta device, a model takes the tensors of a state dict without copying them.
//! let empty = homura::with_device(Device::Meta, || ResNetForImageClassification::new(&config));
//! empty.load_state_dict_with(&model.state_dict(), LoadOptions::default().assign(true))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [transformers]: https://github.com/huggingface/transformers

pub mod activations;
pub mod config;
pub mod resnet;
