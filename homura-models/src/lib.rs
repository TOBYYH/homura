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
//! - [`hub`]: repositories in local directories and in the Hugging Face cache (offline), and
//!   their weight files.
//! - [`pretrained`]: [`FromPretrained`](pretrained::FromPretrained), transformers'
//!   `from_pretrained`, which loads a model from a repository.
//! - [`resnet`]: transformers' ResNet (`ResNetModel`, `ResNetForImageClassification`).
//! - [`config`]: the fields that every model's configuration has, such as the labels and dtype.
//! - [`activations`]: the activation functions that configurations name, transformers' `ACT2FN`.
//!
//! ```no_run
//! use homura::nn::Module;
//! use homura_models::hub::{Repo, Revision};
//! use homura_models::pretrained::{FromPretrained, PretrainedOptions};
//! use homura_models::resnet::ResNetForImageClassification;
//!
//! // Downloaded before with `hf download microsoft/resnet-50`.
//! let repo = Repo::cached("microsoft/resnet-50", Revision::main())?;
//! let model = ResNetForImageClassification::from_pretrained(&repo, PretrainedOptions::default())?;
//! let logits = homura::with_no_grad(|| model.forward(&homura::zeros([1, 3, 224, 224])));
//! # let _ = logits;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A model can also be built from a configuration, without weights:
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
//! # Differences from transformers
//!
//! - Repositories are read offline, from a local directory or the Hugging Face cache; there are
//!   no downloads, image preprocessing (`preprocessor_config.json`), tokenizers or datasets yet.
//! - Missing weights are an error, where transformers initializes them randomly and warns (a
//!   classification head loaded from a base model's repository). There is no
//!   `ignore_mismatched_sizes`, weight `variant`, quantization, `device_map` or offloading.
//! - The ResNet has no `ResNetBackbone`, `output_hidden_states`, `return_dict`, or `labels` and
//!   loss in `forward`.
//! - File names of repositories are checked more strictly (see [`hub`]).
//!
//! [transformers]: https://github.com/huggingface/transformers

// The README's examples are compiled as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub mod activations;
pub mod config;
pub mod hub;
pub mod pretrained;
pub mod resnet;
