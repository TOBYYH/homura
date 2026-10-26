//! transformers' ResNet (`models/resnet`): [`ResNetModel`], the features of images, and
//! [`ResNetForImageClassification`], with a classification head, as in the repositories
//! `microsoft/resnet-18` and `microsoft/resnet-50`.
//!
//! The module tree is transformers': the same type and field names, so the same state dict keys
//! (`resnet.embedder.embedder.convolution.weight`, `classifier.1.weight`), versions and printed
//! tree (with homura's names of the core layers, such as `Relu` for `ReLU`). A model is built
//! from a [`ResNetConfig`], which decides its layers at run time: the layers of a stage are
//! [`ResNetBasicLayer`]s or [`ResNetBottleNeckLayer`]s in a [`Sequential`](homura::nn::Sequential),
//! and the layers that transformers replaces by `nn.Identity` are enums with an `Identity`
//! variant ([`ResNetShortCutOrIdentity`], [`Activation`](crate::activations::Activation)).
//!
//! ```
//! use homura_models::resnet::{ResNetConfig, ResNetModel};
//!
//! let config = ResNetConfig::default().embedding_size(8).hidden_sizes([8, 16]).depths([1, 1]);
//! let model = ResNetModel::new(&config);
//! let output = model.forward(&homura::zeros([1, 3, 64, 64]));
//! assert_eq!(output.last_hidden_state.shape(), [1, 16, 8, 8]);
//! assert_eq!(output.pooler_output.shape(), [1, 16, 1, 1]);
//! ```
//!
//! # Differences from transformers
//!
//! - The forward computations take the pixel values only: there is no loss for `labels` and no
//!   output of each stage's hidden state (`output_hidden_states`); [`ResNetModel::forward`]
//!   returns a [`ResNetModelOutput`], the classification model the logits. `ResNetBackbone` and
//!   the configuration's `out_features` / `out_indices` aren't supported.
//! - The residual sum makes a new tensor where transformers adds in place.
//! - Configurations that transformers builds odd models from are errors (see
//!   [`ResNetConfig::validate`]), and only the activations of
//!   [`Activation::NAMES`](crate::activations::Activation::NAMES) are known.

mod config;
mod modeling;

pub use config::{ResNetConfig, ResNetLayerType};
pub use modeling::{
    ResNetBasicLayer, ResNetBottleNeckLayer, ResNetConvLayer, ResNetEmbeddings, ResNetEncoder,
    ResNetForImageClassification, ResNetModel, ResNetModelOutput, ResNetShortCut,
    ResNetShortCutOrIdentity, ResNetStage,
};
