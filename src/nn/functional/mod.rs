//! The computations of the layers as functions, like PyTorch's `torch.nn.functional`.
//!
//! The functions take the parameters as arguments, so they also work without a module:
//! `linear(&x, &weight, Some(&bias))` is the forward computation of a [`Linear`](super::Linear)
//! layer, `layer_norm(&x, [4], Some(&weight), Some(&bias), 1e-5)` that of a
//! [`LayerNorm`](super::LayerNorm). The activations are also tensor methods (`x.relu()`); the
//! functions are here for code written in PyTorch's `F.relu(x)` style. The losses take PyTorch's
//! optional arguments in its order, or as [`CrossEntropyOptions`] for [`cross_entropy`], which
//! has more of them.

mod activation;
mod dropout;
mod embedding;
mod linear;
mod loss;
mod norm;
mod pad;

pub use activation::{gelu, log_softmax, logsigmoid, relu, sigmoid, silu, softmax, tanh};
pub use dropout::dropout;
pub use embedding::embedding;
pub use linear::linear;
pub use loss::{
    CrossEntropyOptions, Reduction, binary_cross_entropy_with_logits, cross_entropy, mse_loss,
    nll_loss,
};
pub use norm::{layer_norm, rms_norm};
pub use pad::pad;
