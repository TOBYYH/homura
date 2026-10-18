//! The computations of the layers as functions, like PyTorch's `torch.nn.functional`.
//!
//! The functions take the parameters as arguments, so they also work without a module:
//! `linear(&x, &weight, Some(&bias))` is the forward computation of a [`Linear`](super::Linear)
//! layer. The activations are also tensor methods (`x.relu()`); the functions are here for code
//! written in PyTorch's `F.relu(x)` style.

mod activation;
mod linear;

pub use activation::{gelu, log_softmax, relu, sigmoid, silu, softmax, tanh};
pub use linear::linear;
