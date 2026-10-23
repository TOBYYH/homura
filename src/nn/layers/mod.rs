//! Layers: modules with a forward computation.

mod activation;
mod batch_norm;
mod conv;
mod dropout;
mod embedding;
mod flatten;
mod linear;
mod norm;
mod pool;
mod sequential;

pub use activation::{Gelu, Relu, Sigmoid, Silu, Softmax, Tanh};
pub use batch_norm::BatchNorm2d;
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use linear::Linear;
pub use norm::{LayerNorm, RmsNorm};
pub use pool::{AdaptiveAvgPool2d, MaxPool2d};
pub use sequential::Sequential;

use std::sync::Arc;

use super::Module;
use crate::Tensor;

/// A module whose forward computation takes one tensor and returns one, so that it can be a
/// layer of a [`Sequential`].
///
/// Layers also have `forward` as an ordinary method, which is what `layer.forward(&x)` calls;
/// this trait lets a [`Sequential`] call it on layers of any type.
pub trait Forward: Module {
    /// The output of the layer for `input`.
    #[track_caller]
    fn forward(&self, input: &Tensor) -> Tensor;
}

/// A layer shared through an `Arc` (in two [`Sequential`]s, or twice in one), which traversals
/// list once.
impl<T: Forward> Forward for Arc<T> {
    fn forward(&self, input: &Tensor) -> Tensor {
        (**self).forward(input)
    }
}

/// A size for the height and the width of a 2-D layer: one number for both (`3`) or a pair
/// (`[3, 2]`), like PyTorch's `int | tuple[int, int]` arguments.
pub trait IntoPair {
    /// The pair: `[n, n]` for a number.
    #[track_caller]
    fn into_pair(self) -> [usize; 2];
}

impl IntoPair for usize {
    fn into_pair(self) -> [usize; 2] {
        [self, self]
    }
}

impl IntoPair for [usize; 2] {
    fn into_pair(self) -> [usize; 2] {
        self
    }
}
