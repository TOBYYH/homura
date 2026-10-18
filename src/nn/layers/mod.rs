//! Layers: modules with a forward computation.

mod sequential;

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
    fn forward(&self, input: &Tensor) -> Tensor;
}

/// A layer shared through an `Arc` (in two [`Sequential`]s, or twice in one), which traversals
/// list once.
impl<T: Forward> Forward for Arc<T> {
    fn forward(&self, input: &Tensor) -> Tensor {
        (**self).forward(input)
    }
}
