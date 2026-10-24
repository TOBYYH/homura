use super::Forward;
use crate::Tensor;
use crate::nn::Module;

/// Merges the dimensions `start_dim..=end_dim` of its input into one, PyTorch's `nn.Flatten`,
/// with [`Tensor::flatten`]. By default (`Flatten::default()`) it keeps the first dimension, the
/// batch, and merges the others, like PyTorch.
///
/// The result is a view when the dimensions can be merged without copying, as for a contiguous
/// input, and a copy otherwise, like PyTorch's.
///
/// ```
/// use homura::nn::Flatten;
///
/// let x = homura::zeros([2, 3, 4, 5]);
/// assert_eq!(Flatten::default().forward(&x).shape(), [2, 60]);
/// assert_eq!(Flatten::new(0, 2).forward(&x).shape(), [24, 5]);
/// ```
#[derive(Module, Clone, Copy, Debug, PartialEq, Eq)]
#[module(extra_repr = Self::describe)]
#[non_exhaustive]
pub struct Flatten {
    /// The first dimension to merge; negative counts from the end.
    pub start_dim: i64,
    /// The last dimension to merge; negative counts from the end.
    pub end_dim: i64,
}

impl Flatten {
    /// A layer that merges the dimensions `start_dim..=end_dim`
    /// (`nn.Flatten(start_dim, end_dim)`). They are checked by the forward computation, as in
    /// PyTorch.
    #[track_caller]
    pub fn new(start_dim: i64, end_dim: i64) -> Flatten {
        Flatten { start_dim, end_dim }
    }

    /// `input` with the dimensions merged.
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's messages if a dimension is out of range for `input` (also for a
    /// zero-dimensional one, which has no dimension 1) or if `start_dim` comes after `end_dim`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.flatten(self.start_dim, self.end_dim)
    }
}

/// `start_dim = 1` and `end_dim = -1`, PyTorch's defaults.
impl Default for Flatten {
    fn default() -> Self {
        Flatten::new(1, -1)
    }
}

impl Flatten {
    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        format!("start_dim={}, end_dim={}", self.start_dim, self.end_dim)
    }
}

impl Forward for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        Flatten::forward(self, input)
    }
}
