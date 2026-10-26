use super::Forward;
use crate::Tensor;
use crate::nn::Module;

/// A layer that returns its input, PyTorch's `nn.Identity`: a placeholder where a model has a
/// layer in some configurations and none in others, so that the module tree and the keys stay
/// the same, as in transformers' models.
///
/// ```
/// use homura::nn::{Forward, Identity, Linear, Module, Sequential};
///
/// let head = |labels: usize| -> Box<dyn Forward + Send + Sync> {
///     if labels > 0 { Box::new(Linear::new(4, labels)) } else { Box::new(Identity) }
/// };
/// let x = homura::ones([2, 4]);
/// assert_eq!(head(3).forward(&x).shape(), [2, 3]);
/// assert_eq!(head(0).forward(&x).shape(), [2, 4]);
/// assert_eq!(Sequential::new().append(Identity).display().to_string(), "Sequential(\n  (0): Identity()\n)");
/// ```
#[derive(Module, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Identity;

impl Identity {
    /// `input` itself (the same tensor, not a copy), as PyTorch's `nn.Identity` returns its
    /// argument.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }
}

impl Forward for Identity {
    fn forward(&self, input: &Tensor) -> Tensor {
        Identity::forward(self, input)
    }
}
