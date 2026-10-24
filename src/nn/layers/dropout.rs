use super::Forward;
use crate::Tensor;
use crate::nn::{Module, Training, functional, python_float};

/// Dropout, PyTorch's `nn.Dropout`: in training mode, [`functional::dropout`] with probability
/// `p`; in evaluation mode ([`Module::eval`]), the input itself.
///
/// ```
/// use homura::nn::{Dropout, Module};
///
/// let dropout = Dropout::new(0.1);
/// let x = homura::ones([4, 8]);
/// assert_eq!(dropout.forward(&x).shape(), [4, 8]);
/// dropout.eval();
/// assert!(dropout.forward(&x).is_same(&x));
/// ```
#[derive(Module, Clone, Debug)]
#[module(extra_repr = Self::describe)]
#[non_exhaustive]
pub struct Dropout {
    /// The probability of zeroing an element.
    pub p: f64,
    pub training: Training,
}

impl Dropout {
    /// Dropout with probability `p` (`nn.Dropout(p)`), in training mode.
    ///
    /// # Panics
    ///
    /// Panics if `p` is less than 0 or more than 1, like PyTorch; a NaN `p` is accepted here and
    /// rejected by the forward computation, also like PyTorch.
    #[track_caller]
    pub fn new(p: f64) -> Dropout {
        if p < 0.0 || p > 1.0 {
            panic!("dropout probability has to be between 0 and 1, but got {p}");
        }
        Dropout {
            p,
            training: Training::default(),
        }
    }

    /// `input` with dropout in training mode, `input` itself in evaluation mode.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::dropout(input, self.p, self.training.get())
    }
}

/// `p = 0.5`, PyTorch's default.
impl Default for Dropout {
    fn default() -> Self {
        Dropout::new(0.5)
    }
}

impl Dropout {
    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        format!("p={}, inplace=False", python_float(self.p))
    }
}

impl Forward for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        Dropout::forward(self, input)
    }
}
