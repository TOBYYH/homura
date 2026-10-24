//! Activation layers: modules without parameters that apply an elementwise function, or softmax.

use super::Forward;
use crate::nn::Module;
use crate::{GeluApproximate, Tensor};

/// Implements the inherent `forward` and [`Forward`] of an activation layer.
macro_rules! activation {
    ($layer:ty, |$s:ident, $x:ident| $body:expr) => {
        impl $layer {
            /// The activation of `input`; the result is a new tensor.
            #[track_caller]
            pub fn forward(&self, input: &Tensor) -> Tensor {
                let ($s, $x) = (self, input);
                $body
            }
        }

        impl Forward for $layer {
            fn forward(&self, input: &Tensor) -> Tensor {
                <$layer>::forward(self, input)
            }
        }
    };
}

/// `max(x, 0)`, PyTorch's `nn.ReLU` (without `inplace`).
///
/// ```
/// use homura::nn::{Relu, Sequential, Linear};
///
/// let model = Sequential::new().append(Linear::new(4, 8)).append(Relu).append(Linear::new(8, 2));
/// assert_eq!(model.forward(&homura::ones([3, 4])).shape(), [3, 2]);
/// ```
#[derive(Module, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Relu;

activation!(Relu, |_s, x| x.relu());

/// The Gaussian error linear unit, PyTorch's `nn.GELU`: exact by default, or with the tanh
/// approximation.
///
/// ```
/// use homura::GeluApproximate;
/// use homura::nn::Gelu;
///
/// let gelu = Gelu::new().approximate(GeluApproximate::Tanh);
/// assert_eq!(gelu.forward(&homura::zeros([2])).to_vec::<f32>(), [0.0, 0.0]);
/// ```
#[derive(Module, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[module(extra_repr = Self::describe)]
pub struct Gelu {
    pub approximate: GeluApproximate,
}

impl Gelu {
    /// The exact GELU (`nn.GELU()`).
    #[track_caller]
    pub fn new() -> Gelu {
        Gelu::default()
    }

    /// This layer with `approximate` (`approximate='none'` or `'tanh'`).
    #[must_use]
    #[track_caller]
    pub fn approximate(mut self, approximate: GeluApproximate) -> Gelu {
        self.approximate = approximate;
        self
    }

    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        let approximate = match self.approximate {
            GeluApproximate::None => "none",
            GeluApproximate::Tanh => "tanh",
        };
        format!("approximate='{approximate}'")
    }
}

activation!(Gelu, |s, x| x.gelu(s.approximate));

/// `x * sigmoid(x)`, PyTorch's `nn.SiLU` (without `inplace`).
#[derive(Module, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Silu;

activation!(Silu, |_s, x| x.silu());

/// PyTorch's `nn.Tanh`.
#[derive(Module, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tanh;

activation!(Tanh, |_s, x| x.tanh());

/// `1 / (1 + e^-x)`, PyTorch's `nn.Sigmoid`.
#[derive(Module, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sigmoid;

activation!(Sigmoid, |_s, x| x.sigmoid());

/// Softmax along `dim`, PyTorch's `nn.Softmax(dim)`. The dimension is required: PyTorch's
/// deprecated choice of an implicit dimension (`nn.Softmax()`) isn't supported.
///
/// ```
/// use homura::nn::Softmax;
///
/// let y = Softmax::new(-1).forward(&homura::zeros([2, 4]));
/// assert_eq!(y.sum_dim(-1, false).to_vec::<f32>(), [1.0, 1.0]);
/// ```
#[derive(Module, Clone, Copy, Debug, PartialEq, Eq)]
#[module(extra_repr = Self::describe)]
pub struct Softmax {
    pub dim: i64,
}

impl Softmax {
    /// Softmax along `dim`, which may be negative to count from the end.
    #[track_caller]
    pub fn new(dim: i64) -> Softmax {
        Softmax { dim }
    }

    /// PyTorch's `extra_repr()` text.
    fn describe(&self) -> String {
        format!("dim={}", self.dim)
    }
}

activation!(Softmax, |s, x| x.softmax(s.dim));
