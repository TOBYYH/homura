use std::fmt;

use super::Forward;
use crate::Tensor;
use crate::nn::init::{self, FanMode, Nonlinearity};
use crate::nn::{Module, Parameter, functional};

/// A fully connected layer, PyTorch's `nn.Linear`: `y = x @ weightᵀ + bias`, with
/// [`functional::linear`].
///
/// `weight` has the shape `[out_features, in_features]` and `bias` the shape `[out_features]`,
/// like PyTorch's, so state dicts of PyTorch's `Linear` load as they are. Both are initialized
/// like PyTorch does (see [`reset_parameters`](Linear::reset_parameters)), with the default dtype
/// and device; convert the layer with [`Module::to`] for others.
///
/// ```
/// use homura::nn::{Linear, Module};
///
/// let layer = Linear::new(4, 3);
/// assert_eq!(layer.weight.shape(), [3, 4]);
/// let y = layer.forward(&homura::ones([2, 4]));
/// assert_eq!(y.shape(), [2, 3]);
///
/// let layer = Linear::new(4, 3).bias(false);
/// assert!(layer.bias.is_none());
/// assert_eq!(layer.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
/// ```
#[derive(Module)]
#[non_exhaustive]
pub struct Linear {
    /// The weights, `[out_features, in_features]`.
    pub weight: Parameter,
    /// The bias, `[out_features]`, if the layer has one.
    pub bias: Option<Parameter>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// A layer from `in_features` inputs to `out_features` outputs, with a bias
    /// (`nn.Linear(in_features, out_features)`).
    pub fn new(in_features: usize, out_features: usize) -> Linear {
        let layer = Linear {
            weight: Parameter::new(crate::zeros([out_features, in_features])),
            bias: Some(Parameter::new(crate::zeros([out_features]))),
            in_features,
            out_features,
        };
        layer.reset_parameters();
        layer
    }

    /// This layer with a bias if `bias` is true (`bias=True`, the default) and without one
    /// otherwise. A new bias is initialized like [`reset_parameters`](Linear::reset_parameters)
    /// does, with the dtype and device of the weights.
    #[must_use]
    pub fn bias(mut self, bias: bool) -> Linear {
        if !bias {
            self.bias = None;
        } else if self.bias.is_none() {
            self.bias = Some(Parameter::new(self.weight.new_zeros([self.out_features])));
            self.reset_bias();
        }
        self
    }

    /// Initializes the parameters again, like PyTorch: the weights with
    /// [`kaiming_uniform_`](init::kaiming_uniform_)`(a = √5)`, which samples from
    /// `U(-1/√in_features, 1/√in_features)`, and the bias from the same distribution (zeros
    /// for zero `in_features`).
    pub fn reset_parameters(&self) {
        init::kaiming_uniform_(
            &self.weight,
            5f64.sqrt(),
            FanMode::FanIn,
            Nonlinearity::LeakyRelu,
        );
        self.reset_bias();
    }

    fn reset_bias(&self) {
        if let Some(bias) = &self.bias {
            let (fan_in, _) = init::calculate_fan_in_and_fan_out(&self.weight);
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            init::uniform_(bias, -bound, bound);
        }
    }

    /// The output for `input`, whose last dimension has the input features.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::linear(input, &self.weight, self.bias.as_deref())
    }
}

impl Forward for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        Linear::forward(self, input)
    }
}

/// The sizes, like PyTorch's `repr`: `Linear { in_features: 4, out_features: 3, bias: true, .. }`.
impl fmt::Debug for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .finish_non_exhaustive()
    }
}
