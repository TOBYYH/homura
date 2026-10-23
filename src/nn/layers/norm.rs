use std::fmt;

use super::Forward;
use crate::nn::{Module, Parameter, functional, init};
use crate::{IntoShape, Tensor};

/// Layer normalization over the last dimensions `normalized_shape`, PyTorch's `nn.LayerNorm`,
/// with [`functional::layer_norm`].
///
/// By default it has a `weight` (ones) and a `bias` (zeros) of the shape `normalized_shape`, with
/// the default dtype and device, and `eps = 1e-5`, like PyTorch.
///
/// ```
/// use homura::nn::{LayerNorm, Module};
///
/// let norm = LayerNorm::new([4]);
/// let y = norm.forward(&homura::randn([2, 3, 4]));
/// assert_eq!(y.shape(), [2, 3, 4]);
/// assert_eq!(norm.state_dict().keys().collect::<Vec<_>>(), ["weight", "bias"]);
///
/// let plain = LayerNorm::new([4]).elementwise_affine(false).eps(1e-6);
/// assert!(plain.weight.is_none() && plain.bias.is_none());
/// ```
#[derive(Module)]
#[non_exhaustive]
pub struct LayerNorm {
    /// The scale, of the shape `normalized_shape`, with elementwise affine.
    pub weight: Option<Parameter>,
    /// The shift, of the shape `normalized_shape`, with elementwise affine and a bias.
    pub bias: Option<Parameter>,
    pub normalized_shape: Vec<usize>,
    pub eps: f64,
}

impl LayerNorm {
    /// A layer that normalizes the last dimensions `normalized_shape`
    /// (`nn.LayerNorm(normalized_shape)`), with a weight and a bias. Like PyTorch, an empty
    /// shape is accepted here, and the forward computation rejects it.
    #[track_caller]
    pub fn new(normalized_shape: impl IntoShape) -> LayerNorm {
        let normalized_shape = normalized_shape.into_shape().to_vec();
        LayerNorm {
            weight: Some(Parameter::new(crate::ones(&normalized_shape))),
            bias: Some(Parameter::new(crate::zeros(&normalized_shape))),
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// This layer with `eps`, which is added to the variance (`eps=...`).
    #[must_use]
    #[track_caller]
    pub fn eps(mut self, eps: f64) -> LayerNorm {
        self.eps = eps;
        self
    }

    /// This layer with a weight and a bias if `elementwise_affine` is true (the default), and
    /// with neither otherwise (`elementwise_affine=False`). New parameters are initialized like
    /// [`reset_parameters`](LayerNorm::reset_parameters) does, with the dtype and device of the
    /// parameter the layer has (the default ones if it has none); a parameter it has stays.
    #[must_use]
    #[track_caller]
    pub fn elementwise_affine(mut self, elementwise_affine: bool) -> LayerNorm {
        if !elementwise_affine {
            self.weight = None;
            self.bias = None;
        } else {
            let like = self
                .weight
                .as_ref()
                .or(self.bias.as_ref())
                .map(|p| (**p).clone());
            let shape = &self.normalized_shape;
            self.weight
                .get_or_insert_with(|| new_parameter(like.as_ref(), shape, 1.0));
            self.bias
                .get_or_insert_with(|| new_parameter(like.as_ref(), shape, 0.0));
        }
        self
    }

    /// This layer without a bias if `bias` is false (`bias=False`). With true (the default), a
    /// layer with elementwise affine gets a bias of zeros if it has none, with the dtype and
    /// device of its weight; a layer without elementwise affine stays without one, as in PyTorch.
    #[must_use]
    #[track_caller]
    pub fn bias(mut self, bias: bool) -> LayerNorm {
        if !bias {
            self.bias = None;
        } else if let Some(w) = &self.weight {
            let shape = &self.normalized_shape;
            self.bias
                .get_or_insert_with(|| new_parameter(Some(w), shape, 0.0));
        }
        self
    }

    /// Initializes the parameters again, like PyTorch: the weight with ones and the bias with
    /// zeros.
    #[track_caller]
    pub fn reset_parameters(&self) {
        if let Some(w) = &self.weight {
            init::ones_(w);
        }
        if let Some(b) = &self.bias {
            init::zeros_(b);
        }
    }

    /// The normalization of `input`, whose last dimensions are `normalized_shape`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::layer_norm(
            input,
            &self.normalized_shape,
            self.weight.as_deref(),
            self.bias.as_deref(),
            self.eps,
        )
    }
}

/// A parameter of `shape` filled with `value`, with the dtype and device of `like`, or the
/// default ones without it.
fn new_parameter(like: Option<&Tensor>, shape: &[usize], value: f64) -> Parameter {
    Parameter::new(match like {
        Some(t) => t.new_full(shape, value),
        None => crate::full(shape, value),
    })
}

impl Forward for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        LayerNorm::forward(self, input)
    }
}

/// The options, like PyTorch's `repr`:
/// `LayerNorm { normalized_shape: [4], eps: 1e-5, elementwise_affine: true, bias: true, .. }`.
impl fmt::Debug for LayerNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerNorm")
            .field("normalized_shape", &self.normalized_shape)
            .field("eps", &self.eps)
            .field("elementwise_affine", &self.weight.is_some())
            .field("bias", &self.bias.is_some())
            .finish_non_exhaustive()
    }
}

/// Root mean square normalization over the last dimensions `normalized_shape`, PyTorch's
/// `nn.RMSNorm`, with [`functional::rms_norm`].
///
/// By default it has a `weight` (ones) of the shape `normalized_shape`, with the default dtype
/// and device, and no `eps`: [`functional::rms_norm`] then uses the machine epsilon, like
/// PyTorch's `eps=None`.
///
/// ```
/// use homura::nn::{Module, RmsNorm};
///
/// let norm = RmsNorm::new([8]).eps(1e-6);
/// assert_eq!(norm.forward(&homura::randn([2, 8])).shape(), [2, 8]);
/// assert_eq!(norm.state_dict().keys().collect::<Vec<_>>(), ["weight"]);
/// ```
#[derive(Module)]
#[non_exhaustive]
pub struct RmsNorm {
    /// The scale, of the shape `normalized_shape`, with elementwise affine.
    pub weight: Option<Parameter>,
    pub normalized_shape: Vec<usize>,
    pub eps: Option<f64>,
}

impl RmsNorm {
    /// A layer that normalizes the last dimensions `normalized_shape`
    /// (`nn.RMSNorm(normalized_shape)`), with a weight.
    #[track_caller]
    pub fn new(normalized_shape: impl IntoShape) -> RmsNorm {
        let normalized_shape = normalized_shape.into_shape().to_vec();
        RmsNorm {
            weight: Some(new_parameter(None, &normalized_shape, 1.0)),
            normalized_shape,
            eps: None,
        }
    }

    /// This layer with `eps` (`eps=...`): a number, or `None` for the machine epsilon.
    #[must_use]
    #[track_caller]
    pub fn eps(mut self, eps: impl Into<Option<f64>>) -> RmsNorm {
        self.eps = eps.into();
        self
    }

    /// This layer with a weight if `elementwise_affine` is true (the default), and without one
    /// otherwise (`elementwise_affine=False`). A new weight is initialized with ones, with the
    /// default dtype and device: the layer has no other parameter to take them from.
    #[must_use]
    #[track_caller]
    pub fn elementwise_affine(mut self, elementwise_affine: bool) -> RmsNorm {
        if !elementwise_affine {
            self.weight = None;
        } else {
            let shape = &self.normalized_shape;
            self.weight
                .get_or_insert_with(|| new_parameter(None, shape, 1.0));
        }
        self
    }

    /// Initializes the weight again with ones, like PyTorch.
    #[track_caller]
    pub fn reset_parameters(&self) {
        if let Some(w) = &self.weight {
            init::ones_(w);
        }
    }

    /// The normalization of `input`, whose last dimensions are `normalized_shape`.
    #[track_caller]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        functional::rms_norm(
            input,
            &self.normalized_shape,
            self.weight.as_deref(),
            self.eps,
        )
    }
}

impl Forward for RmsNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        RmsNorm::forward(self, input)
    }
}

/// The options, like PyTorch's `repr`:
/// `RmsNorm { normalized_shape: [8], eps: Some(1e-6), elementwise_affine: true, .. }`.
impl fmt::Debug for RmsNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RmsNorm")
            .field("normalized_shape", &self.normalized_shape)
            .field("eps", &self.eps)
            .field("elementwise_affine", &self.weight.is_some())
            .finish_non_exhaustive()
    }
}
