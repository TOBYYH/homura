//! Parameter initialization, like PyTorch's `torch.nn.init`.
//!
//! Each function fills a tensor in place and returns it. They run in no-grad mode, so they also
//! fill parameters and other leaves that require grad, and nothing is recorded. The formulas
//! for bounds and standard deviations are PyTorch's; the random numbers come from homura's
//! generator (see [`manual_seed`](crate::manual_seed)), so the values differ from PyTorch's.
//! On the meta device, filling does nothing.
//!
//! ```
//! use homura::nn::init::{self, FanMode, Nonlinearity};
//!
//! let w = homura::zeros([256, 128]);
//! init::kaiming_uniform_(&w, 0.0, FanMode::FanIn, Nonlinearity::Relu);
//! let bound = (6.0f64 / 128.0).sqrt();
//! assert!(w.abs().max().item::<f32>() as f64 <= bound);
//! ```

use crate::{Scalar, Tensor};

/// Which fan the Kaiming initializations scale by: `fan_in` keeps the variance of the
/// activations in the forward pass, `fan_out` that of the gradients in the backward pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FanMode {
    /// `mode="fan_in"`, the default.
    #[default]
    FanIn,
    /// `mode="fan_out"`.
    FanOut,
}

/// The nonlinearity named in [`calculate_gain`], with PyTorch's names (`"linear"`, `"conv2d"`,
/// `"relu"`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Nonlinearity {
    Linear,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Sigmoid,
    Tanh,
    Relu,
    /// Leaky ReLU, whose negative slope is the parameter of [`calculate_gain`].
    LeakyRelu,
    Selu,
}

/// The recommended gain for `nonlinearity`: 1 for the linear functions (`Linear`, the
/// convolutions) and `Sigmoid`, 5/3 for `Tanh`, √2 for `Relu`, `√(2 / (1 + slope²))` for
/// `LeakyRelu` with `param` as the negative slope (0.01 if `None`) and 3/4 for `Selu`. `param` is
/// ignored for the other nonlinearities.
///
/// ```
/// use homura::nn::init::{Nonlinearity, calculate_gain};
///
/// assert_eq!(calculate_gain(Nonlinearity::Relu, None), 2f64.sqrt());
/// assert_eq!(calculate_gain(Nonlinearity::LeakyRelu, Some(1.0)), 1.0);
/// ```
pub fn calculate_gain(nonlinearity: Nonlinearity, param: Option<f64>) -> f64 {
    use Nonlinearity::*;
    match nonlinearity {
        Linear | Conv1d | Conv2d | Conv3d | ConvTranspose1d | ConvTranspose2d | ConvTranspose3d
        | Sigmoid => 1.0,
        Tanh => 5.0 / 3.0,
        Relu => 2f64.sqrt(),
        LeakyRelu => {
            let slope = param.unwrap_or(0.01);
            (2.0 / (1.0 + slope.powi(2))).sqrt()
        }
        Selu => 3.0 / 4.0,
    }
}

/// The fans of a weight of at least two dimensions in PyTorch's layout `[out, in, *kernel]`:
/// `(in × kernel elements, out × kernel elements)`. PyTorch's private
/// `_calculate_fan_in_and_fan_out`, which layers use to initialize their bias.
///
/// # Panics
///
/// Panics if `tensor` has fewer than two dimensions.
pub fn calculate_fan_in_and_fan_out(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor.shape();
    if shape.len() < 2 {
        panic!("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
    }
    let receptive_field: usize = shape[2..].iter().product();
    (shape[1] * receptive_field, shape[0] * receptive_field)
}

/// Fills `tensor` with samples from the uniform distribution on `[a, b)` (PyTorch's defaults
/// are 0 and 1).
///
/// # Panics
///
/// Panics like [`Tensor::uniform_`]: for integer and bool tensors, and for invalid bounds.
pub fn uniform_(tensor: &Tensor, a: f64, b: f64) -> &Tensor {
    let _no_grad = crate::no_grad();
    tensor.uniform_(a, b)
}

/// Fills `tensor` with samples from the normal distribution with `mean` and `std` (PyTorch's
/// defaults are 0 and 1).
///
/// # Panics
///
/// Panics like [`Tensor::normal_`]: for integer and bool tensors, and for a negative `std`.
pub fn normal_(tensor: &Tensor, mean: f64, std: f64) -> &Tensor {
    let _no_grad = crate::no_grad();
    tensor.normal_(mean, std)
}

/// Fills `tensor` with `value`, converted like [`Tensor::fill_`] does.
///
/// # Panics
///
/// Panics if `value` overflows the dtype of `tensor`, as [`Tensor::fill_`] describes.
pub fn constant_(tensor: &Tensor, value: impl Into<Scalar>) -> &Tensor {
    let _no_grad = crate::no_grad();
    tensor.fill_(value)
}

/// Fills `tensor` with ones.
pub fn ones_(tensor: &Tensor) -> &Tensor {
    constant_(tensor, 1)
}

/// Fills `tensor` with zeros.
pub fn zeros_(tensor: &Tensor) -> &Tensor {
    let _no_grad = crate::no_grad();
    tensor.zero_()
}

/// The standard deviation `gain / √fan` of the Kaiming initializations.
fn kaiming_std(tensor: &Tensor, a: f64, mode: FanMode, nonlinearity: Nonlinearity) -> f64 {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor);
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    calculate_gain(nonlinearity, Some(a)) / (fan as f64).sqrt()
}

/// Fills `tensor` with samples from `U(-bound, bound)` with `bound = gain × √(3 / fan)` (He
/// initialization), where the gain is [`calculate_gain`]`(nonlinearity, Some(a))` and the fan is
/// selected by `mode` from [`calculate_fan_in_and_fan_out`]. PyTorch's defaults are `a = 0`,
/// [`FanMode::FanIn`] and [`Nonlinearity::LeakyRelu`].
///
/// A tensor without elements is returned unchanged, whatever its shape (PyTorch also warns).
///
/// # Panics
///
/// Panics if `tensor` has elements and fewer than two dimensions, or isn't floating point.
pub fn kaiming_uniform_(
    tensor: &Tensor,
    a: f64,
    mode: FanMode,
    nonlinearity: Nonlinearity,
) -> &Tensor {
    if tensor.num_elements() == 0 {
        return tensor;
    }
    // PyTorch takes the bound from the standard deviation in two steps; so does this, so that
    // the bounds round alike.
    let bound = 3f64.sqrt() * kaiming_std(tensor, a, mode, nonlinearity);
    uniform_(tensor, -bound, bound)
}

/// Fills `tensor` with samples from `N(0, std²)` with `std = gain / √fan` (He initialization);
/// the arguments are those of [`kaiming_uniform_`].
///
/// A tensor without elements is returned unchanged, whatever its shape (PyTorch also warns).
///
/// # Panics
///
/// Panics if `tensor` has elements and fewer than two dimensions, or isn't floating point.
pub fn kaiming_normal_(
    tensor: &Tensor,
    a: f64,
    mode: FanMode,
    nonlinearity: Nonlinearity,
) -> &Tensor {
    if tensor.num_elements() == 0 {
        return tensor;
    }
    let std = kaiming_std(tensor, a, mode, nonlinearity);
    normal_(tensor, 0.0, std)
}

/// The standard deviation `gain × √(2 / (fan_in + fan_out))` of the Xavier initializations.
fn xavier_std(tensor: &Tensor, gain: f64) -> f64 {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor);
    if fan_in + fan_out == 0 {
        // PyTorch's Python division raises ZeroDivisionError here.
        panic!(
            "division by zero: fan_in + fan_out is 0 for a tensor of shape {:?}",
            tensor.shape()
        );
    }
    gain * (2.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Fills `tensor` with samples from `U(-a, a)` with `a = gain × √(6 / (fan_in + fan_out))`
/// (Glorot initialization). PyTorch's default `gain` is 1.
///
/// # Panics
///
/// Panics if `tensor` has fewer than two dimensions or both fans are zero, also without elements,
/// if it isn't floating point, or for a negative or infinite `gain`, like [`uniform_`].
pub fn xavier_uniform_(tensor: &Tensor, gain: f64) -> &Tensor {
    let a = 3f64.sqrt() * xavier_std(tensor, gain);
    uniform_(tensor, -a, a)
}

/// Fills `tensor` with samples from `N(0, std²)` with `std = gain × √(2 / (fan_in + fan_out))`
/// (Glorot initialization). PyTorch's default `gain` is 1.
///
/// # Panics
///
/// Panics if `tensor` has fewer than two dimensions or both fans are zero, also without elements,
/// if it isn't floating point, or for a negative or NaN `gain`, like [`normal_`] (an infinite
/// `gain` gives an infinite standard deviation, which [`normal_`] takes).
pub fn xavier_normal_(tensor: &Tensor, gain: f64) -> &Tensor {
    let std = xavier_std(tensor, gain);
    normal_(tensor, 0.0, std)
}
