use crate::{GeluApproximate, Tensor};

/// `max(x, 0)`, [`Tensor::relu`].
#[track_caller]
pub fn relu(input: &Tensor) -> Tensor {
    input.relu()
}

/// The Gaussian error linear unit, [`Tensor::gelu`]; PyTorch's default `approximate` is
/// [`GeluApproximate::None`].
#[track_caller]
pub fn gelu(input: &Tensor, approximate: GeluApproximate) -> Tensor {
    input.gelu(approximate)
}

/// `x * sigmoid(x)`, [`Tensor::silu`].
#[track_caller]
pub fn silu(input: &Tensor) -> Tensor {
    input.silu()
}

/// [`Tensor::tanh`].
#[track_caller]
pub fn tanh(input: &Tensor) -> Tensor {
    input.tanh()
}

/// [`Tensor::sigmoid`].
#[track_caller]
pub fn sigmoid(input: &Tensor) -> Tensor {
    input.sigmoid()
}

/// `log(sigmoid(x))`, [`Tensor::log_sigmoid`].
#[track_caller]
pub fn logsigmoid(input: &Tensor) -> Tensor {
    input.log_sigmoid()
}

/// Softmax along `dim`, [`Tensor::softmax`].
#[track_caller]
pub fn softmax(input: &Tensor, dim: i64) -> Tensor {
    input.softmax(dim)
}

/// The logarithm of the softmax along `dim`, [`Tensor::log_softmax`].
#[track_caller]
pub fn log_softmax(input: &Tensor, dim: i64) -> Tensor {
    input.log_softmax(dim)
}
