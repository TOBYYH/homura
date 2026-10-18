use crate::{GeluApproximate, Tensor};

/// `max(x, 0)`, [`Tensor::relu`].
pub fn relu(input: &Tensor) -> Tensor {
    input.relu()
}

/// The Gaussian error linear unit, [`Tensor::gelu`]; PyTorch's default `approximate` is
/// [`GeluApproximate::None`].
pub fn gelu(input: &Tensor, approximate: GeluApproximate) -> Tensor {
    input.gelu(approximate)
}

/// `x * sigmoid(x)`, [`Tensor::silu`].
pub fn silu(input: &Tensor) -> Tensor {
    input.silu()
}

/// [`Tensor::tanh`].
pub fn tanh(input: &Tensor) -> Tensor {
    input.tanh()
}

/// [`Tensor::sigmoid`].
pub fn sigmoid(input: &Tensor) -> Tensor {
    input.sigmoid()
}

/// Softmax along `dim`, [`Tensor::softmax`].
pub fn softmax(input: &Tensor, dim: i64) -> Tensor {
    input.softmax(dim)
}

/// The logarithm of the softmax along `dim`, [`Tensor::log_softmax`].
pub fn log_softmax(input: &Tensor, dim: i64) -> Tensor {
    input.log_softmax(dim)
}
