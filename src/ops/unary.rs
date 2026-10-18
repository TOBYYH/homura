//! Elementwise unary operations.

use super::{UnaryOp, cast_data};
use crate::backend::cpu;
use crate::{DType, Device, Tensor};

/// Which formula [`Tensor::gelu`] uses (PyTorch's `approximate` argument).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GeluApproximate {
    /// The exact form `x * Φ(x)`, using the error function.
    #[default]
    None,
    /// The tanh approximation.
    Tanh,
}

fn unary_op(op: UnaryOp, x: &Tensor) -> Tensor {
    use UnaryOp::*;
    let src = x.data();
    let dtype = src.dtype;
    let compute = match op {
        LogicalNot => dtype,
        Neg if dtype == DType::Bool => panic!(
            "Negation, the `-` operator, on a bool tensor is not supported. If you are trying to \
             invert a mask, use `logical_not()` instead."
        ),
        Neg | Abs | Relu => {
            if dtype == DType::Bool {
                panic!("{op:?} is not supported for bool tensors");
            }
            dtype
        }
        // Like PyTorch, the sign keeps integer and bool dtypes.
        Sign => dtype,
        Silu | Gelu | GeluTanh | LogSigmoid => {
            if !dtype.is_floating_point() {
                panic!("{op:?} is not implemented for {dtype} tensors");
            }
            dtype
        }
        // Math functions promote integer inputs to the default float dtype.
        _ if dtype.is_floating_point() => dtype,
        _ => crate::default_dtype(),
    };
    let out_dtype = if op == LogicalNot {
        DType::Bool
    } else {
        compute
    };
    let out = Tensor::empty_on(src.layout.shape.clone(), out_dtype, src.device());
    if src.device() == Device::Cpu {
        cpu::unary(op, &out.data(), &cast_data(&src, compute));
    }
    crate::autograd::formulas::unary(op, x, &out);
    out
}

impl Tensor {
    /// `-self`.
    pub fn neg(&self) -> Tensor {
        unary_op(UnaryOp::Neg, self)
    }

    /// Absolute value.
    pub fn abs(&self) -> Tensor {
        unary_op(UnaryOp::Abs, self)
    }

    /// The sign of each element: `-1`, `0` or `1` in the dtype of `self`. As in PyTorch, NaN and
    /// `-0.0` give `0.0`, and bool tensors are copied unchanged.
    pub fn sign(&self) -> Tensor {
        unary_op(UnaryOp::Sign, self)
    }

    /// `e^x`.
    pub fn exp(&self) -> Tensor {
        unary_op(UnaryOp::Exp, self)
    }

    /// Natural logarithm.
    pub fn log(&self) -> Tensor {
        unary_op(UnaryOp::Log, self)
    }

    /// Square root.
    pub fn sqrt(&self) -> Tensor {
        unary_op(UnaryOp::Sqrt, self)
    }

    /// `1 / sqrt(x)`.
    pub fn rsqrt(&self) -> Tensor {
        unary_op(UnaryOp::Rsqrt, self)
    }

    /// `1 / x`.
    pub fn reciprocal(&self) -> Tensor {
        unary_op(UnaryOp::Reciprocal, self)
    }

    /// Sine.
    pub fn sin(&self) -> Tensor {
        unary_op(UnaryOp::Sin, self)
    }

    /// Cosine.
    pub fn cos(&self) -> Tensor {
        unary_op(UnaryOp::Cos, self)
    }

    /// The error function `erf(x) = 2/√π ∫₀ˣ e^(−t²) dt`.
    pub fn erf(&self) -> Tensor {
        unary_op(UnaryOp::Erf, self)
    }

    /// Hyperbolic tangent.
    pub fn tanh(&self) -> Tensor {
        unary_op(UnaryOp::Tanh, self)
    }

    /// `1 / (1 + e^-x)`.
    pub fn sigmoid(&self) -> Tensor {
        unary_op(UnaryOp::Sigmoid, self)
    }

    /// `log(sigmoid(x))`, computed stably as `min(x, 0) - log(1 + e^-|x|)` like PyTorch's CPU
    /// kernel (`F.logsigmoid`), so that it is accurate for large `|x|`: the result for `x = 40`
    /// is `-4.25e-18`, not zero.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not floating point, as in PyTorch.
    pub fn log_sigmoid(&self) -> Tensor {
        unary_op(UnaryOp::LogSigmoid, self)
    }

    /// `max(x, 0)`.
    pub fn relu(&self) -> Tensor {
        unary_op(UnaryOp::Relu, self)
    }

    /// `x * sigmoid(x)` (also known as swish).
    pub fn silu(&self) -> Tensor {
        unary_op(UnaryOp::Silu, self)
    }

    /// The Gaussian error linear unit, exact or with the tanh approximation.
    pub fn gelu(&self, approximate: GeluApproximate) -> Tensor {
        match approximate {
            GeluApproximate::None => unary_op(UnaryOp::Gelu, self),
            GeluApproximate::Tanh => unary_op(UnaryOp::GeluTanh, self),
        }
    }

    /// Elementwise logical not as a `Bool` tensor; nonzero values count as true.
    pub fn logical_not(&self) -> Tensor {
        unary_op(UnaryOp::LogicalNot, self)
    }
}
