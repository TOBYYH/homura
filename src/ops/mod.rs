//! Backend-independent operator entry points.
//!
//! Every operator first checks its arguments and works out the output shape and dtype, then
//! runs a kernel. On the meta device only the first step runs, so meta tensors get correct shapes
//! and dtypes without any data.

use crate::backend::cpu;
use crate::tensor::TensorData;
use crate::{DType, Device, Scalar, Tensor};

mod binary;
mod composite;
mod matmul;
mod overload;
mod random;
mod reduce;
mod shape_ops;
mod unary;

pub use binary::where_;
pub use overload::Number;
pub use random::{
    manual_seed, rand, rand_like, rand_with, randint, randint_with, randn, randn_like, randn_with,
};
pub use shape_ops::{concat, stack};
pub use unary::GeluApproximate;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum UnaryOp {
    Neg,
    Abs,
    Relu,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Reciprocal,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Silu,
    Gelu,
    GeluTanh,
    LogicalNot,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ReduceOp {
    Sum,
    Prod,
    Mean,
    Max,
    Min,
    All,
    Any,
}

/// A tensor or a scalar, accepted by operations such as [`Tensor::add`] that take either (PyTorch's
/// "Tensor or Number"). Created implicitly from `&Tensor`, `Tensor` and Rust numbers.
#[derive(Clone, Debug)]
pub enum Operand {
    /// A tensor operand.
    Tensor(Tensor),
    /// A scalar operand. It takes part in type promotion with the lowest priority.
    Scalar(Scalar),
}

impl From<&Tensor> for Operand {
    fn from(t: &Tensor) -> Self {
        Operand::Tensor(t.clone())
    }
}

impl From<Tensor> for Operand {
    fn from(t: Tensor) -> Self {
        Operand::Tensor(t)
    }
}

impl From<Scalar> for Operand {
    fn from(s: Scalar) -> Self {
        Operand::Scalar(s)
    }
}

macro_rules! operand_from_scalar {
    ($($t:ty),*) => {$(
        impl From<$t> for Operand {
            fn from(v: $t) -> Self {
                Operand::Scalar(v.into())
            }
        }
    )*};
}

operand_from_scalar!(
    bool,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    i64,
    isize,
    usize,
    f32,
    f64,
    half::f16,
    half::bf16
);

/// The device shared by all `tensors`.
///
/// # Panics
///
/// Panics if the tensors are on different devices.
pub(crate) fn common_device<'a>(tensors: impl IntoIterator<Item = &'a TensorData>) -> Device {
    let mut device = None;
    for t in tensors {
        match device {
            None => device = Some(t.device()),
            Some(d) if d != t.device() => panic!(
                "Expected all tensors to be on the same device, but found at least two devices, \
                 {d} and {}!",
                t.device()
            ),
            _ => {}
        }
    }
    device.expect("expected at least one tensor")
}

/// `data` converted to `dtype` (a copy only when the dtype differs).
pub(crate) fn cast_data(data: &TensorData, dtype: DType) -> TensorData {
    if data.dtype == dtype {
        return data.clone();
    }
    let out = Tensor::empty_on(data.layout.shape.clone(), dtype, data.device()).data();
    if data.device() == Device::Cpu {
        cpu::copy(&out, data);
    }
    out
}

/// A zero-dimensional tensor holding `value`.
pub(crate) fn scalar_data(value: Scalar, dtype: DType, device: Device) -> TensorData {
    let out = Tensor::empty_on(crate::Shape::scalar(), dtype, device).data();
    if device == Device::Cpu {
        cpu::fill(&out, value);
    }
    out
}

impl Tensor {
    /// Writes the result of an in-place operation into `self`, after checking that it has the
    /// shape of `self` and that its dtype may be cast to the dtype of `self`.
    pub(crate) fn assign_result(&self, result: &Tensor) {
        let (dst, src) = (self.data(), result.data());
        if dst.layout.shape != src.layout.shape {
            panic!(
                "output with shape {:?} doesn't match the broadcast shape {:?}",
                dst.layout.shape, src.layout.shape
            );
        }
        if !crate::can_cast(src.dtype, dst.dtype) {
            panic!(
                "result type {} can't be cast to the desired output type {}",
                src.dtype, dst.dtype
            );
        }
        self.copy_(result);
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[test]
    fn in_place_ops_bump_the_shared_version() {
        let x = crate::zeros([2, 3]);
        let view = x.narrow(1, 0, 2);
        let mask = crate::tensor([true, false, true]);
        let start = x.version();
        view.add_(1.0);
        x.fill_(2.0);
        x.zero_();
        x.copy_(&crate::ones([3]));
        x.clamp_(0.0, 1.0);
        x.masked_fill_(&mask, 5.0);
        x.lerp_(&crate::ones([2, 3]), 0.5);
        x.uniform_(0.0, 1.0);
        assert_eq!(x.version(), start + 8);
        assert_eq!(view.version(), x.version());
        // Out-of-place operations and views don't count.
        let _ = (&x + 1.0, x.transpose(0, 1), x.deep_clone());
        assert_eq!(x.version(), start + 8);
        let meta = crate::zeros_with([2], Device::Meta);
        meta.add_(1.0);
        assert_eq!(meta.version(), 1);
    }

    #[test]
    fn new_tensors_start_at_version_zero() {
        assert_eq!(Tensor::from_slice(&[1.0f32], [1]).version(), 0);
    }
}
