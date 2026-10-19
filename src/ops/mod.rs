//! Backend-independent operator entry points.
//!
//! Every operator first checks its arguments and works out the output shape and dtype, then
//! runs a kernel. On the meta device only the first step runs, so meta tensors get correct shapes
//! and dtypes without any data.

use crate::backend::cpu;
use crate::dtype::Kind;
use crate::shape::compute_view_strides;
use crate::tensor::{Layout, TensorData};
use crate::{DType, Device, Scalar, Shape, Tensor};

mod advanced_index;
mod binary;
mod composite;
mod matmul;
mod overload;
mod random;
mod reduce;
mod scatter;
mod shape_ops;
mod unary;

pub(crate) use advanced_index::{META_VALUES, check_index_dtype, check_index_put_target};
pub use binary::where_;
pub(crate) use binary::{binary_op, copy_error};
pub use overload::Number;
pub use random::{
    manual_seed, rand, rand_like, rand_with, randint, randint_with, randn, randn_like, randn_with,
};
pub(crate) use reduce::exclusive_cumprod;
pub use shape_ops::{concat, stack};
pub use unary::GeluApproximate;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum UnaryOp {
    Neg,
    Abs,
    Sign,
    Relu,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Reciprocal,
    Sin,
    Cos,
    Erf,
    Tanh,
    Sigmoid,
    LogSigmoid,
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

/// A contiguous copy of `data` in new storage, like [`Tensor::deep_clone`] but for operations
/// to use internally: it is never recorded for autograd.
pub(crate) fn clone_data(data: &TensorData) -> TensorData {
    let out = Tensor::empty_on(data.layout.shape.clone(), data.dtype, data.device()).data();
    if data.device() == Device::Cpu {
        cpu::copy(&out, data);
    }
    out
}

/// `data` with `shape`, which has the same number of elements: the same storage when the layout
/// allows it, otherwise a contiguous copy. Like [`Tensor::reshape`] but for operations to use
/// internally: the result is not a tensor, so nothing is recorded as a view or for autograd.
pub(crate) fn reshape_data(data: &TensorData, shape: Shape) -> TensorData {
    debug_assert_eq!(shape.num_elements(), data.layout.num_elements());
    let l = &data.layout;
    match compute_view_strides(&l.shape, &l.strides, &shape) {
        Some(strides) => TensorData {
            layout: Layout {
                shape,
                strides,
                offset: l.offset,
            },
            ..data.clone()
        },
        None => TensorData {
            layout: Layout::contiguous(shape),
            ..clone_data(data)
        },
    }
}

/// `value` converted for writing into a tensor of `dtype` on `device`, as `fill_`, `full` and
/// `masked_fill` do: see [`Scalar::checked_to`]. Meta tensors take any value, like PyTorch's meta
/// kernels, which don't convert it.
///
/// # Panics
///
/// Panics if `value` overflows `dtype`, with PyTorch's message, also for tensors without
/// elements.
pub(crate) fn fill_value(value: Scalar, dtype: DType, device: Device) -> Scalar {
    if device == Device::Meta {
        return value;
    }
    value
        .checked_to(dtype)
        .unwrap_or_else(|| panic!("value cannot be converted to type {dtype} without overflow"))
}

/// The value of the single element of `t`, a CPU tensor, as a scalar of its kind: a bool, an
/// integer (exact for every integer dtype) or a float (exact for every floating point dtype).
pub(crate) fn scalar_value(t: &Tensor) -> Scalar {
    let t = t.detach();
    match t.dtype().kind() {
        Kind::Bool => Scalar::Bool(t.item()),
        Kind::Int => Scalar::Int(t.to(DType::I64).item()),
        Kind::Float => Scalar::Float(t.to(DType::F64).item()),
    }
}

/// A zero-dimensional tensor holding `value`.
pub(crate) fn scalar_data(value: Scalar, dtype: DType, device: Device) -> TensorData {
    let out = Tensor::empty_on(Shape::scalar(), dtype, device).data();
    if device == Device::Cpu {
        cpu::fill(&out, value);
    }
    out
}

impl Tensor {
    /// The flow that the in-place operations recorded for autograd share; `inputs` are their
    /// tensor inputs other than `self`.
    ///
    /// 1. [`check_in_place`](crate::autograd::check_in_place) rejects writes that autograd
    ///    doesn't allow and tells whether this one is recorded.
    /// 2. `compute` runs the out-of-place operation in no-grad mode, which checks the arguments.
    ///    It returns the result and whatever else `record` needs from the computation.
    /// 3. The result has to fit `self`, see [`check_result`](Tensor::check_result).
    /// 4. The result is written into `self` by [`write_in_place`](Tensor::write_in_place),
    ///    where `record` records the operation's node with `self` as its output and `original`
    ///    in the place of `self` as an input. With `saves_self`, which must hold whenever the
    ///    operation's formula saves that input, `original` is the value of `self` before the
    ///    write, with its history; otherwise it is `self`.
    ///
    /// Every check runs before the write, so a call that panics changes neither the data, the
    /// version nor the history of `self`. The version is bumped exactly once.
    pub(crate) fn in_place<C>(
        &self,
        inputs: &[Option<&Tensor>],
        saves_self: bool,
        compute: impl FnOnce() -> (Tensor, C),
        record: impl FnOnce(&Tensor, C),
    ) -> &Self {
        let recorded = crate::autograd::check_in_place(self, inputs);
        let (result, computed) = {
            let _no_grad = crate::no_grad();
            compute()
        };
        self.check_result(&result);
        let target = self.clone();
        self.write_in_place(
            recorded,
            move || target.write_copy(&result),
            || {
                let original = if saves_self {
                    crate::autograd::original_value(self)
                } else {
                    self.clone()
                };
                record(&original, computed);
            },
        );
        self
    }

    /// The last step of every in-place operation, after its checks: runs `write`, which must be
    /// unable to fail, on its own, or, if the write is `recorded` (as
    /// [`check_in_place`](crate::autograd::check_in_place) tells), inside the recording of the
    /// node that `record` records with `self` as its output (see
    /// [`record_in_place`](crate::autograd::record_in_place)).
    pub(crate) fn write_in_place(
        &self,
        recorded: bool,
        write: impl FnOnce() + 'static,
        record: impl FnOnce(),
    ) {
        if recorded {
            crate::autograd::record_in_place(self, Box::new(write), record);
        } else {
            write();
        }
    }

    /// Checks that the result of an in-place operation can be written into `self` with
    /// [`write_copy`](Tensor::write_copy), which then can't fail: it has the shape of `self`, its
    /// dtype may be cast to the dtype of `self`, and the elements of `self` are distinct memory
    /// locations. The result is on the device of `self`, which computed it.
    fn check_result(&self, result: &Tensor) {
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
        binary::check_writable(&dst);
        debug_assert_eq!(dst.device(), src.device());
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Tensor};

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
    fn each_in_place_op_bumps_the_version_exactly_once() {
        type InPlace = fn(&Tensor);
        let ops: [(&str, InPlace); 26] = [
            ("add_ scalar", |x| {
                x.add_(1.0);
            }),
            ("add_ tensor", |x| {
                x.add_(crate::ones_with([3], x.device()));
            }),
            ("sub_", |x| {
                x.sub_(1.0);
            }),
            ("mul_", |x| {
                x.mul_(crate::ones_with([1, 3], x.device()));
            }),
            ("div_", |x| {
                x.div_(2.0);
            }),
            ("pow_", |x| {
                x.pow_(2.0);
            }),
            ("clamp_ scalar", |x| {
                x.clamp_(0.0, 1.0);
            }),
            ("clamp_ tensor", |x| {
                x.clamp_(crate::zeros_with([3], x.device()), 1.0);
            }),
            ("masked_fill_", |x| {
                x.masked_fill_(&crate::ones_with([3], (DType::Bool, x.device())), 5.0);
            }),
            ("addcmul_", |x| {
                let t = crate::ones_with([3], x.device());
                x.addcmul_(&t, &t, 0.5);
            }),
            ("addcdiv_", |x| {
                let t = crate::ones_with([3], x.device());
                x.addcdiv_(&t, &t, 0.5);
            }),
            ("lerp_ scalar", |x| {
                x.lerp_(&crate::zeros_with([3], x.device()), 0.5);
            }),
            ("lerp_ tensor", |x| {
                let t = crate::full_with([3], 0.25, x.device());
                x.lerp_(&t, &t);
            }),
            ("copy_", |x| {
                x.copy_(&crate::zeros_with([1, 3], x.device()));
            }),
            ("copy_ broadcast", |x| {
                x.copy_(&crate::zeros_with([3], x.device()));
            }),
            ("copy_ other dtype", |x| {
                x.copy_(&crate::zeros_with([3], (DType::I64, x.device())));
            }),
            ("copy_ self", |x| {
                x.copy_(x);
            }),
            ("fill_", |x| {
                x.fill_(2.0);
            }),
            ("zero_", |x| {
                x.zero_();
            }),
            ("uniform_", |x| {
                x.uniform_(0.0, 1.0);
            }),
            ("normal_", |x| {
                x.normal_(0.0, 1.0);
            }),
            ("add_ through a transposed view", |x| {
                x.transpose(0, 1).add_(1.0);
            }),
            ("index_add_", |x| {
                let index = crate::zeros_with([1], (DType::I64, x.device()));
                x.index_add_(1, &index, &crate::ones_with([1, 1], x.device()));
            }),
            ("scatter_add_", |x| {
                let index = crate::zeros_with([1, 1], (DType::I64, x.device()));
                x.scatter_add_(1, &index, &crate::ones_with([1, 1], x.device()));
            }),
            ("index_put_", |x| {
                let index = crate::tensor([2i64, 0, 2]);
                x.index_put_((.., &index), &crate::ones_with([1], x.device()), false);
            }),
            ("index_put_ accumulate", |x| {
                let mask = crate::tensor([true, false, true]);
                x.index_put_((.., &mask), &crate::ones_with([2], x.device()), true);
            }),
        ];
        for device in [Device::Cpu, Device::Meta] {
            for (name, op) in ops {
                // Through the tensor itself and through a view of a larger one.
                let x = crate::ones_with([1, 3], device);
                let base = crate::ones_with([2, 3], device);
                let view = base.narrow(0, 1, 1);
                for (t, observed) in [(&x, &x), (&view, &base)] {
                    let start = observed.version();
                    op(t);
                    assert_eq!(observed.version(), start + 1, "{name} on {device}");
                }
            }
        }
    }

    #[test]
    fn copy_between_overlapping_views_bumps_the_version_once() {
        let x = crate::arange(6.0);
        let start = x.version();
        x.narrow(0, 1, 5).copy_(&x.narrow(0, 0, 5));
        assert_eq!(x.version(), start + 1);
        assert_eq!(x.to_vec::<f32>(), [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn new_tensors_start_at_version_zero() {
        assert_eq!(Tensor::from_slice(&[1.0f32], [1]).version(), 0);
    }
}
