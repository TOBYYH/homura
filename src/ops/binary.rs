//! Elementwise binary and ternary operations, and in-place writes.

use super::{BinaryOp, Operand, cast_data, clone_data, common_device, scalar_data};
use crate::backend::cpu;
use crate::dtype::ResultType;
use crate::shape::broadcast_shapes;
use crate::tensor::TensorData;
use crate::{DType, Device, Scalar, Shape, Tensor};

/// An operand resolved to its data, or a scalar.
enum Resolved {
    Tensor(TensorData),
    Scalar(Scalar),
}

impl Resolved {
    fn new(o: Operand) -> Resolved {
        match o {
            Operand::Tensor(t) => Resolved::Tensor(t.data()),
            Operand::Scalar(s) => Resolved::Scalar(s),
        }
    }

    fn add_to(&self, rt: &mut ResultType) {
        match self {
            Resolved::Tensor(t) => rt.tensor(t.dtype, t.layout.shape.len()),
            Resolved::Scalar(s) => rt.scalar(*s),
        };
    }

    fn shape(&self) -> Shape {
        match self {
            Resolved::Tensor(t) => t.layout.shape.clone(),
            Resolved::Scalar(_) => Shape::scalar(),
        }
    }

    fn tensor(&self) -> Option<&TensorData> {
        match self {
            Resolved::Tensor(t) => Some(t),
            Resolved::Scalar(_) => None,
        }
    }

    /// The operand as data of `dtype`, broadcast to `shape`.
    fn materialize(&self, dtype: DType, device: Device, shape: &Shape) -> TensorData {
        match self {
            Resolved::Tensor(t) => cast_data(t, dtype).expand_to(shape),
            Resolved::Scalar(s) => scalar_data(*s, dtype, device).expand_to(shape),
        }
    }
}

fn device_of(operands: &[&Resolved]) -> Device {
    common_device(operands.iter().filter_map(|o| o.tensor()))
}

/// `op(lhs, rhs)` with broadcasting and type promotion. At least one operand must be a tensor.
pub(crate) fn binary_op(op: BinaryOp, lhs: Operand, rhs: Operand) -> Tensor {
    use BinaryOp::*;
    let (a, b) = (Resolved::new(lhs), Resolved::new(rhs));
    let device = device_of(&[&a, &b]);
    let mut rt = ResultType::new();
    a.add_to(&mut rt);
    b.add_to(&mut rt);
    let common = rt.result();
    let compute = match op {
        // True division: integers are divided as floats.
        Div if !common.is_floating_point() => crate::default_dtype(),
        LogicalAnd | LogicalOr => DType::Bool,
        _ => common,
    };
    if compute == DType::Bool && op == Sub {
        panic!(
            "Subtraction, the `-` operator, with two bool tensors is not supported. Use \
             `logical_xor()` instead."
        );
    }
    if compute == DType::Bool && op == Pow {
        panic!("pow is not supported for bool tensors");
    }
    if compute.is_integral()
        && op == Pow
        && let Resolved::Scalar(s) = b
        && s.to_i64() < 0
    {
        panic!("Integers to negative integer powers are not allowed.");
    }
    let out_dtype = match op {
        Greater | GreaterEqual | Less | LessEqual | Equal | NotEqual | LogicalAnd | LogicalOr => {
            DType::Bool
        }
        _ => compute,
    };
    let shape = broadcast_shapes(&a.shape(), &b.shape());
    let out = Tensor::empty_on(shape.clone(), out_dtype, device);
    if device == Device::Cpu {
        let ad = a.materialize(compute, device, &shape);
        let bd = b.materialize(compute, device, &shape);
        cpu::binary(op, &out.data(), &ad, &bd);
    }
    out
}

/// Selects elements from `x` where `condition` is true and from `y` elsewhere (`torch.where`).
/// All three are broadcast together; `x` and `y` may be scalars.
///
/// # Panics
///
/// Panics if `condition` is not a `Bool` tensor.
pub fn where_(condition: &Tensor, x: impl Into<Operand>, y: impl Into<Operand>) -> Tensor {
    let cond = condition.data();
    if cond.dtype != DType::Bool {
        panic!(
            "where expected condition to be a boolean tensor, but got a tensor with dtype {}",
            cond.dtype
        );
    }
    let (x, y) = (Resolved::new(x.into()), Resolved::new(y.into()));
    let mut devices = vec![&cond];
    devices.extend(x.tensor());
    devices.extend(y.tensor());
    let device = common_device(devices);
    let mut rt = ResultType::new();
    x.add_to(&mut rt);
    y.add_to(&mut rt);
    let dtype = rt.result();
    let shape = broadcast_shapes(
        &broadcast_shapes(&cond.layout.shape, &x.shape()),
        &y.shape(),
    );
    let out = Tensor::empty_on(shape.clone(), dtype, device);
    if device == Device::Cpu {
        cpu::where_(
            &out.data(),
            &cond.expand_to(&shape),
            &x.materialize(dtype, device, &shape),
            &y.materialize(dtype, device, &shape),
        );
    }
    out
}

impl Tensor {
    fn binary(&self, op: BinaryOp, other: impl Into<Operand>) -> Tensor {
        binary_op(op, Operand::Tensor(self.clone()), other.into())
    }

    fn binary_(&self, op: BinaryOp, other: impl Into<Operand>) -> &Self {
        let result = self.binary(op, other);
        self.assign_result(&result);
        self
    }

    /// `self + other`, with broadcasting and type promotion.
    pub fn add(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Add, other)
    }

    /// `self - other`.
    pub fn sub(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Sub, other)
    }

    /// `self * other`.
    pub fn mul(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Mul, other)
    }

    /// `self / other`. Always true division: integer inputs give the default float dtype.
    pub fn div(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Div, other)
    }

    /// `self` raised to the power `exponent` (a tensor or a scalar).
    pub fn pow(&self, exponent: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Pow, exponent)
    }

    /// Elementwise maximum. NaN propagates.
    pub fn maximum(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Maximum, other)
    }

    /// Elementwise minimum. NaN propagates.
    pub fn minimum(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Minimum, other)
    }

    /// `self > other` as a `Bool` tensor.
    pub fn greater(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Greater, other)
    }

    /// `self >= other` as a `Bool` tensor.
    pub fn greater_equal(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::GreaterEqual, other)
    }

    /// `self < other` as a `Bool` tensor.
    pub fn less(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Less, other)
    }

    /// `self <= other` as a `Bool` tensor.
    pub fn less_equal(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::LessEqual, other)
    }

    /// `self == other` elementwise, as a `Bool` tensor. Use `x.equal(&y).all()` to compare whole
    /// tensors.
    pub fn equal(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::Equal, other)
    }

    /// `self != other` elementwise, as a `Bool` tensor.
    pub fn not_equal(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::NotEqual, other)
    }

    /// Elementwise logical and; nonzero values count as true.
    pub fn logical_and(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::LogicalAnd, other)
    }

    /// Elementwise logical or; nonzero values count as true.
    pub fn logical_or(&self, other: impl Into<Operand>) -> Tensor {
        self.binary(BinaryOp::LogicalOr, other)
    }

    /// In-place [`add`](Tensor::add).
    pub fn add_(&self, other: impl Into<Operand>) -> &Self {
        self.binary_(BinaryOp::Add, other)
    }

    /// In-place [`sub`](Tensor::sub).
    pub fn sub_(&self, other: impl Into<Operand>) -> &Self {
        self.binary_(BinaryOp::Sub, other)
    }

    /// In-place [`mul`](Tensor::mul).
    pub fn mul_(&self, other: impl Into<Operand>) -> &Self {
        self.binary_(BinaryOp::Mul, other)
    }

    /// In-place [`div`](Tensor::div).
    pub fn div_(&self, other: impl Into<Operand>) -> &Self {
        self.binary_(BinaryOp::Div, other)
    }

    /// In-place [`pow`](Tensor::pow).
    pub fn pow_(&self, exponent: impl Into<Operand>) -> &Self {
        self.binary_(BinaryOp::Pow, exponent)
    }

    /// Clamps every element into `[min, max]`; the bounds may be tensors or scalars. Equivalent
    /// to `self.maximum(min).minimum(max)`, so NaN propagates and `min > max` gives `max`.
    pub fn clamp(&self, min: impl Into<Operand>, max: impl Into<Operand>) -> Tensor {
        self.maximum(min).minimum(max)
    }

    /// Clamps every element to at least `min`.
    pub fn clamp_min(&self, min: impl Into<Operand>) -> Tensor {
        self.maximum(min)
    }

    /// Clamps every element to at most `max`.
    pub fn clamp_max(&self, max: impl Into<Operand>) -> Tensor {
        self.minimum(max)
    }

    /// In-place [`clamp`](Tensor::clamp).
    pub fn clamp_(&self, min: impl Into<Operand>, max: impl Into<Operand>) -> &Self {
        let result = self.clamp(min, max);
        self.assign_result(&result);
        self
    }

    /// A copy of `self` with `value` written where `mask` is true. `value` must be a scalar or a
    /// zero-dimensional tensor; the result keeps the dtype of `self`.
    pub fn masked_fill(&self, mask: &Tensor, value: impl Into<Operand>) -> Tensor {
        let m = mask.dtype();
        if m != DType::Bool {
            panic!("masked_fill only supports boolean masks, but got mask with dtype {m}");
        }
        let value = match value.into() {
            Operand::Scalar(s) => Tensor::from_data(scalar_data(s, self.dtype(), self.device())),
            Operand::Tensor(t) => {
                if t.num_dims() != 0 {
                    panic!(
                        "masked_fill only supports a 0-dimensional value tensor, but got tensor \
                         with {} dimension(s).",
                        t.num_dims()
                    );
                }
                t.to(self.dtype())
            }
        };
        where_(mask, &value, self)
    }

    /// In-place [`masked_fill`](Tensor::masked_fill).
    pub fn masked_fill_(&self, mask: &Tensor, value: impl Into<Operand>) -> &Self {
        let result = self.masked_fill(mask, value);
        self.assign_result(&result);
        self
    }

    /// Copies the elements of `src` into `self`, broadcasting `src` and converting its dtype.
    ///
    /// # Panics
    ///
    /// Panics if `src` can't be broadcast to the shape of `self`, if the elements of `self` can't
    /// be proven to occupy distinct memory (an expanded tensor, for example), or if `src` is a
    /// meta tensor and `self` is not.
    pub fn copy_(&self, src: &Tensor) -> &Self {
        self.write_copy(src);
        self
    }

    /// Sets every element to `value`.
    pub fn fill_(&self, value: impl Into<Scalar>) -> &Self {
        self.write_fill(value.into());
        self
    }

    /// Sets every element to zero.
    pub fn zero_(&self) -> &Self {
        self.write_fill(0.into());
        self
    }

    /// The data half of [`copy_`](Tensor::copy_), with the same checks and panics: writes `src`
    /// into `self` and bumps the version exactly once, without touching autograd state.
    ///
    /// Every public in-place operation writes its data with this function or
    /// [`write_fill`](Tensor::write_fill), exactly once, and never through another public
    /// in-place operation.
    pub(crate) fn write_copy(&self, src: &Tensor) {
        let dst = self.data();
        let mut s = src.data();
        check_writable(&dst);
        // Checks that `src` broadcasts to `self`.
        s.layout.expand_to(&dst.layout.shape);
        match (dst.device(), s.device()) {
            (Device::Meta, _) => {}
            (Device::Cpu, Device::Cpu) => {
                if dst.partially_overlaps(&s) {
                    s = clone_data(&s);
                }
                // The version changes in the same critical section as the data.
                cpu::exec(|| {
                    cpu::copy(&dst, &s);
                    dst.storage.bump_version();
                });
                return;
            }
            (_, Device::Meta) => panic!("Cannot copy out of meta tensor; no data!"),
            (d, sd) => panic!("copy from {sd} to {d} is not supported"),
        }
        dst.storage.bump_version();
    }

    /// The data half of [`fill_`](Tensor::fill_), see [`write_copy`](Tensor::write_copy).
    pub(crate) fn write_fill(&self, value: Scalar) {
        let dst = self.data();
        check_writable(&dst);
        if dst.device() == Device::Cpu {
            cpu::exec(|| {
                cpu::fill(&dst, value);
                dst.storage.bump_version();
            });
        } else {
            dst.storage.bump_version();
        }
    }
}

/// Panics unless the elements of `dst` are proven to be distinct memory locations (see
/// `Layout::may_overlap_internally`), so that no location is written more than once.
pub(crate) fn check_writable(dst: &TensorData) {
    if dst.layout.may_overlap_internally() {
        panic!(
            "unsupported operation: more than one element of the written-to tensor may refer to \
             a single memory location. Please deep_clone() the tensor before performing the \
             operation."
        );
    }
}
