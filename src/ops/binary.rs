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
    fn new(o: &Operand) -> Resolved {
        match o {
            Operand::Tensor(t) => Resolved::Tensor(t.data()),
            Operand::Scalar(s) => Resolved::Scalar(*s),
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
    let (a, b) = (Resolved::new(&lhs), Resolved::new(&rhs));
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
        // As in PyTorch, `f16` and `bf16` products and quotients keep a scalar operand in their
        // `f32` computation type, while the other operations round it to the dtype first.
        let keeps_scalar = matches!(op, Mul | Div) && matches!(compute, DType::F16 | DType::BF16);
        match (&a, &b) {
            (Resolved::Tensor(_), Resolved::Scalar(s))
            | (Resolved::Scalar(s), Resolved::Tensor(_))
                if keeps_scalar =>
            {
                let scalar_first = matches!(a, Resolved::Scalar(_));
                let t = if scalar_first { &b } else { &a };
                let td = t.materialize(compute, device, &shape);
                cpu::binary_scalar(op, &out.data(), &td, s.to_f64(), scalar_first);
            }
            _ => {
                let ad = a.materialize(compute, device, &shape);
                let bd = b.materialize(compute, device, &shape);
                cpu::binary(op, &out.data(), &ad, &bd);
            }
        }
    }
    crate::autograd::formulas::binary(op, &lhs, &rhs, &out);
    out
}

/// The tensor of an operand, `None` for a scalar.
fn tensor_of(o: &Operand) -> Option<&Tensor> {
    match o {
        Operand::Tensor(t) => Some(t),
        Operand::Scalar(_) => None,
    }
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
    let (x_operand, y_operand) = (x.into(), y.into());
    let (x, y) = (Resolved::new(&x_operand), Resolved::new(&y_operand));
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
    crate::autograd::formulas::where_(condition, &x_operand, &y_operand, &out);
    out
}

impl Tensor {
    fn binary(&self, op: BinaryOp, other: impl Into<Operand>) -> Tensor {
        binary_op(op, Operand::Tensor(self.clone()), other.into())
    }

    fn binary_(&self, op: BinaryOp, other: impl Into<Operand>) -> &Self {
        let other = other.into();
        let other_tensor = tensor_of(&other);
        // Products and quotients save `self` for the gradient of a tensor `other`, powers always.
        let saves_self = match op {
            BinaryOp::Mul | BinaryOp::Div => other_tensor.is_some_and(Tensor::requires_grad),
            BinaryOp::Pow => true,
            _ => false,
        };
        self.in_place(
            &[other_tensor],
            saves_self,
            || (self.binary(op, other.clone()), ()),
            |original, ()| {
                let lhs = Operand::Tensor(original.clone());
                crate::autograd::formulas::binary(op, &lhs, &other, self)
            },
        )
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
    /// Scalar bounds have zero gradients at equality; tensor bounds split ordinary ties.
    /// Unlike PyTorch, one bound may be a scalar and the other a tensor; its gradient follows
    /// the tensor-bound rule with the scalar rounded as in the corresponding forward step.
    pub fn clamp(&self, min: impl Into<Operand>, max: impl Into<Operand>) -> Tensor {
        let (min, max) = (min.into(), max.into());
        // Computed without recording the two steps: clamp gets a node of its own.
        let (out, dtypes) = {
            let _no_grad = crate::no_grad();
            self.clamp_values(&min, &max)
        };
        crate::autograd::formulas::clamp(self, Some(&min), Some(&max), dtypes, &out);
        out
    }

    /// `self.maximum(min).minimum(max)` and the dtypes the two steps compute in, which the
    /// gradient of mixed scalar and tensor bounds depends on.
    fn clamp_values(&self, min: &Operand, max: &Operand) -> (Tensor, [DType; 2]) {
        let lower = self.maximum(min.clone());
        let out = lower.minimum(max.clone());
        let dtypes = [lower.dtype(), out.dtype()];
        (out, dtypes)
    }

    /// Clamps every element to at least `min`.
    pub fn clamp_min(&self, min: impl Into<Operand>) -> Tensor {
        let min = min.into();
        let out = {
            let _no_grad = crate::no_grad();
            self.maximum(min.clone())
        };
        crate::autograd::formulas::clamp(self, Some(&min), None, [out.dtype(); 2], &out);
        out
    }

    /// Clamps every element to at most `max`.
    pub fn clamp_max(&self, max: impl Into<Operand>) -> Tensor {
        let max = max.into();
        let out = {
            let _no_grad = crate::no_grad();
            self.minimum(max.clone())
        };
        crate::autograd::formulas::clamp(self, None, Some(&max), [out.dtype(); 2], &out);
        out
    }

    /// In-place [`clamp`](Tensor::clamp).
    pub fn clamp_(&self, min: impl Into<Operand>, max: impl Into<Operand>) -> &Self {
        let (min, max) = (min.into(), max.into());
        // The formula computes in the dtypes of the computation, not in the dtype of `self`.
        self.in_place(
            &[tensor_of(&min), tensor_of(&max)],
            true,
            || self.clamp_values(&min, &max),
            |original, dtypes| {
                crate::autograd::formulas::clamp(original, Some(&min), Some(&max), dtypes, self)
            },
        )
    }

    /// A copy of `self` with `value` written where `mask` is true. `value` must be a scalar or a
    /// zero-dimensional tensor; the result keeps the dtype of `self`.
    pub fn masked_fill(&self, mask: &Tensor, value: impl Into<Operand>) -> Tensor {
        where_(mask, self.masked_fill_value(mask, value.into()), self)
    }

    /// Checks the arguments of [`masked_fill`](Tensor::masked_fill) and returns `value` as a
    /// zero-dimensional tensor of the dtype of `self`, converted with a recorded `to`.
    fn masked_fill_value(&self, mask: &Tensor, value: Operand) -> Tensor {
        let m = mask.dtype();
        if m != DType::Bool {
            panic!("masked_fill only supports boolean masks, but got mask with dtype {m}");
        }
        match value {
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
        }
    }

    /// In-place [`masked_fill`](Tensor::masked_fill).
    pub fn masked_fill_(&self, mask: &Tensor, value: impl Into<Operand>) -> &Self {
        let value = value.into();
        self.in_place(
            &[Some(mask), tensor_of(&value)],
            false,
            || (self.masked_fill(mask, value.clone()), ()),
            |original, ()| {
                let value = Operand::Tensor(self.masked_fill_value(mask, value.clone()));
                let original = Operand::Tensor(original.clone());
                crate::autograd::formulas::where_(mask, &value, &original, self)
            },
        )
    }

    /// Copies the elements of `src` into `self`, broadcasting `src` and converting its dtype.
    ///
    /// In grad mode, a copy into a floating point tensor is recorded when `self` or `src`
    /// requires grad (PyTorch's `CopyBackwards`): the old value of `self` gets a zero gradient
    /// and `src` the gradient of `self`, converted to its dtype and summed to its shape.
    ///
    /// # Panics
    ///
    /// Panics if the copy would have to be recorded into a leaf that requires grad or into a
    /// view (see [the in-place rules](crate::autograd#in-place-operations)), if `src` can't be
    /// broadcast to the shape of `self`, if the elements of `self` can't be proven to occupy
    /// distinct memory (an expanded tensor, for example), or if `src` is a meta tensor and
    /// `self` is not.
    pub fn copy_(&self, src: &Tensor) -> &Self {
        let recorded = crate::autograd::check_in_place(self, &[Some(src)]);
        self.check_copy(src);
        let (target, s) = (self.clone(), src.clone());
        self.write_in_place(
            recorded,
            move || target.write_copy(&s),
            || crate::autograd::formulas::copy(self, src),
        );
        self
    }

    /// Sets every element to `value`.
    ///
    /// In grad mode, the write into a floating point tensor that requires grad is recorded
    /// (`FillBackward2`), giving the old value a zero gradient.
    pub fn fill_(&self, value: impl Into<Scalar>) -> &Self {
        self.overwrite_(value.into(), "FillBackward2")
    }

    /// Sets every element to zero. Recorded like [`fill_`](Tensor::fill_), as `ZeroBackward0`.
    pub fn zero_(&self) -> &Self {
        self.overwrite_(0.into(), "ZeroBackward0")
    }

    /// `fill_` recorded as node `name`.
    fn overwrite_(&self, value: Scalar, name: &'static str) -> &Self {
        let recorded = crate::autograd::check_in_place(self, &[]);
        check_writable(&self.data());
        let target = self.clone();
        self.write_in_place(
            recorded,
            move || target.write_fill(value),
            || crate::autograd::formulas::overwrite(self, name),
        );
        self
    }

    /// The data half of [`copy_`](Tensor::copy_), with the same checks and panics: writes `src`
    /// into `self` and bumps the version exactly once, without touching autograd state.
    ///
    /// Every public in-place operation writes its data with this function or
    /// [`write_fill`](Tensor::write_fill), exactly once, and never through another public
    /// in-place operation.
    pub(crate) fn write_copy(&self, src: &Tensor) {
        let (dst, mut s) = self.check_copy(src);
        if dst.device() == Device::Meta {
            dst.storage.bump_version();
            return;
        }
        if dst.partially_overlaps(&s) {
            s = clone_data(&s);
        }
        // The version changes in the same critical section as the data.
        cpu::exec(|| {
            cpu::copy(&dst, &s);
            dst.storage.bump_version();
        });
    }

    /// The checks of [`write_copy`](Tensor::write_copy), which can't fail once they pass:
    /// `self` is writable, `src` broadcasts to it, and the devices allow the copy. These are
    /// not the checks of the other in-place operations' results: any dtype may be copied.
    /// Returns the data of `self` and of `src`.
    fn check_copy(&self, src: &Tensor) -> (TensorData, TensorData) {
        let dst = self.data();
        let s = src.data();
        check_writable(&dst);
        // Checks that `src` broadcasts to `self`.
        s.layout.expand_to(&dst.layout.shape);
        if let Some(e) = copy_device_error(&dst, &s) {
            panic!("{e}");
        }
        (dst, s)
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

/// Why `copy_` from a tensor with the same shape as the target `dst` would fail, or `None` if
/// it works: the checks of `copy_` other than broadcasting, for callers that check many copies
/// before doing any (`Module::load_state_dict`).
pub(crate) fn copy_error(dst: &TensorData, src: &TensorData) -> Option<String> {
    writable_error(dst)
        .map(String::from)
        .or_else(|| copy_device_error(dst, src))
}

fn copy_device_error(dst: &TensorData, src: &TensorData) -> Option<String> {
    match (dst.device(), src.device()) {
        (Device::Meta, _) | (Device::Cpu, Device::Cpu) => None,
        (_, Device::Meta) => Some("Cannot copy out of meta tensor; no data!".into()),
        (d, sd) => Some(format!("copy from {sd} to {d} is not supported")),
    }
}

/// Panics unless the elements of `dst` are proven to be distinct memory locations (see
/// `Layout::may_overlap_internally`), so that no location is written more than once.
pub(crate) fn check_writable(dst: &TensorData) {
    if let Some(e) = writable_error(dst) {
        panic!("{e}");
    }
}

fn writable_error(dst: &TensorData) -> Option<&'static str> {
    dst.layout.may_overlap_internally().then_some(
        "unsupported operation: more than one element of the written-to tensor may refer to a \
         single memory location. Please deep_clone() the tensor before performing the operation.",
    )
}
