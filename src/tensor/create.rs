//! Tensor creation functions.

use half::{bf16, f16};

use crate::backend::cpu::{self, dispatch_all};
use crate::dtype::sealed::Sealed;
use crate::shape::IntoShape;
use crate::{DType, Device, Element, Scalar, Shape, Tensor, default_device, default_dtype};

/// The dtype and device of a new tensor. Fields left as `None` use the defaults.
///
/// Converted implicitly from a [`DType`], a [`Device`] or a tuple of both, so creation
/// functions accept `homura::zeros_with([2, 3], DType::F16)` or
/// `homura::zeros_with([2, 3], (DType::F16, Device::Meta))`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TensorOptions {
    /// The dtype, or `None` for the function's default.
    pub dtype: Option<DType>,
    /// The device, or `None` for [`default_device`].
    pub device: Option<Device>,
}

impl TensorOptions {
    pub(crate) fn float_dtype(&self) -> DType {
        self.dtype.unwrap_or_else(default_dtype)
    }

    pub(crate) fn device(&self) -> Device {
        self.device.unwrap_or_else(default_device)
    }
}

impl From<DType> for TensorOptions {
    fn from(dtype: DType) -> Self {
        TensorOptions {
            dtype: Some(dtype),
            device: None,
        }
    }
}

impl From<Device> for TensorOptions {
    fn from(device: Device) -> Self {
        TensorOptions {
            dtype: None,
            device: Some(device),
        }
    }
}

impl From<(DType, Device)> for TensorOptions {
    fn from((dtype, device): (DType, Device)) -> Self {
        TensorOptions {
            dtype: Some(dtype),
            device: Some(device),
        }
    }
}

impl From<(Device, DType)> for TensorOptions {
    fn from((device, dtype): (Device, DType)) -> Self {
        TensorOptions {
            dtype: Some(dtype),
            device: Some(device),
        }
    }
}

/// Nested arrays, vectors and slices of [`Element`]s that [`tensor`] accepts.
pub trait NestedData {
    /// The element type.
    type Elem: Element;
    #[doc(hidden)]
    fn collect_shape(&self, shape: &mut Vec<usize>);
    /// Appends the values in row-major order, checking that the data has the given shape.
    #[doc(hidden)]
    fn collect_values(&self, shape: &[usize], out: &mut Vec<Self::Elem>);
}

macro_rules! nested_scalar {
    ($($t:ty),*) => {$(
        impl NestedData for $t {
            type Elem = $t;
            fn collect_shape(&self, _: &mut Vec<usize>) {}
            fn collect_values(&self, shape: &[usize], out: &mut Vec<$t>) {
                if !shape.is_empty() {
                    panic!("nested data is ragged: found a value where a sequence was expected");
                }
                out.push(*self);
            }
        }
    )*};
}

nested_scalar!(bool, u8, i8, i16, i32, i64, f16, bf16, f32, f64);

impl<U: NestedData> NestedData for [U] {
    type Elem = U::Elem;
    fn collect_shape(&self, shape: &mut Vec<usize>) {
        shape.push(self.len());
        if let Some(first) = self.first() {
            first.collect_shape(shape);
        }
    }
    fn collect_values(&self, shape: &[usize], out: &mut Vec<U::Elem>) {
        match shape.split_first() {
            Some((&n, rest)) if n == self.len() => {
                for x in self {
                    x.collect_values(rest, out);
                }
            }
            Some((&n, _)) => {
                panic!(
                    "nested data is ragged: expected a sequence of length {n}, got {}",
                    self.len()
                )
            }
            None => panic!("nested data is ragged: found a sequence where a value was expected"),
        }
    }
}

impl<U: NestedData, const N: usize> NestedData for [U; N] {
    type Elem = U::Elem;
    fn collect_shape(&self, shape: &mut Vec<usize>) {
        self.as_slice().collect_shape(shape)
    }
    fn collect_values(&self, shape: &[usize], out: &mut Vec<U::Elem>) {
        self.as_slice().collect_values(shape, out)
    }
}

impl<U: NestedData> NestedData for Vec<U> {
    type Elem = U::Elem;
    fn collect_shape(&self, shape: &mut Vec<usize>) {
        self.as_slice().collect_shape(shape)
    }
    fn collect_values(&self, shape: &[usize], out: &mut Vec<U::Elem>) {
        self.as_slice().collect_values(shape, out)
    }
}

impl<U: NestedData + ?Sized> NestedData for &U {
    type Elem = U::Elem;
    fn collect_shape(&self, shape: &mut Vec<usize>) {
        (**self).collect_shape(shape)
    }
    fn collect_values(&self, shape: &[usize], out: &mut Vec<U::Elem>) {
        (**self).collect_values(shape, out)
    }
}

fn nested_to_tensor<D: NestedData + ?Sized>(data: &D, device: Device) -> Tensor {
    let mut shape = vec![];
    data.collect_shape(&mut shape);
    let mut values = vec![];
    data.collect_values(&shape, &mut values);
    Tensor::from_host(&values, Shape::new(&shape), device)
}

/// Creates a tensor from nested arrays, vectors or slices on the default device.
///
/// Like Python numbers in `torch.tensor`, Rust's default literal types are mapped to PyTorch's
/// defaults: `f64` data gets the default dtype (`F32` unless changed) and `i32` data becomes
/// `I64`. Other element types keep their dtype; use [`tensor_with`] or [`Tensor::from_slice`] for
/// an exact dtype.
///
/// ```
/// use homura::DType;
/// let x = homura::tensor([[1., 2.], [3., 4.]]);
/// assert_eq!((x.shape(), x.dtype()), ([2, 2].into(), DType::F32));
/// assert_eq!(homura::tensor([1, 2, 3]).dtype(), DType::I64);
/// assert_eq!(homura::tensor([true, false]).dtype(), DType::Bool);
/// ```
pub fn tensor<D: NestedData>(data: D) -> Tensor {
    let t = nested_to_tensor(&data, default_device());
    match D::Elem::DTYPE {
        DType::F64 => t.to(default_dtype()),
        DType::I32 => t.to(DType::I64),
        _ => t,
    }
}

/// [`tensor`] with an explicit dtype and/or device. Without a dtype, the element type's own
/// dtype is used.
pub fn tensor_with<D: NestedData>(data: D, options: impl Into<TensorOptions>) -> Tensor {
    let o = options.into();
    let t = nested_to_tensor(&data, o.device());
    match o.dtype {
        Some(d) => t.to(d),
        None => t,
    }
}

impl Tensor {
    /// Creates a tensor of the given shape from a slice of values with the exact dtype of `T`,
    /// on the default device.
    pub fn from_slice<T: Element>(values: &[T], shape: impl IntoShape) -> Tensor {
        Tensor::from_host(values, shape.into_shape(), default_device())
    }

    /// Creates a tensor of the given shape from a vector, like [`Tensor::from_slice`].
    pub fn from_vec<T: Element>(values: Vec<T>, shape: impl IntoShape) -> Tensor {
        Tensor::from_slice(&values, shape)
    }

    /// A tensor of zeros with the dtype and device of `self`.
    pub fn new_zeros(&self, shape: impl IntoShape) -> Tensor {
        zeros_with(shape, (self.dtype(), self.device()))
    }

    /// A tensor of ones with the dtype and device of `self`.
    pub fn new_ones(&self, shape: impl IntoShape) -> Tensor {
        ones_with(shape, (self.dtype(), self.device()))
    }

    /// A tensor filled with `value` with the dtype and device of `self`.
    pub fn new_full(&self, shape: impl IntoShape, value: impl Into<Scalar>) -> Tensor {
        full_with(shape, value, (self.dtype(), self.device()))
    }
}

/// A tensor filled with `value`.
fn filled(shape: Shape, value: Scalar, dtype: DType, device: Device) -> Tensor {
    let t = Tensor::empty_on(shape, dtype, device);
    if device == Device::Cpu && value.to_f64() != 0.0 {
        cpu::fill(&t.data(), value);
    }
    t
}

/// Zeros with the default dtype and device.
pub fn zeros(shape: impl IntoShape) -> Tensor {
    zeros_with(shape, TensorOptions::default())
}

/// Zeros with the given dtype and/or device.
pub fn zeros_with(shape: impl IntoShape, options: impl Into<TensorOptions>) -> Tensor {
    let o = options.into();
    filled(
        shape.into_shape(),
        Scalar::Int(0),
        o.float_dtype(),
        o.device(),
    )
}

/// Ones with the default dtype and device.
pub fn ones(shape: impl IntoShape) -> Tensor {
    ones_with(shape, TensorOptions::default())
}

/// Ones with the given dtype and/or device.
pub fn ones_with(shape: impl IntoShape, options: impl Into<TensorOptions>) -> Tensor {
    let o = options.into();
    filled(
        shape.into_shape(),
        Scalar::Int(1),
        o.float_dtype(),
        o.device(),
    )
}

/// A tensor filled with `value`. The dtype follows the value: the default dtype for floats,
/// `I64` for integers and `Bool` for booleans.
pub fn full(shape: impl IntoShape, value: impl Into<Scalar>) -> Tensor {
    full_with(shape, value, TensorOptions::default())
}

/// [`full`] with the given dtype and/or device.
pub fn full_with(
    shape: impl IntoShape,
    value: impl Into<Scalar>,
    options: impl Into<TensorOptions>,
) -> Tensor {
    let (value, o) = (value.into(), options.into());
    filled(
        shape.into_shape(),
        value,
        o.dtype.unwrap_or(value.dtype()),
        o.device(),
    )
}

/// Zeros with the shape, dtype and device of `t`.
pub fn zeros_like(t: &Tensor) -> Tensor {
    t.new_zeros(t.shape())
}

/// Ones with the shape, dtype and device of `t`.
pub fn ones_like(t: &Tensor) -> Tensor {
    t.new_ones(t.shape())
}

/// `value` with the shape, dtype and device of `t`.
pub fn full_like(t: &Tensor, value: impl Into<Scalar>) -> Tensor {
    t.new_full(t.shape(), value)
}

/// A tensor from values computed in `f64` (floats) or `i64` (integers).
fn from_values(n: usize, dtype: DType, device: Device, f: impl Fn(usize) -> Scalar) -> Tensor {
    // Allocating first rejects sizes that are too large before computing any values.
    let t = Tensor::empty_on(Shape::new(&[n]), dtype, device);
    if device == Device::Cpu {
        dispatch_all!(dtype, T => {
            let values: Vec<T> = (0..n).map(|i| T::from_scalar(f(i))).collect();
            cpu::write_host(&t.data(), &values);
        });
    }
    t
}

/// `[0, 1, ..., end)` on the default device; `I64` for an integer `end`, otherwise the default
/// dtype.
pub fn arange(end: impl Into<Scalar>) -> Tensor {
    arange_step_with(0, end, 1, TensorOptions::default())
}

/// [`arange`] with the given dtype and/or device.
pub fn arange_with(end: impl Into<Scalar>, options: impl Into<TensorOptions>) -> Tensor {
    arange_step_with(0, end, 1, options)
}

/// `[start, start + step, ...)` up to (excluding) `end`. The dtype is `I64` when all arguments
/// are integers, otherwise the default dtype.
pub fn arange_step(
    start: impl Into<Scalar>,
    end: impl Into<Scalar>,
    step: impl Into<Scalar>,
) -> Tensor {
    arange_step_with(start, end, step, TensorOptions::default())
}

/// [`arange_step`] with the given dtype and/or device.
pub fn arange_step_with(
    start: impl Into<Scalar>,
    end: impl Into<Scalar>,
    step: impl Into<Scalar>,
    options: impl Into<TensorOptions>,
) -> Tensor {
    let (start, end, step) = (start.into(), end.into(), step.into());
    let o = options.into();
    let all_int = [start, end, step]
        .iter()
        .all(|s| !matches!(s, Scalar::Float(_)));
    let dtype = o
        .dtype
        .unwrap_or(if all_int { DType::I64 } else { default_dtype() });
    if all_int {
        // Integer bounds stay integral: `f64` can't represent every `i64`.
        let (s, e, st) = (start.to_i64(), end.to_i64(), step.to_i64());
        if st == 0 {
            panic!("step must be nonzero");
        }
        if (st > 0 && e < s) || (st < 0 && e > s) {
            panic!("upper bound and larger bound inconsistent with step sign");
        }
        // The number of values is ceil((e - s) / st), which is nonnegative.
        let (span, st_abs) = ((e as i128 - s as i128).abs(), (st as i128).abs());
        let n = (span + st_abs - 1) / st_abs;
        let n = usize::try_from(n).expect("arange: too many elements");
        // Every value lies between `s` and `e`, so it fits in `i64`.
        return from_values(n, dtype, o.device(), |i| {
            Scalar::Int((s as i128 + st as i128 * i as i128) as i64)
        });
    }
    let (s, e, st) = (start.to_f64(), end.to_f64(), step.to_f64());
    if st == 0.0 {
        panic!("step must be nonzero");
    }
    if !(s.is_finite() && e.is_finite() && st.is_finite()) {
        panic!("unsupported range: {s} -> {e} with step {st}");
    }
    if (st > 0.0 && e < s) || (st < 0.0 && e > s) {
        panic!("upper bound and larger bound inconsistent with step sign");
    }
    let n = ((e - s) / st).ceil().max(0.0);
    if n >= isize::MAX as f64 {
        panic!("arange: too many elements");
    }
    from_values(n as usize, dtype, o.device(), |i| {
        Scalar::Float(s + st * i as f64)
    })
}

/// `steps` evenly spaced values from `start` to `end` inclusive, with the default dtype.
pub fn linspace(start: f64, end: f64, steps: usize) -> Tensor {
    linspace_with(start, end, steps, TensorOptions::default())
}

/// [`linspace`] with the given dtype and/or device.
pub fn linspace_with(
    start: f64,
    end: f64,
    steps: usize,
    options: impl Into<TensorOptions>,
) -> Tensor {
    let o = options.into();
    let step = if steps > 1 {
        (end - start) / (steps - 1) as f64
    } else {
        0.0
    };
    // Like PyTorch, fill the second half backwards from `end` so both ends are exact.
    from_values(steps, o.float_dtype(), o.device(), |i| {
        Scalar::Float(if i < steps / 2 || steps == 1 {
            start + step * i as f64
        } else {
            end - step * (steps - 1 - i) as f64
        })
    })
}

/// The `n × n` identity matrix with the default dtype.
pub fn eye(n: usize) -> Tensor {
    eye_with(n, TensorOptions::default())
}

/// [`eye`] with the given dtype and/or device.
pub fn eye_with(n: usize, options: impl Into<TensorOptions>) -> Tensor {
    let o = options.into();
    from_values(n * n, o.float_dtype(), o.device(), |i| {
        Scalar::Int((i / n == i % n) as i64)
    })
    .view([n as i64, n as i64])
}
