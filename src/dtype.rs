//! Element types, scalars and type promotion.

use std::fmt;

use half::{bf16, f16};

/// The element type of a tensor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DType {
    /// `bool`, stored as one byte (0 or 1).
    Bool,
    /// `u8`.
    U8,
    /// `i8`.
    I8,
    /// `i16`.
    I16,
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// IEEE 754 half precision (`half::f16`).
    F16,
    /// bfloat16 (`half::bf16`).
    BF16,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

/// The category of a dtype, used by type promotion.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Kind {
    Bool,
    Int,
    Float,
}

impl DType {
    /// All dtypes, in declaration order.
    pub const ALL: [DType; 10] = [
        DType::Bool,
        DType::U8,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];

    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 => 1,
            DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }

    /// Whether this is a floating point type.
    pub fn is_floating_point(self) -> bool {
        self.kind() == Kind::Float
    }

    /// Whether this is an integer type (`bool` is not counted as an integer).
    pub fn is_integral(self) -> bool {
        self.kind() == Kind::Int
    }

    /// Whether this type can represent negative values.
    pub fn is_signed(self) -> bool {
        !matches!(self, DType::Bool | DType::U8)
    }

    /// Short lowercase name, such as `"f32"` or `"bf16"`.
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "u8",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
        }
    }

    /// The smallest and largest values of an integer or bool dtype.
    pub(crate) fn int_range(self) -> (i64, i64) {
        match self {
            DType::Bool => (0, 1),
            DType::U8 => (0, u8::MAX as i64),
            DType::I8 => (i8::MIN as i64, i8::MAX as i64),
            DType::I16 => (i16::MIN as i64, i16::MAX as i64),
            DType::I32 => (i32::MIN as i64, i32::MAX as i64),
            DType::I64 => (i64::MIN, i64::MAX),
            _ => unreachable!("{self} is not an integer dtype"),
        }
    }

    /// The largest finite value of a floating point dtype.
    pub(crate) fn float_max(self) -> f64 {
        match self {
            DType::F16 => f16::MAX.to_f64(),
            DType::BF16 => bf16::MAX.to_f64(),
            DType::F32 => f32::MAX as f64,
            DType::F64 => f64::MAX,
            _ => unreachable!("{self} is not a floating point dtype"),
        }
    }

    pub(crate) fn kind(self) -> Kind {
        match self {
            DType::Bool => Kind::Bool,
            DType::U8 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => Kind::Int,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Kind::Float,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub(crate) mod sealed {
    use super::Scalar;

    /// Conversions used by the kernels. Kept out of the public API by living in a private module.
    pub trait Sealed: Copy {
        fn to_f64(self) -> f64;
        fn to_i64(self) -> i64;
        fn to_bool(self) -> bool;
        fn from_f64(v: f64) -> Self;
        fn from_i64(v: i64) -> Self;
        fn from_bool(v: bool) -> Self;

        fn from_scalar(s: Scalar) -> Self {
            match s {
                Scalar::Bool(v) => Self::from_bool(v),
                Scalar::Int(v) => Self::from_i64(v),
                Scalar::Float(v) => Self::from_f64(v),
            }
        }
    }
}

/// A Rust type that can be stored in a tensor.
///
/// Implemented for `bool`, `u8`, `i8`, `i16`, `i32`, `i64`, `half::f16`, `half::bf16`, `f32` and
/// `f64`. This trait is sealed.
pub trait Element:
    sealed::Sealed + Default + PartialEq + PartialOrd + fmt::Debug + Send + Sync + 'static
{
    /// The dtype of this element type.
    const DTYPE: DType;

    /// Converts to another element type.
    ///
    /// Integers wrap (two's complement truncation) and any nonzero value becomes `true`. Floats
    /// cast to an integer type are first truncated toward zero to `i64`, saturating at its limits
    /// (NaN becomes 0), and then wrapped to the target type, so unlike C++ `static_cast` the
    /// result is defined for every input.
    ///
    /// This matches PyTorch on x86-64 when the truncated value fits in the target type, and for
    /// `u8`, `i8` and `i16` also when it fits in `i32` (`300.0` becomes `44u8`, `-1.0` becomes
    /// `255u8`). Other conversions deliberately differ: there PyTorch's result is platform
    /// dependent, for example `-2147483649.0` becomes `i32::MIN` on x86-64 but `i32::MAX` here.
    #[doc(hidden)]
    fn cast<D: Element>(self) -> D {
        match Self::DTYPE.kind() {
            Kind::Float => D::from_f64(self.to_f64()),
            Kind::Int => D::from_i64(self.to_i64()),
            Kind::Bool => D::from_bool(self.to_bool()),
        }
    }
}

macro_rules! impl_int_element {
    ($($t:ty => $dt:ident),*) => {$(
        impl sealed::Sealed for $t {
            fn to_f64(self) -> f64 { self as f64 }
            fn to_i64(self) -> i64 { self as i64 }
            fn to_bool(self) -> bool { self != 0 }
            // Saturate to `i64`, then wrap: see `Element::cast`.
            fn from_f64(v: f64) -> Self { v as i64 as $t }
            fn from_i64(v: i64) -> Self { v as $t }
            fn from_bool(v: bool) -> Self { v as $t }
        }
        impl Element for $t {
            const DTYPE: DType = DType::$dt;
        }
    )*};
}

impl_int_element!(u8 => U8, i8 => I8, i16 => I16, i32 => I32, i64 => I64);

macro_rules! impl_float_element {
    ($($t:ty => $dt:ident),*) => {$(
        impl sealed::Sealed for $t {
            fn to_f64(self) -> f64 { self as f64 }
            fn to_i64(self) -> i64 { self as i64 }
            fn to_bool(self) -> bool { self != 0.0 }
            fn from_f64(v: f64) -> Self { v as $t }
            fn from_i64(v: i64) -> Self { v as $t }
            fn from_bool(v: bool) -> Self { if v { 1.0 } else { 0.0 } }
        }
        impl Element for $t {
            const DTYPE: DType = DType::$dt;
        }
    )*};
}

impl_float_element!(f32 => F32, f64 => F64);

macro_rules! impl_half_element {
    ($($t:ty => $dt:ident),*) => {$(
        impl sealed::Sealed for $t {
            fn to_f64(self) -> f64 { self.to_f64() }
            fn to_i64(self) -> i64 { self.to_f32() as i64 }
            fn to_bool(self) -> bool { self.to_f32() != 0.0 }
            fn from_f64(v: f64) -> Self { <$t>::from_f64(v) }
            fn from_i64(v: i64) -> Self { <$t>::from_f64(v as f64) }
            fn from_bool(v: bool) -> Self { if v { <$t>::ONE } else { <$t>::ZERO } }
        }
        impl Element for $t {
            const DTYPE: DType = DType::$dt;
        }
    )*};
}

impl_half_element!(f16 => F16, bf16 => BF16);

impl sealed::Sealed for bool {
    fn to_f64(self) -> f64 {
        self as u8 as f64
    }
    fn to_i64(self) -> i64 {
        self as i64
    }
    fn to_bool(self) -> bool {
        self
    }
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
    fn from_i64(v: i64) -> Self {
        v != 0
    }
    fn from_bool(v: bool) -> Self {
        v
    }
}

impl Element for bool {
    const DTYPE: DType = DType::Bool;
}

/// A single number: the Rust counterpart of a Python number passed to a PyTorch function.
///
/// Scalars take part in type promotion with the lowest priority: `&x * 2.0` keeps the dtype of an
/// `f16` tensor `x`, while `&i * 2.5` on an integer tensor gives the default float dtype.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scalar {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A floating point number.
    Float(f64),
}

impl Scalar {
    /// The value as `f64`.
    pub fn to_f64(self) -> f64 {
        match self {
            Scalar::Bool(v) => v as u8 as f64,
            Scalar::Int(v) => v as f64,
            Scalar::Float(v) => v,
        }
    }

    /// The value as `i64` (floats are truncated toward zero).
    pub fn to_i64(self) -> i64 {
        match self {
            Scalar::Bool(v) => v as i64,
            Scalar::Int(v) => v,
            Scalar::Float(v) => v as i64,
        }
    }

    /// Whether the value is nonzero.
    pub fn to_bool(self) -> bool {
        match self {
            Scalar::Bool(v) => v,
            Scalar::Int(v) => v != 0,
            Scalar::Float(v) => v != 0.0,
        }
    }

    /// The dtype a tensor created from this scalar gets: the default dtype for floats, `I64` for
    /// integers and `Bool` for booleans.
    pub(crate) fn dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::Float(_) => crate::default_dtype(),
        }
    }

    /// The value as a scalar of `dtype`, converted the way PyTorch checks a scalar it writes into
    /// a tensor (`fill_`, `full`, `masked_fill`): `None` if it overflows `dtype`.
    ///
    /// - Bool targets take any value, as nonzero or zero.
    /// - Floating point targets reject finite values beyond the dtype's largest finite value;
    ///   infinities and NaN pass, and the value is kept, to be rounded by the kernel.
    /// - Integer targets take integers in their range, plus, for `U8`, negative integers down to
    ///   `-255`, which wrap around. Floats must lie in `[MIN, MAX + 1)` (checked before they are
    ///   truncated toward zero, so `-0.5` doesn't fit `U8` but `255.9` becomes 255); infinities
    ///   and NaN never fit.
    pub(crate) fn checked_to(self, dtype: DType) -> Option<Scalar> {
        match dtype.kind() {
            Kind::Bool => Some(Scalar::Bool(self.to_bool())),
            Kind::Float => {
                let v = self.to_f64();
                (!(v.is_finite() && v.abs() > dtype.float_max())).then_some(self)
            }
            Kind::Int => {
                let (min, max) = dtype.int_range();
                match self {
                    Scalar::Bool(v) => Some(Scalar::Int(v as i64)),
                    Scalar::Int(v) => {
                        // Like PyTorch, negative integers down to `-255` wrap around for `U8`.
                        let min = if dtype == DType::U8 { -max } else { min };
                        (min..=max).contains(&v).then_some(Scalar::Int(v))
                    }
                    Scalar::Float(v) => {
                        // Fractional values above MAX still truncate to MAX; MAX + 1 is the
                        // exclusive upper bound. Compute it in i128 so I64 uses exactly 2^63, not
                        // a rounded MAX.
                        let upper = (max as i128 + 1) as f64;
                        (min as f64..upper)
                            .contains(&v)
                            .then_some(Scalar::Int(v as i64))
                    }
                }
            }
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Bool(v) => write!(f, "{v}"),
            Scalar::Int(v) => write!(f, "{v}"),
            Scalar::Float(v) => write!(f, "{v}"),
        }
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}

macro_rules! scalar_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Scalar {
            fn from(v: $t) -> Self {
                Scalar::Int(v as i64)
            }
        }
    )*};
}

scalar_from_int!(u8, i8, u16, i16, u32, i32, i64, isize, usize);

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Scalar::Float(v as f64)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::Float(v)
    }
}

impl From<f16> for Scalar {
    fn from(v: f16) -> Self {
        Scalar::Float(v.to_f64())
    }
}

impl From<bf16> for Scalar {
    fn from(v: bf16) -> Self {
        Scalar::Float(v.to_f64())
    }
}

/// The smallest dtype both `a` and `b` can be converted to, following `torch.promote_types`.
pub fn promote_types(a: DType, b: DType) -> DType {
    use DType::*;
    if a == b {
        return a;
    }
    match (a.kind(), b.kind()) {
        (Kind::Bool, _) => b,
        (_, Kind::Bool) => a,
        (Kind::Float, Kind::Float) => {
            if a == F64 || b == F64 {
                F64
            } else {
                // f16 + bf16, f16 + f32 and bf16 + f32 all promote to f32.
                F32
            }
        }
        (Kind::Float, _) => a,
        (_, Kind::Float) => b,
        (Kind::Int, Kind::Int) => {
            let rank = |d: DType| match d {
                I8 => 1,
                I16 => 2,
                I32 => 3,
                I64 => 4,
                _ => 0,
            };
            match (a, b) {
                // u8 mixed with a signed type needs the next wider signed type to hold both ranges.
                (U8, I8) | (I8, U8) => I16,
                (U8, s) | (s, U8) => s,
                _ => {
                    if rank(a) >= rank(b) {
                        a
                    } else {
                        b
                    }
                }
            }
        }
    }
}

/// Whether a value of dtype `from` may be written into a tensor of dtype `to` by an in-place
/// operation (`torch.can_cast`): floats can't go to integers, and only `bool` can go to `bool`.
pub fn can_cast(from: DType, to: DType) -> bool {
    !(from.is_floating_point() && !to.is_floating_point()
        || from != DType::Bool && to == DType::Bool)
}

/// Accumulates the operands of an operation and computes the result dtype like
/// `torch.result_type`: tensors with dimensions take priority over zero-dimensional tensors, which
/// take priority over scalars, unless a lower priority operand belongs to a higher category
/// (bool < integer < float).
#[derive(Default, Debug)]
pub(crate) struct ResultType {
    dim: Option<DType>,
    zero_dim: Option<DType>,
    scalar: Option<DType>,
}

impl ResultType {
    pub fn new() -> Self {
        Self::default()
    }

    fn update(slot: &mut Option<DType>, dtype: DType) {
        *slot = Some(match *slot {
            None => dtype,
            Some(d) => promote_types(d, dtype),
        });
    }

    pub fn tensor(&mut self, dtype: DType, num_dims: usize) -> &mut Self {
        if num_dims == 0 {
            Self::update(&mut self.zero_dim, dtype);
        } else {
            Self::update(&mut self.dim, dtype);
        }
        self
    }

    pub fn scalar(&mut self, s: Scalar) -> &mut Self {
        Self::update(&mut self.scalar, s.dtype());
        self
    }

    pub fn result(&self) -> DType {
        fn combine(higher: Option<DType>, lower: Option<DType>) -> Option<DType> {
            match (higher, lower) {
                (None, l) => l,
                (h, None) => h,
                (Some(h), Some(l)) => {
                    if h.is_floating_point() {
                        Some(h)
                    } else if h == DType::Bool || l.is_floating_point() {
                        Some(promote_types(h, l))
                    } else {
                        Some(h)
                    }
                }
            }
        }
        combine(self.dim, combine(self.zero_dim, self.scalar))
            .expect("result_type needs at least one operand")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DType::*;

    #[test]
    fn promotion_table() {
        assert_eq!(promote_types(U8, I8), I16);
        assert_eq!(promote_types(U8, I32), I32);
        assert_eq!(promote_types(I16, I32), I32);
        assert_eq!(promote_types(F16, BF16), F32);
        assert_eq!(promote_types(BF16, F64), F64);
        assert_eq!(promote_types(I64, F16), F16);
        assert_eq!(promote_types(Bool, U8), U8);
        for &d in &DType::ALL {
            assert_eq!(promote_types(d, d), d);
            assert_eq!(promote_types(Bool, d), d);
        }
    }

    #[test]
    fn result_type_priorities() {
        // An f16 tensor times a float scalar stays f16.
        assert_eq!(
            ResultType::new()
                .tensor(F16, 1)
                .scalar(Scalar::Float(2.0))
                .result(),
            F16
        );
        // An integer tensor times a float scalar gets the default dtype.
        assert_eq!(
            ResultType::new()
                .tensor(I32, 1)
                .scalar(Scalar::Float(2.5))
                .result(),
            F32
        );
        // A zero-dim tensor of a higher category wins over a dimensioned one.
        assert_eq!(
            ResultType::new().tensor(I64, 2).tensor(F64, 0).result(),
            F64
        );
        // ...but not within the same category.
        assert_eq!(
            ResultType::new().tensor(F32, 2).tensor(F64, 0).result(),
            F32
        );
        assert_eq!(
            ResultType::new()
                .tensor(U8, 1)
                .scalar(Scalar::Int(-1))
                .result(),
            U8
        );
        assert_eq!(
            ResultType::new()
                .tensor(Bool, 1)
                .scalar(Scalar::Int(1))
                .result(),
            I64
        );
    }

    #[test]
    fn casting_rules() {
        assert!(can_cast(I64, F16));
        assert!(!can_cast(F32, I64));
        assert!(!can_cast(I32, Bool));
        assert!(can_cast(Bool, Bool));
        assert!(can_cast(F64, F16));
    }

    #[test]
    fn element_cast() {
        assert_eq!((-1.7f32).cast::<i32>(), -1);
        assert_eq!(300i64.cast::<u8>(), 44);
        assert_eq!(300.0f32.cast::<u8>(), 44);
        assert_eq!((-1.0f64).cast::<u8>(), 255);
        assert_eq!((-129.9f32).cast::<i8>(), 127);
        assert_eq!(f16::from_f32(-2.5).cast::<u8>(), 254);
        assert_eq!(f64::NAN.cast::<i32>(), 0);
        assert_eq!(f64::INFINITY.cast::<i64>(), i64::MAX);
        assert_eq!(f64::INFINITY.cast::<i32>(), -1);
        assert_eq!(1e30f32.cast::<i16>(), -1);
        assert!(0.1f64.cast::<bool>());
        assert_eq!(true.cast::<f16>(), f16::ONE);
        assert_eq!(f16::from_f32(2.5).cast::<f32>(), 2.5);
    }
}
