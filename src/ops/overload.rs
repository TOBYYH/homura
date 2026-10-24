//! Arithmetic operator overloads for tensors and scalars.
//!
//! `&a + &b` borrows both operands; `a + b` also works but moves them. Any Rust [`Number`] may
//! appear on the right (`&x * 2.0`, `&x / n` with `n: f32`); on the left only `f64` and `i64`
//! are supported (`1.0 - &x`), so that literals infer without annotations. Compound assignment
//! (`x += &y`) is in place, like Python. Iterators of tensors have a [`Sum`] and a [`Product`],
//! like Python's `sum` and `math.prod`.

use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::{BinaryOp, Operand, binary::binary_op};
use crate::{Scalar, Tensor};

mod sealed {
    pub trait Sealed {}
}

/// Rust number types usable as scalar operands of tensor operators (`&x * 2.0`).
pub trait Number: Into<Scalar> + Copy + sealed::Sealed {}

macro_rules! number {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl Number for $t {}
    )*};
}

number!(
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

macro_rules! tensor_tensor {
    ($Trait:ident, $method:ident, $op:ident) => {
        impl $Trait<&Tensor> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
        impl $Trait<Tensor> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
        impl $Trait<&Tensor> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
        impl $Trait<Tensor> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
    };
}

macro_rules! tensor_scalar {
    ($Trait:ident, $method:ident, $op:ident) => {
        impl<S: Number> $Trait<S> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: S) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into().into())
            }
        }
        impl<S: Number> $Trait<S> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: S) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into().into())
            }
        }
    };
}

macro_rules! scalar_tensor {
    ($Trait:ident, $method:ident, $op:ident, $($s:ty),*) => {$(
        impl $Trait<&Tensor> for $s {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                binary_op(BinaryOp::$op, Scalar::from(self).into(), rhs.into())
            }
        }
        impl $Trait<Tensor> for $s {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                binary_op(BinaryOp::$op, Scalar::from(self).into(), rhs.into())
            }
        }
    )*};
}

macro_rules! assign {
    ($Trait:ident, $method:ident, $inplace:ident) => {
        impl<T: Into<Operand>> $Trait<T> for Tensor {
            #[track_caller]
            fn $method(&mut self, rhs: T) {
                self.$inplace(rhs);
            }
        }
    };
}

macro_rules! arith {
    ($Trait:ident, $method:ident, $op:ident, $AssignTrait:ident, $assign:ident, $inplace:ident) => {
        tensor_tensor!($Trait, $method, $op);
        tensor_scalar!($Trait, $method, $op);
        scalar_tensor!($Trait, $method, $op, f64, i64);
        assign!($AssignTrait, $assign, $inplace);
    };
}

arith!(Add, add, Add, AddAssign, add_assign, add_);
arith!(Sub, sub, Sub, SubAssign, sub_assign, sub_);
arith!(Mul, mul, Mul, MulAssign, mul_assign, mul_);
arith!(Div, div, Div, DivAssign, div_assign, div_);

impl Neg for &Tensor {
    type Output = Tensor;
    #[track_caller]
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    #[track_caller]
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}

/// Folds `iter` like Python's `sum(iter)` / `math.prod(iter)`: `start` is a plain integer, so it
/// takes part in the first operation as a weak scalar and the dtype and device of the result
/// come from the tensors. Without tensors the result is `start`, as a zero-dimensional `I64`
/// tensor on the default device.
#[track_caller]
fn fold<T: Into<Operand>>(mut iter: impl Iterator<Item = T>, start: i64, op: BinaryOp) -> Tensor {
    let Some(first) = iter.next() else {
        return crate::tensor(start);
    };
    let mut acc = binary_op(op, first.into(), Scalar::Int(start).into());
    for t in iter {
        acc = binary_op(op, acc.into(), t.into());
    }
    acc
}

/// Python's `sum(tensors)`: `0 + a + b + ...`, from left to right, where `0` is a plain integer,
/// which doesn't change the dtype (an `I32` sum stays `I32`, a `Bool` one is `I64`) or the
/// device. A single tensor gives a new tensor, recorded as `AddBackward0`. Without tensors, the
/// sum is `0` as a zero-dimensional `I64` tensor on the [default device](crate::default_device).
///
/// ```
/// use homura::{Tensor, tensor};
///
/// let losses = [tensor(1.0), tensor(2.5)];
/// let total: Tensor = losses.iter().sum();
/// assert_eq!(total.item::<f32>(), 3.5);
/// ```
///
/// A panic in an addition (tensors that don't broadcast, or on different devices) is reported
/// where the standard library's `Iterator::sum` calls this, not at the caller's line.
impl Sum for Tensor {
    #[track_caller]
    fn sum<I: Iterator<Item = Tensor>>(iter: I) -> Tensor {
        fold(iter, 0, BinaryOp::Add)
    }
}

/// Python's `sum(tensors)`, as for an iterator of tensors.
impl<'a> Sum<&'a Tensor> for Tensor {
    #[track_caller]
    fn sum<I: Iterator<Item = &'a Tensor>>(iter: I) -> Tensor {
        fold(iter, 0, BinaryOp::Add)
    }
}

/// Python's `math.prod(tensors)`: `1 * a * b * ...`, from left to right, where `1` is a plain
/// integer, which doesn't change the dtype (an `I32` product stays `I32`, a `Bool` one is `I64`)
/// or the device. A single tensor gives a new tensor, recorded as `MulBackward0`. Without
/// tensors, the product is `1` as a zero-dimensional `I64` tensor on the
/// [default device](crate::default_device).
///
/// A panic in a multiplication is reported where the standard library's `Iterator::product`
/// calls this, not at the caller's line.
impl Product for Tensor {
    #[track_caller]
    fn product<I: Iterator<Item = Tensor>>(iter: I) -> Tensor {
        fold(iter, 1, BinaryOp::Mul)
    }
}

/// Python's `math.prod(tensors)`, as for an iterator of tensors.
impl<'a> Product<&'a Tensor> for Tensor {
    #[track_caller]
    fn product<I: Iterator<Item = &'a Tensor>>(iter: I) -> Tensor {
        fold(iter, 1, BinaryOp::Mul)
    }
}
