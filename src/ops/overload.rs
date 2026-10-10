//! Arithmetic operator overloads for tensors and scalars.
//!
//! `&a + &b` borrows both operands; `a + b` also works but moves them. Any Rust [`Number`] may
//! appear on the right (`&x * 2.0`, `&x / n` with `n: f32`); on the left only `f64` and `i64`
//! are supported (`1.0 - &x`), so that literals infer without annotations. Compound assignment
//! (`x += &y`) is in place, like Python.

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
            fn $method(self, rhs: &Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
        impl $Trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
        impl $Trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into())
            }
        }
        impl $Trait<Tensor> for Tensor {
            type Output = Tensor;
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
            fn $method(self, rhs: S) -> Tensor {
                binary_op(BinaryOp::$op, self.into(), rhs.into().into())
            }
        }
        impl<S: Number> $Trait<S> for Tensor {
            type Output = Tensor;
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
            fn $method(self, rhs: &Tensor) -> Tensor {
                binary_op(BinaryOp::$op, Scalar::from(self).into(), rhs.into())
            }
        }
        impl $Trait<Tensor> for $s {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                binary_op(BinaryOp::$op, Scalar::from(self).into(), rhs.into())
            }
        }
    )*};
}

macro_rules! assign {
    ($Trait:ident, $method:ident, $inplace:ident) => {
        impl<T: Into<Operand>> $Trait<T> for Tensor {
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
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
