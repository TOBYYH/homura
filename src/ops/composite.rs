//! Operations defined in terms of other operations, plus softmax.

use super::{Operand, common_device};
use crate::backend::cpu;
use crate::dtype::Kind;
use crate::shape::normalize_dim;
use crate::{DType, Device, Scalar, Tensor};

fn softmax_op(x: &Tensor, dim: i64, log: bool) -> Tensor {
    let src = x.data();
    if !src.dtype.is_floating_point() {
        panic!(
            "{} is not implemented for {} tensors",
            if log { "log_softmax" } else { "softmax" },
            src.dtype
        );
    }
    let dim = normalize_dim(dim, src.layout.shape.len());
    let out = Tensor::empty_on(src.layout.shape.clone(), src.dtype, src.device());
    if src.device() == Device::Cpu {
        cpu::softmax(&out.data(), &src, dim, log);
    }
    let name = if log { "log_softmax" } else { "softmax" };
    crate::autograd::record_unimplemented(name, &[Some(x)], &out);
    out
}

/// How an `add*` operation treats floating point coefficients for integer or bool inputs, following
/// the PyTorch kernels.
#[derive(Clone, Copy, PartialEq)]
enum Coefficients {
    /// `addmm`, `addmv`, `baddbmm` and `addbmm` truncate them toward zero.
    Truncate,
    /// `addr` rejects them, as well as bool coefficients for non-bool inputs.
    Strict,
}

/// Converts the coefficient `name` to a scalar of `dtype`, so that unlike a scalar passed to
/// `mul` it never changes the result dtype.
///
/// # Panics
///
/// Panics if the value doesn't fit in `dtype`. For floating point dtypes only `addr` checks this,
/// and lets infinities and NaN through.
fn coefficient(s: Scalar, dtype: DType, name: &str, rule: Coefficients) -> Scalar {
    if rule == Coefficients::Strict {
        if matches!(s, Scalar::Bool(_)) && dtype != DType::Bool {
            panic!("Boolean {name} only supported for Boolean results");
        }
        if matches!(s, Scalar::Float(_)) && !dtype.is_floating_point() {
            panic!(
                "For integral input tensors, argument {name} must not be a floating point number"
            );
        }
    }
    let overflow = || -> ! { panic!("{name}={s} cannot be converted to {dtype} without overflow") };
    match dtype.kind() {
        Kind::Bool => Scalar::Bool(s.to_bool()),
        Kind::Float => {
            let v = s.to_f64();
            if rule == Coefficients::Strict && v.is_finite() && v.abs() > dtype.float_max() {
                overflow();
            }
            s
        }
        Kind::Int => {
            let (min, max) = dtype.int_range();
            Scalar::Int(match s {
                Scalar::Bool(v) => v as i64,
                Scalar::Int(v) => {
                    // Like PyTorch, negative integers down to `-255` wrap around for `U8`.
                    let min = if dtype == DType::U8 { -max } else { min };
                    if !(min..=max).contains(&v) {
                        overflow();
                    }
                    v
                }
                Scalar::Float(v) => {
                    // Check before truncating to preserve the lower-bound rule. Fractional
                    // values above MAX still truncate to MAX; MAX + 1 is the exclusive upper
                    // bound. Compute it in i128 so I64 uses exactly 2^63, not a rounded MAX.
                    let upper = (max as i128 + 1) as f64;
                    if !(min as f64..upper).contains(&v) {
                        overflow();
                    }
                    v as i64
                }
            })
        }
    }
}

/// `beta * input + alpha * product` with the dtype of `product`, where the values of `input` are
/// ignored when `beta` is zero (so NaN in it does not propagate), as PyTorch's `add*` operations
/// do. `input` must still have the dtype and device of `product` and broadcast to its shape,
/// whatever `beta` is. The coefficients are converted to the dtype first, see [`coefficient`].
///
/// `op` names the operation for autograd: with `beta` zero, `input` still needs a zero gradient,
/// which isn't implemented yet.
fn scaled_sum(
    op: &'static str,
    input: &Tensor,
    product: Tensor,
    beta: Scalar,
    alpha: Scalar,
    rule: Coefficients,
) -> Tensor {
    let (i, p) = (input.data(), product.data());
    common_device([&i, &p]);
    if i.dtype != p.dtype {
        panic!(
            "self and mat2 must have the same dtype, but got {} and {}",
            i.dtype, p.dtype
        );
    }
    i.layout.expand_to(&p.layout.shape);
    let beta = coefficient(beta, p.dtype, "beta", rule);
    let alpha = coefficient(alpha, p.dtype, "alpha", rule);
    let product = if alpha.to_f64() == 1.0 {
        product
    } else {
        product.mul(alpha)
    };
    if beta.to_f64() == 0.0 {
        if crate::autograd::needs_grad(&[Some(input)]) {
            // The value ignores `input`, but its gradient must be zeros rather than missing.
            let out = Tensor::from_data(product.data());
            crate::autograd::record_unimplemented(op, &[Some(input), Some(&product)], &out);
            return out;
        }
        return product;
    }
    let scaled = if beta.to_f64() == 1.0 {
        input.clone()
    } else {
        input.mul(beta)
    };
    scaled.add(&product)
}

fn check_dims(t: &Tensor, n: usize, what: &str) {
    if t.num_dims() != n {
        panic!(
            "{what} must be a {n}-D tensor, got {}-D tensor",
            t.num_dims()
        );
    }
}

impl Tensor {
    /// Softmax along `dim`, computed stably by subtracting the maximum first.
    pub fn softmax(&self, dim: i64) -> Tensor {
        softmax_op(self, dim, false)
    }

    /// Logarithm of the softmax along `dim`, computed stably.
    pub fn log_softmax(&self, dim: i64) -> Tensor {
        softmax_op(self, dim, true)
    }

    /// `beta * self + alpha * (mat1 @ mat2)` for 2-D `mat1` and `mat2`; `self` is broadcast to the
    /// result.
    ///
    /// The result keeps the input dtype: for integer inputs, floating point coefficients are
    /// truncated toward zero, as in PyTorch. This also holds for [`addmv`](Tensor::addmv),
    /// [`baddbmm`](Tensor::baddbmm) and [`addbmm`](Tensor::addbmm).
    ///
    /// # Panics
    ///
    /// Panics if a coefficient doesn't fit in an integer input dtype.
    pub fn addmm(
        &self,
        mat1: &Tensor,
        mat2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(mat1, 2, "mat1");
        check_dims(mat2, 2, "mat2");
        scaled_sum(
            "addmm",
            self,
            mat1.matmul(mat2),
            beta.into(),
            alpha.into(),
            Coefficients::Truncate,
        )
    }

    /// `beta * self + alpha * (mat @ vec)` for a 2-D `mat` and a 1-D `vec`.
    pub fn addmv(
        &self,
        mat: &Tensor,
        vec: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(mat, 2, "mat");
        check_dims(vec, 1, "vec");
        scaled_sum(
            "addmv",
            self,
            mat.matmul(vec),
            beta.into(),
            alpha.into(),
            Coefficients::Truncate,
        )
    }

    /// `beta * self + alpha * outer(vec1, vec2)` for 1-D `vec1` and `vec2`.
    /// The three tensor inputs are promoted to a common dtype before computation.
    ///
    /// # Panics
    ///
    /// Panics if a coefficient doesn't fit in the result dtype, is a floating point number for
    /// integer or bool inputs, or is a bool for non-bool inputs, as PyTorch does.
    pub fn addr(
        &self,
        vec1: &Tensor,
        vec2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(vec1, 1, "vec1");
        check_dims(vec2, 1, "vec2");
        let inputs = [self.data(), vec1.data(), vec2.data()];
        common_device(&inputs);
        // `self` participates fully in promotion even when it is zero-dimensional, since addr
        // broadcasts it to the matrix shape before computing the result type.
        let dtype = crate::promote_types(
            crate::promote_types(inputs[0].dtype, inputs[1].dtype),
            inputs[2].dtype,
        );
        let input = self.to(dtype);
        // A broadcast product rather than a matmul, which doesn't support bool.
        // Cast before multiplication so a wider self also widens the product computation.
        let outer = vec1.to(dtype).unsqueeze(1).mul(vec2.to(dtype).unsqueeze(0));
        scaled_sum(
            "addr",
            &input,
            outer,
            beta.into(),
            alpha.into(),
            Coefficients::Strict,
        )
    }

    /// `beta * self + alpha * (batch1 @ batch2)` for 3-D `batch1` and `batch2`.
    pub fn baddbmm(
        &self,
        batch1: &Tensor,
        batch2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(batch1, 3, "batch1");
        check_dims(batch2, 3, "batch2");
        if batch1.size(0) != batch2.size(0) {
            panic!(
                "batch1 and batch2 must have same number of batches, got {} and {}",
                batch1.size(0),
                batch2.size(0)
            );
        }
        scaled_sum(
            "baddbmm",
            self,
            batch1.matmul(batch2),
            beta.into(),
            alpha.into(),
            Coefficients::Truncate,
        )
    }

    /// `beta * self + alpha * sum_i (batch1[i] @ batch2[i])` for 3-D `batch1` and `batch2`.
    pub fn addbmm(
        &self,
        batch1: &Tensor,
        batch2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(batch1, 3, "batch1");
        check_dims(batch2, 3, "batch2");
        if batch1.size(0) != batch2.size(0) {
            panic!(
                "batch1 and batch2 must have same number of batches, got {} and {}",
                batch1.size(0),
                batch2.size(0)
            );
        }
        // Integer sums are computed in `I64`; casting back wraps exactly like accumulating in the
        // input dtype would.
        let product = batch1.matmul(batch2).sum_dim(0, false).to(batch1.dtype());
        scaled_sum(
            "addbmm",
            self,
            product,
            beta.into(),
            alpha.into(),
            Coefficients::Truncate,
        )
    }

    /// `self + value * tensor1 * tensor2`.
    pub fn addcmul(&self, tensor1: &Tensor, tensor2: &Tensor, value: impl Into<Scalar>) -> Tensor {
        self.add(tensor1.mul(value.into()).mul(tensor2))
    }

    /// `self + value * tensor1 / tensor2`.
    ///
    /// # Panics
    ///
    /// Panics if both `tensor1` and `tensor2` are integer tensors, as PyTorch does.
    pub fn addcdiv(&self, tensor1: &Tensor, tensor2: &Tensor, value: impl Into<Scalar>) -> Tensor {
        if !tensor1.is_floating_point() && !tensor2.is_floating_point() {
            panic!(
                "Integer division with addcdiv is no longer supported. Use \
                 self.add(tensor1.div(tensor2).mul(value)) instead."
            );
        }
        self.add(tensor1.mul(value.into()).div(tensor2))
    }

    /// Linear interpolation `self + weight * (end - self)`, with `weight` a tensor or a scalar.
    /// Uses PyTorch's formula, which is exact at both ends.
    pub fn lerp(&self, end: &Tensor, weight: impl Into<Operand>) -> Tensor {
        let diff = end.sub(self);
        match weight.into() {
            Operand::Scalar(w) => {
                let wf = w.to_f64();
                if wf.abs() < 0.5 {
                    self.add(diff.mul(w))
                } else {
                    end.sub(diff.mul(1.0 - wf))
                }
            }
            Operand::Tensor(w) => {
                let low = self.add(w.mul(&diff));
                let high = end.sub(diff.mul(w.neg().add(1.0)));
                crate::where_(&w.abs().less(0.5), &low, &high)
            }
        }
    }

    /// In-place [`addcmul`](Tensor::addcmul).
    pub fn addcmul_(&self, tensor1: &Tensor, tensor2: &Tensor, value: impl Into<Scalar>) -> &Self {
        crate::autograd::check_inplace(self, &[Some(tensor1), Some(tensor2)]);
        let result = self.addcmul(tensor1, tensor2, value);
        self.assign_result(&result);
        self
    }

    /// In-place [`addcdiv`](Tensor::addcdiv).
    pub fn addcdiv_(&self, tensor1: &Tensor, tensor2: &Tensor, value: impl Into<Scalar>) -> &Self {
        crate::autograd::check_inplace(self, &[Some(tensor1), Some(tensor2)]);
        let result = self.addcdiv(tensor1, tensor2, value);
        self.assign_result(&result);
        self
    }

    /// In-place [`lerp`](Tensor::lerp).
    pub fn lerp_(&self, end: &Tensor, weight: impl Into<Operand>) -> &Self {
        let weight = weight.into();
        let weight_tensor = match &weight {
            Operand::Tensor(w) => Some(w),
            Operand::Scalar(_) => None,
        };
        crate::autograd::check_inplace(self, &[Some(end), weight_tensor]);
        let result = self.lerp(end, weight);
        self.assign_result(&result);
        self
    }
}
