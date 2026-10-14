//! Operations defined in terms of other operations, plus softmax.

use super::{Operand, common_device, scalar_data};
use crate::autograd::formulas::ScaledProduct;
use crate::backend::cpu;
use crate::dtype::Kind;
use crate::shape::normalize_dim;
use crate::{DType, Device, Scalar, Shape, Tensor};

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
    crate::autograd::formulas::softmax(x, dim, log, &out);
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

/// Checks the `input` of an `add*` operation against its result of `shape`, which has the dtype
/// and device of `factor`, before anything is computed: `input` must have that dtype and device
/// and broadcast to the shape, whatever `beta` is. Returns the coefficients converted to the
/// dtype, see [`coefficient`].
fn check_input(
    input: &Tensor,
    factor: &Tensor,
    shape: &Shape,
    (beta, alpha): (Scalar, Scalar),
    rule: Coefficients,
) -> (Scalar, Scalar) {
    let (i, f) = (input.data(), factor.data());
    common_device([&i, &f]);
    if i.dtype != f.dtype {
        panic!(
            "self and mat2 must have the same dtype, but got {} and {}",
            i.dtype, f.dtype
        );
    }
    i.layout.expand_to(shape);
    (
        coefficient(beta, f.dtype, "beta", rule),
        coefficient(alpha, f.dtype, "alpha", rule),
    )
}

/// `t * s`, skipped when `s` is one.
fn scale(t: Tensor, s: Scalar) -> Tensor {
    if s.to_f64() == 1.0 { t } else { t.mul(s) }
}

/// `beta * input + alpha * product`, where the values of `input` are ignored when `beta` is zero
/// (so NaN in it does not propagate), as PyTorch's `add*` operations do.
fn scaled_sum(input: &Tensor, product: Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    let product = scale(product, alpha);
    if beta.to_f64() == 0.0 {
        return product;
    }
    scale(input.clone(), beta).add(&product)
}

/// The coefficient `name` converted to the floating point type `ty` that a kernel takes it in,
/// as PyTorch's checked conversion does: rounded to `ty` whatever kind of scalar it is, and
/// rejected if it is finite but out of the range of `ty` (infinities and NaN pass).
///
/// # Panics
///
/// Panics if the value doesn't fit in `ty`.
fn converted(s: Scalar, ty: DType, name: &str) -> Scalar {
    let v = s.to_f64();
    if v.is_finite() && v.abs() > ty.float_max() {
        panic!("{name}={s} cannot be converted to {ty} without overflow");
    }
    let t = Tensor::from_data(scalar_data(s, ty, Device::Cpu));
    Scalar::Float(t.to(DType::F64).item())
}

/// The CPU kernel PyTorch computes a matrix product scaled by `alpha` with.
#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    /// `addmm`, `baddbmm` and `addbmm`: BLAS `gemm`, or for half precision a product
    /// accumulated in f32 and combined with the coefficients in f32 before rounding once. The
    /// coefficients are converted to f32 (`F64` for `F64`).
    Gemm,
    /// `addmv`: BLAS `gemv`, or for half precision the same. The coefficients are converted
    /// to the dtype.
    Gemv,
}

/// `beta * input + alpha * product(x, y)` for the matrix products, with the dtype of the factors
/// and `product` computing the matrix product itself.
///
/// The factors are checked before anything is computed, by computing `product` on metadata only,
/// so that the checks and messages don't depend on the coefficients. The value follows the CPU
/// kernels PyTorch dispatches to, so that a zero coefficient doesn't produce NaN where they
/// don't:
///
/// - `F32` and `F64`: BLAS doesn't read the factors when `alpha` is zero, so neither do these
///   operations; the result is `beta * input` (zeros for a zero `beta`), even if the factors
///   hold infinities or their product overflows. PyTorch's `baddbmm` computes products of fewer
///   than 400 elements (`m * k * n`) with a kernel of its own instead, where an overflowing
///   product times zero is NaN; homura follows BLAS at every size, which is also what GPU
///   libraries do.
/// - `F16` and `BF16`: the product is accumulated in f32 and scaled there, then rounded once, so
///   it overflows to NaN only if it overflows f32. This is PyTorch's `gemv` for a row-major
///   matrix; for a column-major one (including `[m, 1]`) it scales the vector first and rounds
///   `beta * input` separately, and its `baddbmm` of 400 or more elements with several batches
///   drops `beta` when `alpha` is zero. homura's values don't depend on the layout and don't
///   follow either.
/// - integers: the product in the dtype, scaled with wrapping arithmetic.
fn scaled_matrix_product(
    input: &Tensor,
    (x, y): (&Tensor, &Tensor),
    coefficients: (Scalar, Scalar),
    kernel: Kernel,
    product: impl Fn(&Tensor, &Tensor) -> Tensor,
) -> Tensor {
    let device = common_device([&x.data(), &y.data()]);
    let meta = |t: &Tensor| Tensor::empty_on(t.shape(), t.dtype(), Device::Meta);
    let shape = product(&meta(x), &meta(y)).shape();
    let dtype = x.dtype();
    let (beta, alpha) = check_input(input, x, &shape, coefficients, Coefficients::Truncate);
    let (beta, alpha) = if dtype.is_floating_point() {
        // `gemm` takes the coefficients in its accumulation type, `gemv` in the dtype.
        let ty = match kernel {
            Kernel::Gemm if dtype == DType::F64 => DType::F64,
            Kernel::Gemm => DType::F32,
            Kernel::Gemv => dtype,
        };
        (converted(beta, ty, "beta"), converted(alpha, ty, "alpha"))
    } else {
        (beta, alpha)
    };
    match dtype {
        DType::F32 | DType::F64 if alpha.to_f64() == 0.0 => {
            if beta.to_f64() == 0.0 {
                crate::zeros_with(shape, (dtype, device))
            } else {
                scale(input.clone(), beta).expand(&shape).deep_clone()
            }
        }
        DType::F16 | DType::BF16 => {
            let p = product(&x.to(DType::F32), &y.to(DType::F32));
            scaled_sum(&input.to(DType::F32), p, beta, alpha).to(dtype)
        }
        _ => scaled_sum(input, product(x, y), beta, alpha),
    }
}

/// `addr` of inputs already promoted to one dtype.
///
/// Floating point values follow PyTorch's elementwise kernel, `beta * input + alpha * vec1 *
/// vec2` evaluated left to right in the dtype with the coefficients rounded to it: `alpha`
/// scales `vec1` before the outer product, so a zero or tiny `alpha` doesn't turn a finite
/// result into NaN or infinity. Integers and bools compute the outer product first, which gives
/// the same values; a broadcast product rather than a matmul, which doesn't support bool.
fn addr_value(input: &Tensor, v1: &Tensor, v2: &Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    let dtype = v1.dtype();
    let shape = Shape::new(&[v1.size(0), v2.size(0)]);
    let (beta, alpha) = check_input(input, v1, &shape, (beta, alpha), Coefficients::Strict);
    let outer = |v1: Tensor| v1.unsqueeze(1).mul(v2.unsqueeze(0));
    if !dtype.is_floating_point() {
        return scaled_sum(input, outer(v1.clone()), beta, alpha);
    }
    let (beta, alpha) = (
        converted(beta, dtype, "beta"),
        converted(alpha, dtype, "alpha"),
    );
    let product = outer(scale(v1.clone(), alpha));
    if beta.to_f64() == 0.0 {
        return product;
    }
    scale(input.clone(), beta).add(&product)
}

/// Computes `beta * input + alpha * product(x, y)` with `forward` in no-grad mode and records
/// the node of the scaled product.
fn scaled_product(
    product: ScaledProduct,
    input: &Tensor,
    x: &Tensor,
    y: &Tensor,
    beta: Scalar,
    alpha: Scalar,
    forward: impl FnOnce() -> Tensor,
) -> Tensor {
    let out = {
        let _guard = crate::no_grad();
        forward()
    };
    crate::autograd::formulas::scaled_product(product, input, x, y, beta, alpha, &out);
    out
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
    /// truncated toward zero, as in PyTorch. For `F32` and `F64`, a zero `alpha` means that the
    /// factors aren't read, as in BLAS, so infinities in them or an overflowing product don't
    /// make the result NaN; `F16` and `BF16` products are accumulated and scaled in f32 before
    /// rounding once. All this also holds for [`addmv`](Tensor::addmv),
    /// [`baddbmm`](Tensor::baddbmm) and [`addbmm`](Tensor::addbmm).
    ///
    /// # Panics
    ///
    /// Panics if a coefficient doesn't fit in an integer input dtype, or is finite and doesn't
    /// fit in the type the coefficients are taken in for floating point inputs, as PyTorch's
    /// kernels do: f32 for `F16`, `BF16` and `F32` (`F64` for `F64`) in `addmm`, `baddbmm` and
    /// `addbmm`, the input dtype itself in `addmv`.
    pub fn addmm(
        &self,
        mat1: &Tensor,
        mat2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(mat1, 2, "mat1");
        check_dims(mat2, 2, "mat2");
        let (beta, alpha) = (beta.into(), alpha.into());
        scaled_product(ScaledProduct::Mm, self, mat1, mat2, beta, alpha, || {
            scaled_matrix_product(self, (mat1, mat2), (beta, alpha), Kernel::Gemm, |a, b| {
                a.matmul(b)
            })
        })
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
        let (beta, alpha) = (beta.into(), alpha.into());
        scaled_product(ScaledProduct::Mv, self, mat, vec, beta, alpha, || {
            scaled_matrix_product(self, (mat, vec), (beta, alpha), Kernel::Gemv, |a, b| {
                a.matmul(b)
            })
        })
    }

    /// `beta * self + alpha * outer(vec1, vec2)` for 1-D `vec1` and `vec2`.
    /// The three tensor inputs are promoted to a common dtype before computation. Floating point
    /// values are computed like PyTorch's kernel, `beta * self + (alpha * vec1) * vec2` in the
    /// dtype, so a zero or tiny `alpha` doesn't make a finite result NaN or infinite.
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
        // The conversions are recorded, so that each input gets its gradient in its own dtype
        // (PyTorch's backward rejects mixed dtypes).
        let (input, v1, v2) = (self.to(dtype), vec1.to(dtype), vec2.to(dtype));
        let (beta, alpha) = (beta.into(), alpha.into());
        // Cast before multiplication so a wider self also widens the product computation.
        scaled_product(ScaledProduct::Outer, &input, &v1, &v2, beta, alpha, || {
            addr_value(&input, &v1, &v2, beta, alpha)
        })
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
        let (beta, alpha) = (beta.into(), alpha.into());
        scaled_product(
            ScaledProduct::Bmm,
            self,
            batch1,
            batch2,
            beta,
            alpha,
            || {
                scaled_matrix_product(
                    self,
                    (batch1, batch2),
                    (beta, alpha),
                    Kernel::Gemm,
                    |a, b| a.matmul(b),
                )
            },
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
        let (beta, alpha) = (beta.into(), alpha.into());
        scaled_product(
            ScaledProduct::SumBmm,
            self,
            batch1,
            batch2,
            beta,
            alpha,
            || {
                scaled_matrix_product(
                    self,
                    (batch1, batch2),
                    (beta, alpha),
                    Kernel::Gemm,
                    |a, b| {
                        // Integer sums are computed in `I64`; casting back wraps exactly like
                        // accumulating in the input dtype would.
                        a.matmul(b).sum_dim(0, false).to(a.dtype())
                    },
                )
            },
        )
    }

    /// `self + value * tensor1 * tensor2`.
    pub fn addcmul(&self, tensor1: &Tensor, tensor2: &Tensor, value: impl Into<Scalar>) -> Tensor {
        let value = value.into();
        let out = {
            let _guard = crate::no_grad();
            self.add(tensor1.mul(value).mul(tensor2))
        };
        crate::autograd::formulas::addcmul(self, tensor1, tensor2, value, &out);
        out
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
        let value = value.into();
        let out = {
            let _guard = crate::no_grad();
            self.add(tensor1.mul(value).div(tensor2))
        };
        crate::autograd::formulas::addcdiv(self, tensor1, tensor2, value, &out);
        out
    }

    /// Linear interpolation `self + weight * (end - self)`, with `weight` a tensor or a scalar.
    /// Uses PyTorch's formula, which is exact at both ends.
    pub fn lerp(&self, end: &Tensor, weight: impl Into<Operand>) -> Tensor {
        let weight = weight.into();
        let out = {
            let _guard = crate::no_grad();
            let diff = end.sub(self);
            match &weight {
                Operand::Scalar(w) => {
                    let wf = w.to_f64();
                    if wf.abs() < 0.5 {
                        self.add(diff.mul(*w))
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
        };
        crate::autograd::formulas::lerp(self, end, &weight, &out);
        out
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
