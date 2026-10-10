//! Operations defined in terms of other operations, plus softmax.

use super::Operand;
use crate::backend::cpu;
use crate::shape::normalize_dim;
use crate::{Device, Scalar, Tensor};

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
    out
}

/// `beta * input + alpha * product`, where `input` is ignored entirely when `beta` is zero (so
/// NaN in it does not propagate), as PyTorch's `add*` operations do.
fn scaled_sum(input: &Tensor, product: Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    let product = if alpha.to_f64() == 1.0 {
        product
    } else {
        product.mul(alpha)
    };
    if beta.to_f64() == 0.0 {
        // Still check that `input` broadcasts to the result.
        input.layout().expand_to(&product.shape());
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
    pub fn addmm(
        &self,
        mat1: &Tensor,
        mat2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(mat1, 2, "mat1");
        check_dims(mat2, 2, "mat2");
        scaled_sum(self, mat1.matmul(mat2), beta.into(), alpha.into())
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
        scaled_sum(self, mat.matmul(vec), beta.into(), alpha.into())
    }

    /// `beta * self + alpha * outer(vec1, vec2)` for 1-D `vec1` and `vec2`.
    pub fn addr(
        &self,
        vec1: &Tensor,
        vec2: &Tensor,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Tensor {
        check_dims(vec1, 1, "vec1");
        check_dims(vec2, 1, "vec2");
        let outer = vec1.unsqueeze(1).matmul(&vec2.unsqueeze(0));
        scaled_sum(self, outer, beta.into(), alpha.into())
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
        scaled_sum(self, batch1.matmul(batch2), beta.into(), alpha.into())
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
        let product = batch1.matmul(batch2).sum_dim(0, false);
        scaled_sum(self, product, beta.into(), alpha.into())
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
        let result = self.addcmul(tensor1, tensor2, value);
        self.assign_result(&result);
        self
    }

    /// In-place [`addcdiv`](Tensor::addcdiv).
    pub fn addcdiv_(&self, tensor1: &Tensor, tensor2: &Tensor, value: impl Into<Scalar>) -> &Self {
        let result = self.addcdiv(tensor1, tensor2, value);
        self.assign_result(&result);
        self
    }

    /// In-place [`lerp`](Tensor::lerp).
    pub fn lerp_(&self, end: &Tensor, weight: impl Into<Operand>) -> &Self {
        let result = self.lerp(end, weight);
        self.assign_result(&result);
        self
    }
}
