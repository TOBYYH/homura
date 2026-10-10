//! Matrix multiplication.

use super::common_device;
use crate::backend::cpu;
use crate::shape::broadcast_shapes;
use crate::{Device, Shape, Tensor};

impl Tensor {
    /// Matrix product with the semantics of `torch.matmul`:
    ///
    /// - 1-D × 1-D is a dot product (zero-dimensional result);
    /// - 2-D × 2-D is a matrix product;
    /// - a 1-D operand is treated as a row (left) or column (right) vector, and that dimension is
    ///   removed from the result;
    /// - with more dimensions, the leading (batch) dimensions are broadcast.
    ///
    /// # Panics
    ///
    /// Panics if either operand is zero-dimensional, if the dtypes differ or if the inner
    /// dimensions don't match.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.data(), other.data());
        let device = common_device([&a, &b]);
        let (da, db) = (a.layout.shape.len(), b.layout.shape.len());
        if da == 0 || db == 0 {
            panic!("both arguments to matmul need to be at least 1D, but they are {da}D and {db}D");
        }
        if a.dtype != b.dtype {
            panic!(
                "expected m1 and m2 to have the same dtype, but got: {} != {}",
                a.dtype, b.dtype
            );
        }
        let a2 = if da == 1 {
            self.unsqueeze(0)
        } else {
            self.clone()
        };
        let b2 = if db == 1 {
            other.unsqueeze(-1)
        } else {
            other.clone()
        };
        let (sa, sb) = (a2.shape(), b2.shape());
        let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
        let (k2, n) = (sb[sb.len() - 2], sb[sb.len() - 1]);
        if k != k2 {
            panic!("mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {k2}x{n})");
        }
        let batch = broadcast_shapes(&sa[..sa.len() - 2], &sb[..sb.len() - 2]);
        let batch_size = batch.num_elements();
        let with = |tail: [usize; 2]| -> Vec<i64> {
            batch.iter().chain(tail.iter()).map(|&d| d as i64).collect()
        };
        let a3 = a2
            .expand(with([m, k]))
            .reshape([batch_size as i64, m as i64, k as i64]);
        let b3 = b2
            .expand(with([k, n]))
            .reshape([batch_size as i64, k as i64, n as i64]);
        let out = Tensor::empty_on(Shape::new(&[batch_size, m, n]), a.dtype, device);
        if device == Device::Cpu {
            cpu::matmul(&out.data(), &a3.data(), &b3.data());
        }
        let mut shape = batch.to_vec();
        if da > 1 {
            shape.push(m);
        }
        if db > 1 {
            shape.push(n);
        }
        let shape: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        out.view(shape)
    }
}
