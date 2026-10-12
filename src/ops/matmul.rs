//! Matrix multiplication.

use super::{common_device, reshape_data};
use crate::backend::cpu;
use crate::shape::broadcast_shapes;
use crate::tensor::{Layout, TensorData};
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
        // The operands are rearranged as raw layouts rather than with the view operations, so
        // that nothing but the result is recorded as a view or for autograd.
        let a2 = if da == 1 {
            a.layout.unsqueeze(0)
        } else {
            a.layout.clone()
        };
        let b2 = if db == 1 {
            b.layout.unsqueeze(1)
        } else {
            b.layout.clone()
        };
        let (sa, sb) = (&a2.shape, &b2.shape);
        let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
        let (k2, n) = (sb[sb.len() - 2], sb[sb.len() - 1]);
        if k != k2 {
            panic!("mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {k2}x{n})");
        }
        let batch = broadcast_shapes(&sa[..sa.len() - 2], &sb[..sb.len() - 2]);
        let batch_size = batch.num_elements();
        // Broadcasts the batch dimensions and merges them into one. That copies when they can't
        // share a stride, as with the batch [2, 5] of `[2, 1, 3, 4] @ [1, 5, 4, 6]`, where the
        // left operand has a stride of 0 inside it.
        let batched = |data: &TensorData, layout: &Layout, rows: usize, cols: usize| {
            let full: Shape = batch.iter().copied().chain([rows, cols]).collect();
            let expanded = TensorData {
                layout: layout.expand_to(&full),
                ..data.clone()
            };
            reshape_data(&expanded, Shape::new(&[batch_size, rows, cols]))
        };
        let a3 = batched(&a, &a2, m, k);
        let b3 = batched(&b, &b2, k, n);
        let out = Tensor::empty_on(Shape::new(&[batch_size, m, n]), a.dtype, device).data();
        if device == Device::Cpu {
            cpu::matmul(&out, &a3, &b3);
        }
        let mut shape = batch.to_vec();
        if da > 1 {
            shape.push(m);
        }
        if db > 1 {
            shape.push(n);
        }
        // The result is contiguous, so this never copies.
        Tensor::from_data(reshape_data(&out, Shape::new(&shape)))
    }
}
