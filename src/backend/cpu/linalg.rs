//! Matrix multiplication.

use super::{base, dispatch_float, dispatch_int, exec};
use crate::Element;
use crate::dtype::sealed::Sealed;
use crate::tensor::TensorData;

/// Batched `out[b] = a[b] @ b[b]` on 3-D views: `a` is `[B, M, K]`, `b` is `[B, K, N]` and `out` is
/// `[B, M, N]`. Any strides are accepted (a stride of 0 broadcasts). Floats accumulate in `f64`,
/// integers in wrapping `i64`.
pub(crate) fn matmul(out: &TensorData, a: &TensorData, b: &TensorData) {
    let dtype = a.dtype;
    if dtype.is_floating_point() {
        dispatch_float!(dtype, T => {
            mm::<T, f64>(out, a, b, 0.0, |acc, x, y| acc + x.to_f64() * y.to_f64(), T::from_f64)
        })
    } else if dtype.is_integral() {
        dispatch_int!(dtype, T => {
            mm::<T, i64>(
                out,
                a,
                b,
                0,
                |acc, x, y| acc.wrapping_add(x.to_i64().wrapping_mul(y.to_i64())),
                T::from_i64,
            )
        })
    } else {
        panic!("matmul is not supported for {dtype} tensors");
    }
}

fn mm<T: Element, A: Copy>(
    out: &TensorData,
    a: &TensorData,
    b: &TensorData,
    zero: A,
    mul_add: impl Fn(A, T, T) -> A,
    finish: impl Fn(A) -> T,
) {
    let (la, lb, lo) = (&a.layout, &b.layout, &out.layout);
    let (batch, m, k) = (la.shape[0], la.shape[1], la.shape[2]);
    let n = lb.shape[2];
    assert_eq!(lb.shape[..2], [batch, k]);
    assert_eq!(lo.shape, [batch, m, n]);
    let (pa, pb, po) = (base::<T>(a), base::<T>(b), base::<T>(out));
    let (sa, sb, so) = (&la.strides, &lb.strides, &lo.strides);
    let mut row = vec![zero; n];
    exec(|| {
        for bi in 0..batch {
            let a0 = la.offset + bi * sa[0];
            let b0 = lb.offset + bi * sb[0];
            let o0 = lo.offset + bi * so[0];
            for i in 0..m {
                row.fill(zero);
                for kk in 0..k {
                    // SAFETY: `bi < batch`, `i < m`, `kk < k` and `j < n` stay inside the layouts.
                    let x = unsafe { *pa.add(a0 + i * sa[1] + kk * sa[2]) };
                    let brow = b0 + kk * sb[1];
                    for (j, acc) in row.iter_mut().enumerate() {
                        // SAFETY: as above.
                        *acc = mul_add(*acc, x, unsafe { *pb.add(brow + j * sb[2]) });
                    }
                }
                for (j, &acc) in row.iter().enumerate() {
                    // SAFETY: as above.
                    unsafe { *po.add(o0 + i * so[1] + j * so[2]) = finish(acc) };
                }
            }
        }
    });
}
