//! Strided iteration over several tensors of the same shape.

use crate::Strides;
use crate::tensor::Layout;

/// The strides to walk `l` with: its own, or zeros if it has no elements. An empty layout touches
/// no memory, so its strides may be arbitrary (see `as_strided`); kernels that walk only part of
/// its shape, such as the kept dimensions of a reduction, mustn't do offset arithmetic with them.
pub(crate) fn walk_strides(l: &Layout) -> Strides {
    if l.num_elements() == 0 {
        Strides::new(&vec![0; l.strides.len()])
    } else {
        l.strides.clone()
    }
}

/// A precomputed loop nest over a shape, tracking one element offset per operand.
///
/// Size-1 dimensions are dropped and neighbouring dimensions that are contiguous with each other
/// in every operand are merged, so contiguous tensors run as a single flat loop.
pub(crate) struct Plan<const N: usize> {
    /// `(size, stride per operand)` from the outermost to the innermost dimension.
    dims: Vec<(usize, [usize; N])>,
    empty: bool,
}

impl<const N: usize> Plan<N> {
    pub fn new(shape: &[usize], strides: [&[usize]; N]) -> Self {
        for s in &strides {
            debug_assert_eq!(s.len(), shape.len());
        }
        // Empty layouts touch no memory, so their strides may be arbitrary: don't look at them.
        if shape.contains(&0) {
            return Plan {
                dims: vec![],
                empty: true,
            };
        }
        let mut dims: Vec<(usize, [usize; N])> = Vec::with_capacity(shape.len());
        for (i, &size) in shape.iter().enumerate() {
            if size == 1 {
                continue;
            }
            let st: [usize; N] = std::array::from_fn(|k| strides[k][i]);
            if let Some(outer) = dims.last_mut()
                && (0..N).all(|k| st[k].checked_mul(size) == Some(outer.1[k]))
            {
                outer.0 *= size;
                outer.1 = st;
                continue;
            }
            dims.push((size, st));
        }
        Plan { dims, empty: false }
    }

    /// Calls `f` with the offsets of every element, in row-major order, starting from `base`.
    #[inline]
    pub fn for_each(&self, base: [usize; N], mut f: impl FnMut([usize; N])) {
        if self.empty {
            return;
        }
        if self.dims.is_empty() {
            f(base);
        } else {
            recurse(&self.dims, base, &mut f);
        }
    }
}

#[inline]
fn recurse<const N: usize>(
    dims: &[(usize, [usize; N])],
    base: [usize; N],
    f: &mut impl FnMut([usize; N]),
) {
    let (size, strides) = dims[0];
    let mut off = base;
    if dims.len() == 1 {
        for _ in 0..size {
            f(off);
            for k in 0..N {
                off[k] += strides[k];
            }
        }
    } else {
        for _ in 0..size {
            recurse(&dims[1..], off, f);
            for k in 0..N {
                off[k] += strides[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect<const N: usize>(shape: &[usize], strides: [&[usize]; N]) -> Vec<[usize; N]> {
        let mut out = vec![];
        Plan::new(shape, strides).for_each([0; N], |o| out.push(o));
        out
    }

    #[test]
    fn contiguous_and_transposed() {
        let c = collect(&[2, 3], [&[3, 1], &[1, 2]]);
        assert_eq!(c, vec![[0, 0], [1, 2], [2, 4], [3, 1], [4, 3], [5, 5]]);
        // Fully contiguous operands merge into one dimension.
        assert_eq!(Plan::new(&[2, 3, 4], [&[12, 4, 1]]).dims.len(), 1);
        assert_eq!(collect::<1>(&[2, 0], [&[1, 1]]).len(), 0);
        assert_eq!(collect::<1>(&[], [&[]]), vec![[0]]);
        assert_eq!(collect(&[3], [&[0]]), vec![[0], [0], [0]]);
    }

    #[test]
    fn extreme_strides_do_not_overflow() {
        assert!(collect::<1>(&[0, 2], [&[0, usize::MAX]]).is_empty());
        // `stride * size` doesn't fit in `usize`, so the dimensions can't be merged.
        let half = usize::MAX / 2 + 1;
        assert_eq!(Plan::new(&[2, 2], [&[0, half]]).dims.len(), 2);
    }
}
