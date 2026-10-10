//! Shapes, strides, dimension arguments and broadcasting.

use std::fmt;
use std::ops::{Deref, DerefMut};

/// Number of dimensions stored inline before spilling to the heap.
const INLINE_CAP: usize = 8;

/// A small vector of `usize` that stores up to [`INLINE_CAP`] values without allocating.
#[derive(Clone)]
enum DimVec {
    Inline { len: u8, data: [usize; INLINE_CAP] },
    Heap(Vec<usize>),
}

impl DimVec {
    fn from_slice(s: &[usize]) -> Self {
        if s.len() <= INLINE_CAP {
            let mut data = [0; INLINE_CAP];
            data[..s.len()].copy_from_slice(s);
            DimVec::Inline {
                len: s.len() as u8,
                data,
            }
        } else {
            DimVec::Heap(s.to_vec())
        }
    }

    fn as_slice(&self) -> &[usize] {
        match self {
            DimVec::Inline { len, data } => &data[..*len as usize],
            DimVec::Heap(v) => v,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [usize] {
        match self {
            DimVec::Inline { len, data } => &mut data[..*len as usize],
            DimVec::Heap(v) => v,
        }
    }

    fn edit(&mut self, f: impl FnOnce(&mut Vec<usize>)) {
        let mut v = self.as_slice().to_vec();
        f(&mut v);
        *self = DimVec::from_slice(&v);
    }
}

macro_rules! dim_vec_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name(DimVec);

        impl $name {
            /// Creates a value from a slice of dimension sizes.
            pub fn new(dims: &[usize]) -> Self {
                $name(DimVec::from_slice(dims))
            }

            /// The values as a slice.
            pub fn as_slice(&self) -> &[usize] {
                self.0.as_slice()
            }

            /// The values as a `Vec`.
            pub fn to_vec(&self) -> Vec<usize> {
                self.as_slice().to_vec()
            }

            pub(crate) fn insert(&mut self, index: usize, value: usize) {
                self.0.edit(|v| v.insert(index, value));
            }

            pub(crate) fn remove(&mut self, index: usize) -> usize {
                let mut out = 0;
                self.0.edit(|v| out = v.remove(index));
                out
            }
        }

        impl Deref for $name {
            type Target = [usize];
            fn deref(&self) -> &[usize] {
                self.0.as_slice()
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut [usize] {
                self.0.as_mut_slice()
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.as_slice() == other.as_slice()
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.as_slice().hash(state);
            }
        }

        impl PartialEq<[usize]> for $name {
            fn eq(&self, other: &[usize]) -> bool {
                self.as_slice() == other
            }
        }

        impl PartialEq<&[usize]> for $name {
            fn eq(&self, other: &&[usize]) -> bool {
                self.as_slice() == *other
            }
        }

        impl<const N: usize> PartialEq<[usize; N]> for $name {
            fn eq(&self, other: &[usize; N]) -> bool {
                self.as_slice() == other
            }
        }

        impl PartialEq<Vec<usize>> for $name {
            fn eq(&self, other: &Vec<usize>) -> bool {
                self.as_slice() == other.as_slice()
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(self.as_slice(), f)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(self.as_slice(), f)
            }
        }

        impl From<&[usize]> for $name {
            fn from(v: &[usize]) -> Self {
                $name::new(v)
            }
        }

        impl<const N: usize> From<[usize; N]> for $name {
            fn from(v: [usize; N]) -> Self {
                $name::new(&v)
            }
        }

        impl From<Vec<usize>> for $name {
            fn from(v: Vec<usize>) -> Self {
                $name::new(&v)
            }
        }

        impl FromIterator<usize> for $name {
            fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
                let v: Vec<usize> = iter.into_iter().collect();
                $name::new(&v)
            }
        }
    };
}

dim_vec_newtype!(
    /// The size of each dimension of a tensor.
    Shape
);

dim_vec_newtype!(
    /// The distance in elements between neighbours along each dimension of a tensor.
    Strides
);

impl Shape {
    /// The shape of a zero-dimensional tensor.
    pub fn scalar() -> Self {
        Shape::new(&[])
    }

    /// Number of dimensions.
    pub fn num_dims(&self) -> usize {
        self.len()
    }

    /// Total number of elements (1 for a zero-dimensional shape).
    ///
    /// # Panics
    ///
    /// Panics if the shape is not representable (see [`checked_num_elements`]).
    pub fn num_elements(&self) -> usize {
        checked_num_elements(self).unwrap_or_else(|| panic!("shape {self:?} has too many elements"))
    }
}

/// The number of elements of `shape`, or `None` if the shape is not representable.
///
/// A shape is representable when the product of its sizes, with zeros counted as ones, is at most
/// `isize::MAX`. Counting zeros as ones means the contiguous strides of an empty shape are
/// representable too, and the check doesn't depend on the order of the dimensions.
pub(crate) fn checked_num_elements(shape: &[usize]) -> Option<usize> {
    let product = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d.max(1)))
        .filter(|&n| n <= isize::MAX as usize)?;
    Some(if shape.contains(&0) { 0 } else { product })
}

/// Types that can be used as a tensor shape: `[usize; N]`, `&[usize]`, `Vec<usize>`, [`Shape`]
/// and `()` (zero-dimensional).
pub trait IntoShape {
    /// Converts into a [`Shape`].
    fn into_shape(self) -> Shape;
}

impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}

impl IntoShape for &Shape {
    fn into_shape(self) -> Shape {
        self.clone()
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Shape {
        Shape::new(&self)
    }
}

impl<const N: usize> IntoShape for &[usize; N] {
    fn into_shape(self) -> Shape {
        Shape::new(self)
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Shape {
        Shape::new(self)
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Shape {
        Shape::new(&self)
    }
}

impl IntoShape for &Vec<usize> {
    fn into_shape(self) -> Shape {
        Shape::new(self)
    }
}

impl IntoShape for () {
    fn into_shape(self) -> Shape {
        Shape::scalar()
    }
}

/// A shape argument that may contain `-1` for one dimension, as taken by `view`, `reshape` and
/// `expand`: `[i64; N]`, `&[i64]`, `Vec<i64>` or a [`Shape`].
pub trait IntoShapeSpec {
    /// Converts into a list of sizes, where `-1` has an operation-specific meaning.
    fn into_shape_spec(self) -> Vec<i64>;
}

impl<const N: usize> IntoShapeSpec for [i64; N] {
    fn into_shape_spec(self) -> Vec<i64> {
        self.to_vec()
    }
}

impl<const N: usize> IntoShapeSpec for &[i64; N] {
    fn into_shape_spec(self) -> Vec<i64> {
        self.to_vec()
    }
}

impl IntoShapeSpec for &[i64] {
    fn into_shape_spec(self) -> Vec<i64> {
        self.to_vec()
    }
}

impl IntoShapeSpec for Vec<i64> {
    fn into_shape_spec(self) -> Vec<i64> {
        self
    }
}

impl IntoShapeSpec for Shape {
    fn into_shape_spec(self) -> Vec<i64> {
        self.iter().map(|&d| d as i64).collect()
    }
}

impl IntoShapeSpec for &Shape {
    fn into_shape_spec(self) -> Vec<i64> {
        self.iter().map(|&d| d as i64).collect()
    }
}

/// One or more dimension indices (negative values count from the end): `i64`, `[i64; N]`,
/// `&[i64]` or `Vec<i64>`.
pub trait Dims {
    /// Converts into a list of dimension indices.
    fn into_dims(self) -> Vec<i64>;
}

impl Dims for i64 {
    fn into_dims(self) -> Vec<i64> {
        vec![self]
    }
}

impl<const N: usize> Dims for [i64; N] {
    fn into_dims(self) -> Vec<i64> {
        self.to_vec()
    }
}

impl<const N: usize> Dims for &[i64; N] {
    fn into_dims(self) -> Vec<i64> {
        self.to_vec()
    }
}

impl Dims for &[i64] {
    fn into_dims(self) -> Vec<i64> {
        self.to_vec()
    }
}

impl Dims for Vec<i64> {
    fn into_dims(self) -> Vec<i64> {
        self
    }
}

/// Wraps a possibly negative dimension index into `0..num_dims`. Zero-dimensional tensors accept
/// `0` and `-1`, like PyTorch.
pub(crate) fn normalize_dim(dim: i64, num_dims: usize) -> usize {
    let n = num_dims.max(1) as i64;
    if dim < -n || dim >= n {
        panic!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})",
            -n,
            n - 1,
            dim
        );
    }
    (if dim < 0 { dim + n } else { dim }) as usize
}

/// Normalizes a list of dimensions into a per-dimension mask. An empty list selects every
/// dimension (PyTorch's behaviour for reductions).
pub(crate) fn dims_mask(dims: &[i64], num_dims: usize) -> Vec<bool> {
    if dims.is_empty() {
        return vec![true; num_dims];
    }
    let mut mask = vec![false; num_dims];
    for &d in dims {
        let d = normalize_dim(d, num_dims);
        if num_dims == 0 {
            continue;
        }
        if mask[d] {
            panic!("dim {d} appears multiple times in the list of dims");
        }
        mask[d] = true;
    }
    mask
}

/// Row-major strides for a contiguous tensor of the given shape.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = vec![0; shape.len()];
    let mut acc: usize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(shape[i].max(1))
            .unwrap_or_else(|| panic!("shape {shape:?} has too many elements"));
    }
    Strides::new(&strides)
}

/// Whether a tensor with this shape and these strides is laid out in row-major order without
/// gaps. Dimensions of size 1 are ignored, as in PyTorch.
pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut expected = 1;
    for i in (0..shape.len()).rev() {
        if shape[i] == 1 {
            continue;
        }
        if strides[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}

/// The broadcast shape of `a` and `b` (NumPy rules).
pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Shape {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i < n - a.len() {
            1
        } else {
            a[i - (n - a.len())]
        };
        let db = if i < n - b.len() {
            1
        } else {
            b[i - (n - b.len())]
        };
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            panic!(
                "The size of tensor a ({da}) must match the size of tensor b ({db}) at \
                 non-singleton dimension {i}"
            );
        };
    }
    Shape::new(&out)
}

/// Resolves a shape spec with at most one `-1` against a number of elements.
pub(crate) fn infer_size(spec: &[i64], num_elements: usize) -> Shape {
    let invalid = || -> ! {
        panic!("shape '{spec:?}' is invalid for input of size {num_elements}");
    };
    let mut infer = None;
    let mut known: usize = 1;
    for (i, &d) in spec.iter().enumerate() {
        if d == -1 {
            if infer.is_some() {
                panic!("only one dimension can be inferred");
            }
            infer = Some(i);
        } else if d < 0 {
            panic!("invalid shape dimension {d}");
        } else {
            known = known.checked_mul(d as usize).unwrap_or_else(|| invalid());
        }
    }
    let mut out: Vec<usize> = spec.iter().map(|&d| d.max(0) as usize).collect();
    match infer {
        Some(i) => {
            if known == 0 || !num_elements.is_multiple_of(known) {
                invalid();
            }
            out[i] = num_elements / known;
        }
        None => {
            if known != num_elements {
                invalid();
            }
        }
    }
    Shape::new(&out)
}

/// Strides that let a tensor with `old_shape`/`old_strides` be viewed as `new_shape` without
/// copying, or `None` if that is impossible. Port of PyTorch's `computeStride`.
pub(crate) fn compute_view_strides(
    old_shape: &[usize],
    old_strides: &[usize],
    new_shape: &[usize],
) -> Option<Strides> {
    if old_shape.is_empty() {
        return Some(Strides::new(&vec![1; new_shape.len()]));
    }
    let numel: usize = old_shape.iter().product();
    if numel == 0 && old_shape == new_shape {
        return Some(Strides::new(old_strides));
    }
    let mut new_strides = vec![0usize; new_shape.len()];
    if numel == 0 {
        for d in (0..new_shape.len()).rev() {
            new_strides[d] = if d == new_shape.len() - 1 {
                1
            } else {
                new_shape[d + 1].max(1).checked_mul(new_strides[d + 1])?
            };
        }
        return Some(Strides::new(&new_strides));
    }

    let mut view_d = new_shape.len() as isize - 1;
    let mut chunk_base_stride = *old_strides.last().unwrap();
    let mut tensor_numel = 1;
    let mut view_numel = 1;
    for tensor_d in (0..old_shape.len()).rev() {
        tensor_numel *= old_shape[tensor_d];
        let chunk_ends = tensor_d == 0
            || old_shape[tensor_d - 1] != 1
                && Some(old_strides[tensor_d - 1]) != tensor_numel.checked_mul(chunk_base_stride);
        if chunk_ends {
            while view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d as usize] == 1) {
                new_strides[view_d as usize] = view_numel.checked_mul(chunk_base_stride)?;
                view_numel *= new_shape[view_d as usize];
                view_d -= 1;
            }
            if view_numel != tensor_numel {
                return None;
            }
            if tensor_d > 0 {
                chunk_base_stride = old_strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if view_d != -1 {
        return None;
    }
    Some(Strides::new(&new_strides))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_vec_inline_and_heap() {
        let mut s = Shape::new(&[1, 2, 3]);
        s.insert(3, 4);
        s.insert(0, 9);
        assert_eq!(s, [9, 1, 2, 3, 4]);
        assert_eq!(s.remove(1), 1);
        let big: Shape = (0..12).collect();
        assert_eq!(big.len(), 12);
        assert_eq!(big[11], 11);
        assert_eq!(format!("{s:?}"), "[9, 2, 3, 4]");
    }

    #[test]
    fn broadcasting() {
        assert_eq!(broadcast_shapes(&[3, 1, 5], &[4, 1]), [3, 4, 5]);
        assert_eq!(broadcast_shapes(&[], &[2]), [2]);
    }

    #[test]
    #[should_panic(expected = "must match the size")]
    fn broadcasting_mismatch() {
        broadcast_shapes(&[3], &[4]);
    }

    #[test]
    fn view_strides() {
        // Contiguous tensors can always be viewed.
        assert_eq!(
            compute_view_strides(&[2, 3, 4], &[12, 4, 1], &[6, 4]).unwrap(),
            [4, 1]
        );
        // A transposed matrix can't be flattened.
        assert!(compute_view_strides(&[3, 2], &[1, 3], &[6]).is_none());
        // ...but size-1 dims can be added anywhere.
        assert_eq!(
            compute_view_strides(&[3, 2], &[1, 3], &[3, 1, 2]).unwrap(),
            [1, 6, 3]
        );
        assert_eq!(infer_size(&[2, -1], 6), [2, 3]);
    }

    #[test]
    fn contiguity() {
        assert!(is_contiguous(&[2, 1, 3], &[3, 100, 1]));
        assert!(!is_contiguous(&[2, 3], &[1, 2]));
        assert_eq!(contiguous_strides(&[2, 0, 3]), [3, 3, 1]);
    }

    #[test]
    fn representable_shapes() {
        assert_eq!(checked_num_elements(&[]), Some(1));
        assert_eq!(checked_num_elements(&[2, 0, 3]), Some(0));
        assert_eq!(checked_num_elements(&[0, isize::MAX as usize]), Some(0));
        assert_eq!(checked_num_elements(&[isize::MAX as usize + 1]), None);
        // Zeros count as ones, so the result doesn't depend on the order of the dimensions.
        assert_eq!(checked_num_elements(&[usize::MAX, 2, 0]), None);
        assert_eq!(checked_num_elements(&[0, usize::MAX, 2]), None);
    }
}
