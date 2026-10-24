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
            #[track_caller]
            pub fn new(dims: &[usize]) -> Self {
                $name(DimVec::from_slice(dims))
            }

            /// The values as a slice.
            #[track_caller]
            pub fn as_slice(&self) -> &[usize] {
                self.0.as_slice()
            }

            /// The values as a `Vec`.
            #[track_caller]
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
    #[track_caller]
    pub fn scalar() -> Self {
        Shape::new(&[])
    }

    /// Number of dimensions.
    #[track_caller]
    pub fn num_dims(&self) -> usize {
        self.len()
    }

    /// Total number of elements (1 for a zero-dimensional shape).
    ///
    /// # Panics
    ///
    /// Panics if the shape is not representable: the product of its sizes, with zeros counted as
    /// ones, must be at most `isize::MAX`.
    #[track_caller]
    pub fn num_elements(&self) -> usize {
        match checked_num_elements(self) {
            Some(n) => n,
            None => panic!("shape {self:?} has too many elements"),
        }
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
    #[track_caller]
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

pub(crate) mod sealed {
    /// The conversion behind [`DimSize`](super::DimSize). It isn't documented and no type
    /// outside homura can implement it, though generic code with a `DimSize` bound can call it.
    pub trait Sealed: Copy {
        /// Converts to `i64`.
        ///
        /// # Panics
        ///
        /// Panics if the value is above `i64::MAX`.
        #[track_caller]
        fn to_dim(self) -> i64;
    }
}

use sealed::Sealed;

/// The integer types of sizes in a shape argument, dimension indices and index items: `i64`,
/// `i32` and `usize`. This trait is sealed.
///
/// `usize` values (from [`Tensor::size`](crate::Tensor::size), [`Shape`] or a length) can be
/// used directly, and so can `i64` ones, which may be negative. `i32` is there for literals:
/// with several integer types to choose from, Rust gives an unsuffixed literal the type `i32`,
/// so `view([2, -1])` and `sum_dim(1, true)` still compile. Mixing types in one argument takes a
/// tuple, such as `view((batch, seq, -1))` with `batch` and `seq` of type `usize`.
///
/// A `usize` above `i64::MAX` panics with "dimension or index … is too large".
///
/// ```
/// use homura::Shape;
/// let x = homura::zeros([2, 3, 4]);
/// let n: usize = x.size(-1);
/// let d: i64 = -1;
/// assert_eq!(x.view([6, -1]).shape(), [6, 4]);
/// assert_eq!(x.view([6, n]).shape(), [6, 4]);
/// assert_eq!(x.view([d, 4]).shape(), [6, 4]);
/// assert_eq!(x.view((x.size(0), 3i32, d)).shape(), [2, 3, 4]);
/// assert_eq!(x.sum_dim(x.num_dims() - 1, true).shape(), [2, 3, 1]);
/// assert_eq!(x.index((.., .., 1..n)).shape(), [2, 3, 3]);
/// let s = x.sum();
/// assert_eq!(s.view(()).shape(), []);
/// assert_eq!(s.view(Shape::scalar()).shape(), []);
/// assert_eq!(s.sum_dim([] as [i64; 0], false).shape(), []);
/// assert_eq!(homura::zeros([]).shape(), []);
/// ```
///
/// # Empty lists
///
/// An empty list holds no literal to infer the integer type from, so `view([])`, `view(&[])`,
/// `view(Vec::new())` and `sum_dim([], false)` don't compile (since 0.4.0). Write `view(())` or
/// `view(Shape::scalar())` for a zero-dimensional shape and `[] as [i64; 0]` for no dimensions.
/// [`zeros([])`](crate::zeros) and the other functions that take an [`IntoShape`] are not
/// affected.
///
/// ```compile_fail,E0282
/// homura::zeros([1]).sum().view([]);
/// ```
///
/// ```compile_fail,E0282
/// homura::zeros([1]).sum().view(&[]);
/// ```
///
/// ```compile_fail,E0282
/// homura::zeros([1]).sum().view(Vec::new());
/// ```
///
/// ```compile_fail,E0282
/// homura::zeros([1]).sum_dim([], false);
/// ```
pub trait DimSize: Sealed {}

impl Sealed for i64 {
    fn to_dim(self) -> i64 {
        self
    }
}

impl Sealed for i32 {
    fn to_dim(self) -> i64 {
        self.into()
    }
}

impl Sealed for usize {
    fn to_dim(self) -> i64 {
        match i64::try_from(self) {
            Ok(v) => v,
            Err(_) => panic!("dimension or index {self} is too large"),
        }
    }
}

impl DimSize for i64 {}
impl DimSize for i32 {}
impl DimSize for usize {}

/// Converts a list of [`DimSize`] values to `i64`.
#[track_caller]
fn to_dims<T: DimSize>(values: &[T]) -> Vec<i64> {
    let mut dims = Vec::with_capacity(values.len());
    for &v in values {
        dims.push(v.to_dim());
    }
    dims
}

/// A shape argument that may contain `-1` for one dimension, as taken by `view`, `reshape` and
/// `expand`: `[T; N]`, `&[T; N]`, `&[T]` or `Vec<T>` with `T` one of the [`DimSize`] types
/// (`i64`, `i32`, `usize`), a tuple of up to 8 sizes whose types may differ, `()` (zero
/// dimensions) or a [`Shape`].
///
/// ```
/// let x = homura::zeros([2, 3, 4]);
/// let (batch, seq) = (x.size(0), x.size(1));
/// assert_eq!(x.view([batch * seq, 4]).shape(), [6, 4]);
/// assert_eq!(x.view((batch, -1)).shape(), [2, 12]);
/// assert_eq!(x.view([-1, 4]).shape(), [6, 4]);
/// assert_eq!(x.sum().view(()).shape(), []);
/// ```
pub trait IntoShapeSpec {
    /// Converts into a list of sizes, where `-1` has an operation-specific meaning.
    #[track_caller]
    fn into_shape_spec(self) -> Vec<i64>;
}

impl<T: DimSize, const N: usize> IntoShapeSpec for [T; N] {
    fn into_shape_spec(self) -> Vec<i64> {
        to_dims(&self)
    }
}

impl<T: DimSize, const N: usize> IntoShapeSpec for &[T; N] {
    fn into_shape_spec(self) -> Vec<i64> {
        to_dims(self)
    }
}

impl<T: DimSize> IntoShapeSpec for &[T] {
    fn into_shape_spec(self) -> Vec<i64> {
        to_dims(self)
    }
}

impl<T: DimSize> IntoShapeSpec for Vec<T> {
    fn into_shape_spec(self) -> Vec<i64> {
        to_dims(&self)
    }
}

impl IntoShapeSpec for () {
    fn into_shape_spec(self) -> Vec<i64> {
        Vec::new()
    }
}

macro_rules! tuple_shape_spec {
    ($($name:ident),+) => {
        impl<$($name: DimSize),+> IntoShapeSpec for ($($name,)+) {
            #[allow(non_snake_case)]
            fn into_shape_spec(self) -> Vec<i64> {
                let ($($name,)+) = self;
                vec![$($name.to_dim()),+]
            }
        }
    };
}

tuple_shape_spec!(A);
tuple_shape_spec!(A, B);
tuple_shape_spec!(A, B, C);
tuple_shape_spec!(A, B, C, D);
tuple_shape_spec!(A, B, C, D, E);
tuple_shape_spec!(A, B, C, D, E, F);
tuple_shape_spec!(A, B, C, D, E, F, G);
tuple_shape_spec!(A, B, C, D, E, F, G, H);

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

/// One or more dimension indices (negative values count from the end): a single [`DimSize`]
/// value (`i64`, `i32` or `usize`), or `[T; N]`, `&[T; N]`, `&[T]` or `Vec<T>` of one of them.
///
/// An empty list has no literal to infer its type from; write `[] as [i64; 0]`.
pub trait Dims {
    /// Converts into a list of dimension indices.
    #[track_caller]
    fn into_dims(self) -> Vec<i64>;

    /// Whether this is a single dimension rather than a list. PyTorch records `squeeze(dim)` and
    /// `squeeze(dims)` with different backward nodes, and so does homura.
    #[doc(hidden)]
    #[track_caller]
    fn is_single(&self) -> bool {
        false
    }
}

macro_rules! single_dim {
    ($($t:ty),*) => {$(
        impl Dims for $t {
            fn into_dims(self) -> Vec<i64> {
                vec![self.to_dim()]
            }

            fn is_single(&self) -> bool {
                true
            }
        }
    )*};
}

single_dim!(i64, i32, usize);

impl<T: DimSize, const N: usize> Dims for [T; N] {
    fn into_dims(self) -> Vec<i64> {
        to_dims(&self)
    }
}

impl<T: DimSize, const N: usize> Dims for &[T; N] {
    fn into_dims(self) -> Vec<i64> {
        to_dims(self)
    }
}

impl<T: DimSize> Dims for &[T] {
    fn into_dims(self) -> Vec<i64> {
        to_dims(self)
    }
}

impl<T: DimSize> Dims for Vec<T> {
    fn into_dims(self) -> Vec<i64> {
        to_dims(&self)
    }
}

/// Wraps a possibly negative dimension index into `0..num_dims`. Zero-dimensional tensors accept
/// `0` and `-1`, like PyTorch.
#[track_caller]
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
///
/// # Panics
///
/// Panics if a dimension is out of range or appears twice. A zero-dimensional tensor accepts
/// `0` and `-1` like a tensor with one dimension, so naming it twice (`[0, -1]`) is rejected
/// too, as in PyTorch; its mask is still empty.
#[track_caller]
pub(crate) fn dims_mask(dims: &[i64], num_dims: usize) -> Vec<bool> {
    if dims.is_empty() {
        return vec![true; num_dims];
    }
    let mut mask = vec![false; num_dims.max(1)];
    for &d in dims {
        let d = normalize_dim(d, num_dims);
        if mask[d] {
            panic!("dim {d} appears multiple times in the list of dims");
        }
        mask[d] = true;
    }
    mask.truncate(num_dims);
    mask
}

/// Row-major strides for a contiguous tensor of the given shape.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = vec![0; shape.len()];
    let mut acc: usize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        let Some(next) = acc.checked_mul(shape[i].max(1)) else {
            panic!("shape {shape:?} has too many elements");
        };
        acc = next;
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

/// Whether `shape` broadcasts to `desired`: it has no more dimensions, and each of its trailing
/// sizes equals the one in `desired` or is 1 (PyTorch's `is_expandable_to`).
pub(crate) fn is_expandable_to(shape: &[usize], desired: &[usize]) -> bool {
    desired.len().checked_sub(shape.len()).is_some_and(|lead| {
        shape
            .iter()
            .zip(&desired[lead..])
            .all(|(&s, &n)| s == n || s == 1)
    })
}

/// The broadcast shape of `a` and `b` (NumPy rules).
#[track_caller]
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
#[track_caller]
pub(crate) fn infer_size(spec: &[i64], num_elements: usize) -> Shape {
    // A macro, not a closure: `#[track_caller]` doesn't pass through closures.
    macro_rules! invalid {
        () => {
            panic!("shape '{spec:?}' is invalid for input of size {num_elements}")
        };
    }
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
            let Some(product) = known.checked_mul(d as usize) else {
                invalid!();
            };
            known = product;
        }
    }
    let mut out: Vec<usize> = spec.iter().map(|&d| d.max(0) as usize).collect();
    match infer {
        Some(i) => {
            if known == 0 || !num_elements.is_multiple_of(known) {
                invalid!();
            }
            out[i] = num_elements / known;
        }
        None => {
            if known != num_elements {
                invalid!();
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
    fn dims_masks_have_one_entry_per_dimension() {
        assert_eq!(dims_mask(&[-1, 0], 3), [true, false, true]);
        assert_eq!(dims_mask(&[], 2), [true, true]);
        // A zero-dimensional tensor accepts `0` or `-1` once (review round 20, R37), and its
        // mask stays empty.
        for dims in [&[0][..], &[-1], &[]] {
            assert!(dims_mask(dims, 0).is_empty(), "{dims:?}");
        }
        for dims in [[0, 0], [0, -1], [-1, -1]] {
            let err = std::panic::catch_unwind(|| dims_mask(&dims, 0)).unwrap_err();
            assert_eq!(
                err.downcast_ref::<String>().unwrap(),
                "dim 0 appears multiple times in the list of dims"
            );
        }
    }

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
