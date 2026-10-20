//! Operations that return views sharing the storage of the input.

use super::{Layout, Tensor};
use crate::autograd::formulas::{self, Squeeze};
use crate::shape::{
    Dims, IntoShape, IntoShapeSpec, compute_view_strides, dims_mask, infer_size, normalize_dim,
};
use crate::{Shape, Strides};

/// The dimension and the position that `select(dim, index)` picks in a tensor of `shape`, with
/// negative values counted from the end.
///
/// # Panics
///
/// Panics with PyTorch's messages if `shape` is zero-dimensional or `dim` or `index` is out of
/// range.
pub(crate) fn select_position(shape: &Shape, dim: i64, index: i64) -> (usize, usize) {
    if shape.is_empty() {
        panic!("select() cannot be applied to a 0-dim tensor.");
    }
    let d = normalize_dim(dim, shape.len());
    let size = shape[d] as i64;
    if index < -size || index >= size {
        panic!(
            "select(): index {index} out of range for tensor of size {shape:?} at dimension {d}"
        );
    }
    (d, (if index < 0 { index + size } else { index }) as usize)
}

/// The Python-style slice `start:end:step` of a dimension of `size`: the start and end clamped
/// into `0..=size` (negative values count from the end) and the number of positions.
///
/// # Panics
///
/// Panics if `step` is not positive.
pub(crate) fn slice_bounds(
    size: usize,
    start: Option<i64>,
    end: Option<i64>,
    step: i64,
) -> (usize, usize, usize) {
    if step <= 0 {
        panic!("slice step must be positive");
    }
    let size = size as i64;
    let clamp = |v: i64| (if v < 0 { v + size } else { v }).clamp(0, size);
    let start = clamp(start.unwrap_or(0));
    let end = clamp(end.unwrap_or(size));
    let len = if end > start {
        (end - start - 1) / step + 1
    } else {
        0
    };
    (start as usize, end as usize, len as usize)
}

impl Tensor {
    /// A view with a different shape (one dimension may be `-1`). The memory layout must allow
    /// it without copying; [`reshape`](Tensor::reshape) copies when needed.
    pub fn view(&self, shape: impl IntoShapeSpec) -> Tensor {
        let out = self.make_view(self.view_layout(shape));
        formulas::view(self, &out);
        out
    }

    /// [`view`](Tensor::view) for a result that the calling operation has just allocated, like
    /// PyTorch's `_unsafe_view`: the result is not recorded as a view, since no one else holds the
    /// tensor whose storage it shares. Otherwise in-place operations on the result would be
    /// treated as writes to a view. It is still differentiable: gradients flow through it.
    pub(crate) fn unsafe_view(&self, shape: impl IntoShapeSpec) -> Tensor {
        let out = self.with_layout(self.view_layout(shape));
        formulas::unsafe_view(self, &out);
        out
    }

    fn view_layout(&self, shape: impl IntoShapeSpec) -> Layout {
        let l = self.layout();
        let shape = infer_size(&shape.into_shape_spec(), l.num_elements());
        let Some(strides) = compute_view_strides(&l.shape, &l.strides, &shape) else {
            panic!(
                "view size is not compatible with input tensor's size and stride (at least one \
                 dimension spans across two contiguous subspaces). Use .reshape(...) instead."
            );
        };
        Layout {
            shape,
            strides,
            offset: l.offset,
        }
    }

    /// A tensor with a different shape (one dimension may be `-1`): a view when the memory
    /// layout allows it, otherwise a contiguous copy.
    pub fn reshape(&self, shape: impl IntoShapeSpec) -> Tensor {
        let l = self.layout();
        let shape = infer_size(&shape.into_shape_spec(), l.num_elements());
        match compute_view_strides(&l.shape, &l.strides, &shape) {
            Some(strides) => {
                let out = self.make_view(Layout {
                    shape,
                    strides,
                    offset: l.offset,
                });
                formulas::view(self, &out);
                out
            }
            None => self.deep_clone().unsafe_view(shape),
        }
    }

    /// Reorders the dimensions: dimension `i` of the result is dimension `dims[i]` of `self`.
    pub fn permute(&self, dims: impl Dims) -> Tensor {
        let l = self.layout();
        let dims = dims.into_dims();
        let n = l.shape.len();
        if dims.len() != n {
            panic!(
                "permute(): number of dimensions in the tensor input does not match the length of \
                 the desired ordering of dimensions i.e. input.dim() = {n} is not equal to \
                 len(dims) = {}",
                dims.len()
            );
        }
        let mut seen = vec![false; n];
        let mut wrapped = vec![0; n];
        let mut shape = vec![0; n];
        let mut strides = vec![0; n];
        for (i, &d) in dims.iter().enumerate() {
            let d = normalize_dim(d, n);
            if seen[d] {
                panic!("permute(): duplicate dims are not allowed.");
            }
            seen[d] = true;
            wrapped[i] = d;
            shape[i] = l.shape[d];
            strides[i] = l.strides[d];
        }
        let out = self.make_view(Layout {
            shape: Shape::new(&shape),
            strides: Strides::new(&strides),
            offset: l.offset,
        });
        formulas::permute(self, wrapped, &out);
        out
    }

    /// Swaps two dimensions.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Tensor {
        let mut l = self.layout();
        let n = l.shape.len();
        if n == 0 {
            normalize_dim(dim0, n);
            normalize_dim(dim1, n);
        } else {
            let (a, b) = (normalize_dim(dim0, n), normalize_dim(dim1, n));
            l.shape.swap(a, b);
            l.strides.swap(a, b);
        }
        let out = self.make_view(l);
        formulas::transpose(self, dim0, dim1, &out);
        out
    }

    /// A view broadcast to `shape`: size-1 dimensions may grow and new leading dimensions may be
    /// added; `-1` keeps a size. No memory is copied.
    pub fn expand(&self, shape: impl IntoShapeSpec) -> Tensor {
        let l = self.layout();
        let spec = shape.into_shape_spec();
        let (n, m) = (spec.len(), l.shape.len());
        if n < m {
            panic!(
                "expand: the number of sizes provided ({n}) must be greater or equal to the \
                 number of dimensions in the tensor ({m})"
            );
        }
        let mut target = vec![0; n];
        for i in 0..n {
            target[i] = if spec[i] == -1 {
                if i < n - m {
                    panic!(
                        "The expanded size of the tensor (-1) isn't allowed in a leading, \
                         non-existing dimension {i}"
                    );
                }
                l.shape[i - (n - m)]
            } else if spec[i] < 0 {
                panic!("invalid shape dimension {}", spec[i]);
            } else {
                spec[i] as usize
            };
        }
        let out = self.make_view(l.expand_to(&Shape::new(&target)));
        formulas::expand(self, &out);
        out
    }

    /// [`expand`](Tensor::expand) to the shape of `other`.
    pub fn expand_as(&self, other: &Tensor) -> Tensor {
        self.expand(other.shape())
    }

    /// Removes all dimensions of size 1.
    pub fn squeeze(&self) -> Tensor {
        let l = self.layout();
        let remove: Vec<bool> = l.shape.iter().map(|&size| size == 1).collect();
        self.remove_dims(Squeeze::All, &l, &remove)
    }

    /// Removes the given dimensions if they have size 1 (others are left alone, like PyTorch).
    /// An empty list removes none.
    pub fn squeeze_dim(&self, dims: impl Dims) -> Tensor {
        let kind = if dims.is_single() {
            Squeeze::Dim
        } else {
            Squeeze::Dims
        };
        let dims = dims.into_dims();
        let l = self.layout();
        // Unlike reductions, where it stands for all dimensions, PyTorch's empty list here
        // squeezes nothing.
        let mask = if dims.is_empty() {
            vec![false; l.shape.len()]
        } else {
            dims_mask(&dims, l.shape.len())
        };
        let remove: Vec<bool> = (0..l.shape.len())
            .map(|d| mask[d] && l.shape[d] == 1)
            .collect();
        self.remove_dims(kind, &l, &remove)
    }

    /// The view of `l` without the dimensions `d` where `remove[d]` holds.
    fn remove_dims(&self, kind: Squeeze, l: &Layout, remove: &[bool]) -> Tensor {
        let keep = || (0..l.shape.len()).filter(|&d| !remove[d]);
        let out = self.make_view(Layout {
            shape: keep().map(|d| l.shape[d]).collect(),
            strides: keep().map(|d| l.strides[d]).collect(),
            offset: l.offset,
        });
        let removed = (0..l.shape.len()).filter(|&d| remove[d]).collect();
        formulas::squeeze(self, kind, removed, &out);
        out
    }

    /// Inserts a dimension of size 1 at position `dim` (`-1` appends at the end).
    pub fn unsqueeze(&self, dim: i64) -> Tensor {
        let l = self.layout();
        let d = normalize_dim(dim, l.shape.len() + 1);
        let out = self.make_view(l.unsqueeze(d));
        formulas::unsqueeze(self, d, &out);
        out
    }

    /// A view of all of `self` with the same layout (PyTorch's `alias`): what indexing returns
    /// when no index item changes anything, so that the result is still a new tensor.
    pub(crate) fn alias(&self) -> Tensor {
        let out = self.make_view(self.layout());
        formulas::alias(self, &out);
        out
    }

    /// The `length` elements along `dim` starting at `start` (negative values count from the
    /// end).
    pub fn narrow(&self, dim: i64, start: i64, length: usize) -> Tensor {
        let l = self.layout();
        if l.shape.is_empty() {
            panic!("narrow() cannot be applied to a 0-dim tensor.");
        }
        let d = normalize_dim(dim, l.shape.len());
        let size = l.shape[d] as i64;
        if start < -size || start > size {
            panic!(
                "start out of range (expected to be in range of [{}, {size}], but got {start})",
                -size
            );
        }
        let start = if start < 0 { start + size } else { start } as usize;
        if start
            .checked_add(length)
            .is_none_or(|end| end > size as usize)
        {
            panic!("start ({start}) + length ({length}) exceeds dimension size ({size}).");
        }
        let out = self.make_view(l.slice(d, start, length, 1));
        // As in PyTorch, where `narrow` is a slice.
        formulas::slice(self, d, start, start + length, 1, &out);
        out
    }

    /// The slice at `index` along `dim`, with that dimension removed.
    pub fn select(&self, dim: i64, index: i64) -> Tensor {
        let l = self.layout();
        let (d, index) = select_position(&l.shape, dim, index);
        let out = self.make_view(l.select(d, index));
        formulas::select(self, d, index, &out);
        out
    }

    /// A Python-style slice `start:end:step` along `dim`: bounds may be negative and are clamped
    /// to the size.
    pub(crate) fn slice(
        &self,
        dim: usize,
        start: Option<i64>,
        end: Option<i64>,
        step: i64,
    ) -> Tensor {
        let l = self.layout();
        let (start, end, len) = slice_bounds(l.shape[dim], start, end, step);
        let out = self.make_view(l.slice(dim, start, len, step as usize));
        formulas::slice(self, dim, start, end, step as usize, &out);
        out
    }

    /// Merges dimensions `start_dim..=end_dim` into one.
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Tensor {
        let shape = self.shape();
        if shape.is_empty() {
            normalize_dim(start_dim, 0);
            normalize_dim(end_dim, 0);
            return self.reshape([1]);
        }
        let (s, e) = (
            normalize_dim(start_dim, shape.len()),
            normalize_dim(end_dim, shape.len()),
        );
        if s > e {
            panic!("flatten() has invalid args: start_dim cannot come after end_dim");
        }
        let mut spec: Vec<i64> = shape[..s].iter().map(|&d| d as i64).collect();
        spec.push(shape[s..=e].iter().product::<usize>() as i64);
        spec.extend(shape[e + 1..].iter().map(|&d| d as i64));
        self.reshape(spec)
    }

    /// A view of the windows of `size` elements along `dim` that start every `step` elements
    /// (PyTorch's `unfold`): dimension `dim` counts the `(n - size) / step + 1` windows of its
    /// `n` elements, and a new last dimension of `size` holds the elements of each window.
    /// Elements left over after the last full window aren't in any. A zero-dimensional tensor
    /// counts as one element (`dim` is `0` or `-1`).
    ///
    /// Multiple windows overlap when `step < size`: the view then refers to some elements more
    /// than once, so in-place operations on it are rejected, where PyTorch writes such elements
    /// several times. A single window introduces no overlap; a view can also inherit overlap
    /// from its input. Gradients of elements in several windows add up, in the dtype and in
    /// window order like PyTorch's CPU kernel; the gradient is recorded as
    /// `UnfoldBackwardBackward0`, whose own gradient unfolds again.
    ///
    /// ```
    /// let x = homura::arange(5.0);
    /// let windows = x.unfold(0, 3, 2);
    /// assert_eq!(windows.shape(), [2, 3]);
    /// assert_eq!(windows.to_vec::<f32>(), [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's messages if `dim` is out of range, `size` exceeds the size of `dim`
    /// or `step` is zero.
    pub fn unfold(&self, dim: i64, size: usize, step: usize) -> Tensor {
        let l = self.layout();
        let nd = l.shape.len();
        let d = normalize_dim(dim, nd);
        let max = if nd == 0 { 1 } else { l.shape[d] };
        if size > max {
            panic!("maximum size for tensor at dimension {d} is {max} but size is {size}");
        }
        if step == 0 {
            panic!("step is 0 but must be > 0");
        }
        let mut layout = l.clone();
        layout.shape.insert(nd, size);
        layout
            .strides
            .insert(nd, if nd == 0 { 1 } else { l.strides[d] });
        if nd > 0 {
            layout.shape[d] = (l.shape[d] - size) / step + 1;
            // With a single window the stride is never used, and a huge step may not fit.
            layout.strides[d] = l.strides[d].checked_mul(step).unwrap_or(l.strides[d]);
        }
        let out = self.make_view(layout);
        formulas::unfold(self, d, size, step, &out);
        out
    }

    /// A view with an explicit shape, strides and storage offset (in elements).
    ///
    /// Gradients flow back to the elements of `self` that the view shares with it. Computing the
    /// gradient of a `self` whose elements overlap in memory (an expanded tensor, for example) is
    /// not supported yet and panics when a backward pass reaches it.
    ///
    /// # Panics
    ///
    /// Panics if the view would reach outside the storage.
    pub fn as_strided(
        &self,
        size: impl IntoShape,
        stride: impl IntoShape,
        storage_offset: usize,
    ) -> Tensor {
        let (shape, stride) = (size.into_shape(), stride.into_shape());
        if shape.len() != stride.len() {
            panic!("mismatch in length of strides and shape");
        }
        // `make_view` checks that the view lies inside the storage.
        let out = self.make_view(Layout {
            shape,
            strides: Strides::new(&stride),
            offset: storage_offset,
        });
        formulas::as_strided(self, &out);
        out
    }

    /// `self` if it is already contiguous, otherwise a contiguous copy.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            self.clone()
        } else {
            self.deep_clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Ellipsis, NewAxis, Tensor};

    /// Whether `t` is recorded as a view of the root `base`.
    fn is_view_of(t: &Tensor, base: &Tensor) -> bool {
        t.view_info()
            .is_some_and(|v| v.base.is_same(base) && v.base_history_gen == 0)
    }

    #[test]
    fn public_views_record_their_root_base() {
        for device in [Device::Cpu, Device::Meta] {
            let x = crate::zeros_with([2, 3, 4], device);
            let views = [
                ("view", x.view([6, 4])),
                ("reshape", x.reshape([4, 6])),
                ("flatten", x.flatten(0, 1)),
                ("permute", x.permute([2, 0, 1])),
                ("transpose", x.transpose(0, 2)),
                ("expand", x.expand([5, 2, 3, 4])),
                (
                    "expand_as",
                    x.expand_as(&crate::zeros_with([2, 2, 3, 4], device)),
                ),
                ("squeeze", x.squeeze()),
                ("squeeze_dim", x.squeeze_dim(0)),
                ("unsqueeze", x.unsqueeze(-1)),
                ("narrow", x.narrow(1, 1, 2)),
                ("select", x.select(0, 1)),
                ("index", x.index((.., 1..3, NewAxis))),
                (
                    "index with a zero-dimensional integer tensor",
                    x.index((.., crate::tensor(-1i64))),
                ),
                ("as_strided", x.as_strided([2, 2], [1, 2], 1)),
                ("split", x.split(1, 0)[1].clone()),
                ("chunk", x.chunk(2, 2)[0].clone()),
                ("contiguous of a view", x.narrow(0, 0, 1).contiguous()),
                // Views of views record the root.
                (
                    "view of a view",
                    x.transpose(0, 1).narrow(0, 1, 2).unsqueeze(1),
                ),
            ];
            for (name, v) in &views {
                assert!(is_view_of(v, &x), "{name} on {device}");
            }
        }
        let s = crate::tensor(1.0);
        for (name, v) in [
            ("transpose", s.transpose(0, -1)),
            ("view", s.view([1])),
            ("reshape", s.reshape([1, 1])),
            ("flatten", s.flatten(0, -1)),
        ] {
            assert!(is_view_of(&v, &s), "zero-dimensional {name}");
        }
    }

    #[test]
    fn empty_splits_and_chunks_are_separate_views_of_the_root() {
        let x = crate::zeros([0, 3]);
        let parent = x.narrow(1, 1, 2);
        let pieces = |t: &Tensor| {
            let mut all = t.chunk(3, 0);
            all.extend(t.split(0, 0));
            all
        };
        for t in [&x, &parent] {
            let all = pieces(t);
            {
                let _guard = crate::no_grad();
                let more = pieces(t);
                assert!(more.iter().all(|p| is_view_of(p, &x)));
            }
            for (i, p) in all.iter().enumerate() {
                assert!(is_view_of(p, &x));
                assert!(all[i + 1..].iter().all(|q| !q.is_same(p)));
            }
        }
        // Pieces of a stale view are stale too: they keep the generation their parent recorded.
        x.autograd_meta()
            .get_or_insert_with(Default::default)
            .history_gen = 1;
        for p in pieces(&parent) {
            assert_eq!(p.view_info().unwrap().base_history_gen, 0);
        }
        assert!(
            pieces(&x)
                .iter()
                .all(|p| p.view_info().unwrap().base_history_gen == 1)
        );
    }

    #[test]
    fn stale_permuted_squeezed_and_aliased_views_are_rejected() {
        let x = crate::rand([2, 1, 3]);
        x.requires_grad_(true);
        let views = [
            x.permute([2, 0, 1]),
            x.transpose(0, 2),
            x.squeeze(),
            x.squeeze_dim(1),
            x.squeeze_dim([1]),
            x.unsqueeze(0),
            x.index((.., ..)),
            x.index(Ellipsis),
            x.index((.., .., 1..)),
        ];
        let views_of_views: Vec<Tensor> = views.iter().map(|v| v.unsqueeze(0)).collect();
        // Stands in for an in-place operation in grad mode, which rewrites the history of `x`.
        x.autograd_meta()
            .get_or_insert_with(Default::default)
            .history_gen = 1;
        let stale = |f: &dyn Fn()| {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
            let message = err
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| err.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            assert!(
                message.starts_with("a view whose base was modified"),
                "{message}"
            );
        };
        for v in views.iter().chain(&views_of_views) {
            assert!(is_view_of(v, &x));
            stale(&|| drop(v * 2.0));
            stale(&|| drop(v.transpose(0, -1)));
            stale(&|| v.sum().backward());
        }
    }

    #[test]
    fn fresh_results_are_not_views() {
        let x = crate::rand([2, 3, 4]);
        let t = x.transpose(0, 1);
        let fresh = [
            ("reshape that copies", t.reshape([-1])),
            ("flatten that copies", t.flatten(0, -1)),
            ("contiguous", t.contiguous()),
            ("deep_clone", t.deep_clone()),
            ("repeat", x.repeat([2, 1, 1])),
            ("repeat once", x.repeat([1, 1, 1])),
            ("tensor index", x.index((.., crate::tensor([[2i64, 0]])))),
            (
                "two tensor indices",
                x.index((crate::tensor([1i64]), .., crate::tensor([0i64, 3]))),
            ),
            ("mask", x.index(x.greater(0.5))),
            ("bool item", x.index(true)),
            (
                "index_put",
                x.index_put(
                    (crate::tensor([1i64]), .., crate::tensor([0i64])),
                    &x.index((0, 0, 0)),
                    false,
                ),
            ),
            ("index_select", x.index_select(1, &crate::tensor([0i64]))),
            (
                "tensor index of an empty layout",
                crate::zeros([0])
                    .as_strided([4, 0], [isize::MAX as usize, 0], 0)
                    .index((crate::tensor([3i64]), ..)),
            ),
            ("matmul", x.matmul(&crate::ones([4, 5]))),
            ("matmul with a vector", x.matmul(&crate::ones([4]))),
            ("dot", crate::ones([4]).matmul(&crate::ones([4]))),
            ("broadcast matmul", t.matmul(&crate::ones([2, 1, 4, 5]))),
            ("concat", crate::concat(&[&x, &x], 1)),
            ("stack", crate::stack(&[&x, &x], 0)),
            ("argmax", t.argmax()),
            ("max", t.max()),
            ("to", x.to(DType::F64)),
            ("add", &t + 1.0),
            ("eye", crate::eye(3)),
            ("tensor", crate::tensor([[1.0, 2.0]])),
            ("arange", crate::arange(4)),
            ("zeros", crate::zeros([2, 2])),
        ];
        for (name, r) in &fresh {
            assert!(r.view_info().is_none(), "{name}");
        }
    }
}
