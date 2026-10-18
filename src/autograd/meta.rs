//! Per-tensor autograd state and the tensor properties derived from it.
//!
//! The functions here lock a tensor's state only briefly and never two tensors' states at once,
//! so callers may hold no other tensor's state lock while calling them. The locks are taken in
//! one order: a tensor's `.grad` lock ([`with_grad_lock`]), then its state, then the list of
//! retained outputs of a node (while registering or moving a `retain_grad`); each may also be
//! taken alone.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use super::node::{GradFn, NodeRef, WeakNodeRef};
use super::{BackwardOptions, record};
use crate::Tensor;
use crate::tensor::{Layout, TensorData};

/// The autograd state of one tensor, created on demand: tensors that never take part in autograd
/// have none.
#[derive(Debug, Default)]
pub(crate) struct AutogradMeta {
    /// The tensor's own flag, set on leaves by `requires_grad_`. A view also requires grad when
    /// its base does, see [`requires_grad`].
    pub requires_grad: bool,
    /// The gradient accumulated into a leaf, or stored into a non-leaf that called
    /// `retain_grad`.
    pub grad: Option<Tensor>,
    /// The node that computed the tensor; `None` for leaves.
    pub grad_fn: Option<NodeRef>,
    /// Which output of `grad_fn` the tensor is.
    pub output_nr: u32,
    /// The leaf's accumulator, created on first use. Weak, since the accumulator holds the leaf.
    pub grad_accumulator: Option<WeakNodeRef>,
    /// Incremented each time an in-place operation in grad mode rewrites the tensor's history.
    /// Views compare it with the value they were recorded against to detect stale history.
    pub history_gen: u64,
    /// Shared with the saved copies of this non-leaf, created when it is first saved. Set when
    /// `detach_` clears the history, so that those copies refuse to be used; a new history
    /// starts without one.
    pub detached: Option<Arc<AtomicBool>>,
    /// Set by `retain_grad` on a non-leaf, whose `grad_fn` then has it registered. Kept by
    /// `detach_`, as in PyTorch.
    pub retains_grad: bool,
    /// Serializes the backward passes that update `grad`, created on first use; see
    /// [`with_grad_lock`].
    pub grad_lock: Option<Arc<Mutex<()>>>,
}

/// How a view relates to the tensor whose storage it shares. Set once, when the view is created.
#[derive(Debug)]
pub(crate) struct ViewInfo {
    /// The root base: the tensor, itself not a view, whose storage the view shares. Views of
    /// views record the root, as in PyTorch. Gradients don't flow to the root through this
    /// field: the view's own history connects it to its direct parent.
    pub base: Tensor,
    /// The `history_gen` of the root base that this view's history belongs to. Views of views
    /// inherit it from their parent, so views created from a stale view are stale too.
    pub base_history_gen: u64,
}

/// A view of `parent` with `layout`, recording its root base. The view operation records the
/// backward node, if any, separately.
///
/// # Panics
///
/// Panics if the layout reaches outside the storage of `parent`.
pub(crate) fn view_of(parent: &Tensor, layout: Layout) -> Tensor {
    let data = parent.data();
    layout.check_bounds(data.dtype.size_in_bytes(), data.storage.nbytes());
    let view = match parent.view_info() {
        Some(v) => ViewInfo {
            base: v.base.clone(),
            base_history_gen: v.base_history_gen,
        },
        None => ViewInfo {
            base: parent.clone(),
            base_history_gen: history_gen(parent),
        },
    };
    let data = TensorData {
        storage: data.storage,
        layout,
        dtype: data.dtype,
    };
    Tensor::from_parts(data, None, Some(view))
}

/// A tensor with `data` that is output `output_nr` of `grad_fn`.
pub(crate) fn with_history(data: TensorData, grad_fn: NodeRef, output_nr: u32) -> Tensor {
    let meta = AutogradMeta {
        grad_fn: Some(grad_fn),
        output_nr,
        ..AutogradMeta::default()
    };
    Tensor::from_parts(data, Some(meta), None)
}

/// Like [`with_history`], for a saved non-leaf input being unpacked: the result shares the
/// original's `detach_` flag. PyTorch hands back the original tensor itself here, so a
/// higher-order graph that saves the unpacked tensor again must still see a later `detach_` of
/// the original.
pub(crate) fn with_saved_history(
    data: TensorData,
    grad_fn: NodeRef,
    output_nr: u32,
    detached: Arc<AtomicBool>,
) -> Tensor {
    let meta = AutogradMeta {
        grad_fn: Some(grad_fn),
        output_nr,
        detached: Some(detached),
        ..AutogradMeta::default()
    };
    Tensor::from_parts(data, Some(meta), None)
}

/// Whether gradients flow to `t` (PyTorch's `requires_grad`): its own flag is set, it has a
/// history, or it is a view of a tensor that requires grad.
pub(crate) fn requires_grad(t: &Tensor) -> bool {
    let own = t
        .autograd_meta()
        .as_ref()
        .is_some_and(|m| m.requires_grad || m.grad_fn.is_some());
    // The base is a root, so this recurses at most once.
    own || t.view_info().is_some_and(|v| requires_grad(&v.base))
}

/// Whether `t` has no history (PyTorch's `is_leaf`). A view created in no-grad mode, or before
/// its base required grad, is a leaf even though it requires grad through its base.
pub(crate) fn is_leaf(t: &Tensor) -> bool {
    t.autograd_meta()
        .as_ref()
        .is_none_or(|m| m.grad_fn.is_none())
}

/// The node that computed `t` and which of its outputs `t` is.
pub(crate) fn grad_fn(t: &Tensor) -> Option<(NodeRef, u32)> {
    let guard = t.autograd_meta();
    let meta = guard.as_ref()?;
    Some((meta.grad_fn.clone()?, meta.output_nr))
}

/// The history of the non-leaf `t` for saving it as an input: its `grad_fn`, which output of it
/// `t` is, and the flag that `detach_` sets, created on first use. `None` for leaves.
pub(crate) fn history_for_saving(t: &Tensor) -> Option<(NodeRef, u32, Arc<AtomicBool>)> {
    let mut guard = t.autograd_meta();
    let meta = guard.as_mut()?;
    let grad_fn = meta.grad_fn.clone()?;
    let detached = meta.detached.get_or_insert_default().clone();
    Some((grad_fn, meta.output_nr, detached))
}

/// Sets the tensor's own flag. `requires_grad_` adds the checks of the public API.
pub(crate) fn set_requires_grad(t: &Tensor, flag: bool) {
    t.autograd_meta()
        .get_or_insert_with(AutogradMeta::default)
        .requires_grad = flag;
}

/// Drops the cached accumulator of `t` after its data was replaced (`Tensor::set_data`), so that
/// new graphs get an accumulator recording the new metadata. Old graphs keep theirs.
pub(crate) fn forget_accumulator(t: &Tensor) {
    let previous = t
        .autograd_meta()
        .as_mut()
        .and_then(|m| m.grad_accumulator.take());
    drop(previous);
}

/// Replaces the data of the leaf `t` with `data` (see `Tensor::set_data`) and, if `grad` is given,
/// its `.grad` with `grad`, for `Module::to`. Holds the `.grad` lock throughout, so that a backward
/// pass never sees one replaced without the other.
pub(crate) fn replace_leaf_data(t: &Tensor, data: TensorData, grad: Option<Tensor>) {
    with_grad_lock(t, || {
        t.set_data(data);
        if grad.is_some() {
            set_grad(t, grad);
        }
    });
}

/// The gradient accumulated into `t`.
pub(crate) fn grad(t: &Tensor) -> Option<Tensor> {
    t.autograd_meta().as_ref().and_then(|m| m.grad.clone())
}

/// Replaces the gradient of `t`. The previous one is dropped after the lock is released.
pub(crate) fn set_grad(t: &Tensor, grad: Option<Tensor>) {
    let previous = {
        let mut guard = t.autograd_meta();
        if guard.is_none() && grad.is_none() {
            return;
        }
        let meta = guard.get_or_insert_with(AutogradMeta::default);
        std::mem::replace(&mut meta.grad, grad)
    };
    drop(previous);
}

/// Runs `update`, which reads and replaces the `.grad` of `t`, serialized with every other such
/// update of `t` by a backward pass: the accumulation into a leaf and the storing of a retained
/// gradient. Concurrent backward passes into the same `.grad` then lose nothing.
///
/// The lock is taken after the state lock of `t` is released and is never held while waiting for
/// another tensor's, so `update` may lock states (through [`grad`] and [`set_grad`]) and run
/// tensor operations, which take no such lock.
pub(crate) fn with_grad_lock<R>(t: &Tensor, update: impl FnOnce() -> R) -> R {
    let lock = t
        .autograd_meta()
        .get_or_insert_with(AutogradMeta::default)
        .grad_lock
        .get_or_insert_default()
        .clone();
    let _serial = lock.lock().unwrap_or_else(|e| e.into_inner());
    update()
}

/// The autograd part of the `Debug` output of `t`, as in PyTorch: `grad_fn=<Name>` if it has a
/// history, else `requires_grad=True` if it requires grad. A stale view, whose `grad_fn` panics,
/// shows `grad_fn=<Invalid>`, what PyTorch shows when getting a tensor's `grad_fn` fails: the
/// view has a history, but it can't be used.
pub(crate) fn repr_suffix(t: &Tensor) -> Option<String> {
    if record::is_stale(t) {
        return Some("grad_fn=<Invalid>".into());
    }
    if let Some((node, _)) = grad_fn(t) {
        return Some(format!("grad_fn=<{}>", node.name()));
    }
    requires_grad(t).then(|| "requires_grad=True".into())
}

/// Whether `t` called `retain_grad` (as a non-leaf).
pub(crate) fn retains_grad(t: &Tensor) -> bool {
    t.autograd_meta().as_ref().is_some_and(|m| m.retains_grad)
}

/// How many times an in-place operation in grad mode has rewritten the history of `t`.
pub(crate) fn history_gen(t: &Tensor) -> u64 {
    t.autograd_meta().as_ref().map_or(0, |m| m.history_gen)
}

impl Tensor {
    /// Whether gradients flow to this tensor (PyTorch's `requires_grad`): it is a leaf whose own
    /// flag is set, it was computed by a recorded operation, or it is a view of a tensor that
    /// requires grad.
    pub fn requires_grad(&self) -> bool {
        requires_grad(self)
    }

    /// Sets whether this leaf requires grad and returns `self`. Operations on tensors that
    /// require grad record the backward graph.
    ///
    /// A view without history (created in no-grad mode, or before its base required grad) is a
    /// leaf too: setting its flag makes it a leaf of its own, whose gradients accumulate into its
    /// `.grad`, not into its base's. For a tensor computed by a recorded operation,
    /// `requires_grad_(true)` does nothing (it already requires grad).
    ///
    /// # Panics
    ///
    /// Panics if `requires_grad` is true and the dtype is not floating point, or if it is false
    /// and the tensor is not a leaf, as PyTorch does.
    pub fn requires_grad_(&self, requires_grad: bool) -> &Self {
        if !is_leaf(self) {
            if !requires_grad {
                panic!(
                    "you can only change requires_grad flags of leaf variables. If you want to \
                     use a computed variable in a subgraph that doesn't require differentiation \
                     use var_no_grad = var.detach()."
                );
            }
            return self;
        }
        if requires_grad && !self.is_floating_point() {
            panic!("only Tensors of floating point dtype can require gradients");
        }
        if requires_grad || self.autograd_meta().is_some() {
            set_requires_grad(self, requires_grad);
        }
        self
    }

    /// Whether this tensor has no history, PyTorch's `is_leaf`: it was created by the user
    /// rather than computed by a recorded operation. Backward passes accumulate gradients into
    /// the `.grad` of leaves; a non-leaf stores its gradient only after
    /// [`retain_grad`](Tensor::retain_grad).
    pub fn is_leaf(&self) -> bool {
        is_leaf(self)
    }

    /// The gradient accumulated into this tensor by backward passes, PyTorch's `.grad`: for a
    /// leaf that requires grad, or for a non-leaf after [`retain_grad`](Tensor::retain_grad).
    pub fn grad(&self) -> Option<Tensor> {
        grad(self)
    }

    /// Makes backward passes store the gradient of this non-leaf in its `.grad`, as they do for
    /// leaves, and returns `self` (PyTorch's `retain_grad`). Does nothing on a leaf, or if
    /// called before.
    ///
    /// The stored gradient is the total gradient flowing into this tensor in a pass, from all
    /// its uses; later passes add theirs to it, out of place, so tensors returned by
    /// [`grad`](Tensor::grad) earlier don't change. A pass stores it whenever it computes it:
    /// [`autograd::grad`](crate::autograd::grad) too, when this tensor is on a path to the
    /// requested inputs or is one of them. With `create_graph`, the copy and later additions
    /// are recorded when their inputs require grad; the first recorded copy is a
    /// `CloneBackward0`. A constant incoming gradient can still be stored without a graph.
    /// A stored gradient's graph can keep this tensor alive; see the
    /// [`autograd`](crate::autograd) module.
    ///
    /// The registration follows the tensor through in-place operations: after `y.add_(..)`, the
    /// new history stores into `y.grad` and the old one no longer does. After
    /// [`detach_`](Tensor::detach_), the old history still stores into it, as in PyTorch.
    ///
    /// ```
    /// let x = homura::tensor([1.0, 2.0]);
    /// x.requires_grad_(true);
    /// let y = &x * 2.0;
    /// y.retain_grad();
    /// (&y * &y).sum().backward();
    /// assert_eq!(y.grad().unwrap().to_vec::<f32>(), [4.0, 8.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the tensor doesn't require grad, with PyTorch's message, or if it is a stale
    /// view (see [`grad_fn`](Tensor::grad_fn)).
    pub fn retain_grad(&self) -> &Self {
        if !requires_grad(self) {
            panic!("can't retain_grad on Tensor that has requires_grad=False");
        }
        if is_leaf(self) || retains_grad(self) {
            return self;
        }
        record::check_not_stale(self);
        let mut guard = self.autograd_meta();
        let meta = guard.as_mut().expect("a non-leaf has autograd state");
        // Checked again under the lock, so that concurrent calls register once.
        if !meta.retains_grad {
            let node = meta.grad_fn.as_ref().expect("a non-leaf has a grad_fn");
            node.add_retained(meta.output_nr, self);
            meta.retains_grad = true;
        }
        self
    }

    /// Whether this tensor is a non-leaf that called [`retain_grad`](Tensor::retain_grad)
    /// (PyTorch's `retains_grad`). Stays true after [`detach_`](Tensor::detach_), as in PyTorch.
    pub fn retains_grad(&self) -> bool {
        retains_grad(self)
    }

    /// Replaces the gradient of this tensor, like assigning to PyTorch's `.grad`; `None` clears
    /// it, which also breaks the reference cycle that `create_graph` can leave (see the
    /// [`autograd`](crate::autograd) module).
    ///
    /// # Panics
    ///
    /// Panics if the gradient's dtype, device or shape differ from the tensor's.
    pub fn set_grad(&self, grad: Option<Tensor>) {
        if let Some(g) = &grad {
            let (dtype, grad_dtype) = (self.dtype(), g.dtype());
            if grad_dtype != dtype {
                panic!(
                    "attempting to assign a gradient with dtype '{grad_dtype}' to a tensor with \
                     dtype '{dtype}'. Please ensure that the gradient and the tensor have the \
                     same dtype"
                );
            }
            let (device, grad_device) = (self.device(), g.device());
            if grad_device != device {
                panic!(
                    "attempting to assign a gradient with device type '{grad_device}' to a tensor \
                     with device type '{device}'. Please ensure that the gradient and the tensor \
                     are on the same device"
                );
            }
            let (shape, grad_shape) = (self.shape(), g.shape());
            if grad_shape != shape {
                panic!(
                    "attempting to assign a gradient of size '{grad_shape:?}' to a tensor of size \
                     '{shape:?}'. Please ensure that the gradient and the tensor are the same size"
                );
            }
        }
        set_grad(self, grad);
    }

    /// The node that computed this tensor, PyTorch's `grad_fn`; `None` for leaves.
    ///
    /// # Panics
    ///
    /// Panics on a view whose base had its history rewritten by an in-place operation in grad
    /// mode after the view was created: the view's node no longer describes how it depends on
    /// the base's new history (PyTorch replays the view on it, which homura doesn't support
    /// yet).
    pub fn grad_fn(&self) -> Option<GradFn> {
        record::check_not_stale(self);
        grad_fn(self).map(|(node, _)| GradFn(node))
    }

    /// A tensor sharing this tensor's data (and version counter) without its history: it doesn't
    /// require grad, and isn't treated as a view, so it can be modified in place. Modifying it
    /// in place is visible in `self`, and is detected by backward passes that saved `self`.
    pub fn detach(&self) -> Tensor {
        Tensor::from_data(self.data())
    }

    /// Removes this tensor from the graph in place, making it a leaf that doesn't require grad,
    /// and returns `self`. Its gradient, if any, is kept. Graphs that saved it while it had a
    /// history can't be run any more, as in PyTorch.
    ///
    /// # Panics
    ///
    /// Panics on views, which can't be detached in place; use [`detach`](Tensor::detach).
    pub fn detach_(&self) -> &Self {
        if self.view_info().is_some() {
            panic!("Can't detach views in-place. Use detach() instead.");
        }
        record::detach_history(self);
        self
    }

    /// Computes the gradient of this single-element tensor with respect to the leaves of its
    /// graph and adds it to their `.grad` (PyTorch's `y.backward()`). The tensors the graph
    /// saved are freed; use [`autograd::backward`](crate::autograd::backward) to keep them or to
    /// record the pass.
    ///
    /// # Panics
    ///
    /// Panics if the tensor doesn't require grad or has more than one element, or if the
    /// backward pass fails; see [`autograd::backward`](crate::autograd::backward).
    pub fn backward(&self) {
        super::backward(&[self], &[], BackwardOptions::default());
    }

    /// Like [`backward`](Tensor::backward), with `gradient` as the gradient of this tensor
    /// (PyTorch's `y.backward(gradient)`): the vector of the vector-Jacobian product. It must
    /// have the shape of this tensor.
    pub fn backward_with(&self, gradient: &Tensor) {
        super::backward(&[self], &[Some(gradient)], BackwardOptions::default());
    }
}
