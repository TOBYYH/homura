//! Per-tensor autograd state and the tensor properties derived from it.
//!
//! The functions here lock a tensor's state only briefly and never two tensors' states at once,
//! so callers may hold no other autograd lock while calling them.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::node::{NodeRef, WeakNodeRef};
use crate::Tensor;
use crate::tensor::{Layout, TensorData};

/// The autograd state of one tensor, created on demand: tensors that never take part in autograd
/// have none.
#[derive(Debug, Default)]
pub(crate) struct AutogradMeta {
    /// The tensor's own flag, set on leaves by `requires_grad_`. A view also requires grad when
    /// its base does, see [`requires_grad`].
    pub requires_grad: bool,
    /// The gradient accumulated into a leaf.
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

/// How many times an in-place operation in grad mode has rewritten the history of `t`.
pub(crate) fn history_gen(t: &Tensor) -> u64 {
    t.autograd_meta().as_ref().map_or(0, |m| m.history_gen)
}
