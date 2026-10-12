//! Connecting tensors to the backward graph.

use std::sync::atomic::Ordering;

use super::accumulate::grad_accumulator;
use super::meta;
use super::node::{Edge, NodeRef};
use crate::Tensor;

/// Where a gradient for `t` goes: output `output_nr` of its `grad_fn`, the accumulator of a leaf
/// whose own flag is set, or nowhere (`None`). A view created in no-grad mode, or before its
/// base required grad, requires grad but has neither, so gradients reaching it are dropped, as
/// in PyTorch.
///
/// This is the only way to find a tensor's edge. Recording an operation's inputs, the roots of
/// `backward` and `grad`, the inputs requested from `grad` and new views all go through it, so
/// the stale-view check here covers every use.
///
/// # Panics
///
/// Panics if `t` is a view whose base had its history rewritten by an in-place operation in
/// grad mode after the view was created. Replaying the view on the new history is not
/// supported yet.
pub(crate) fn gradient_edge(t: &Tensor) -> Option<Edge> {
    if let Some(view) = t.view_info()
        && meta::history_gen(&view.base) != view.base_history_gen
    {
        panic!(
            "a view whose base was modified by an in-place operation in grad mode after the view \
             was created can't be used for gradient computation yet; create the view again \
             after the in-place operation"
        );
    }
    if let Some((node, output_nr)) = meta::grad_fn(t) {
        return Some(Edge {
            node,
            input_nr: output_nr,
        });
    }
    let own_flag = t.autograd_meta().as_ref().is_some_and(|m| m.requires_grad);
    own_flag.then(|| Edge {
        node: grad_accumulator(t),
        input_nr: 0,
    })
}

/// Makes `t` output `output_nr` of `node`, which has just been recorded.
pub(crate) fn set_history(t: &Tensor, node: NodeRef, output_nr: u32) {
    let previous = {
        let mut guard = t.autograd_meta();
        let meta = guard.get_or_insert_with(meta::AutogradMeta::default);
        meta.output_nr = output_nr;
        meta.grad_fn.replace(node)
    };
    drop(previous);
}

/// Replaces the history of `t`, which an in-place operation in grad mode has just overwritten.
/// Views of `t` recorded against the previous history become stale.
pub(crate) fn rebase_history(t: &Tensor, node: NodeRef, output_nr: u32) {
    debug_assert!(
        t.view_info().is_none(),
        "the history of a view can't be rebased yet"
    );
    let previous = {
        let mut guard = t.autograd_meta();
        let meta = guard.get_or_insert_with(meta::AutogradMeta::default);
        meta.output_nr = output_nr;
        meta.history_gen += 1;
        meta.grad_fn.replace(node)
    };
    drop(previous);
}

/// Removes `t` from the graph in place, what `detach_` does once its checks pass: `t` becomes a
/// leaf that doesn't require grad. Saved copies of `t` taken while it had a history refuse to be
/// used from now on, even after `t` is dropped, as in PyTorch.
pub(crate) fn detach_history(t: &Tensor) {
    let previous = {
        let mut guard = t.autograd_meta();
        let Some(meta) = guard.as_mut() else {
            return;
        };
        if let Some(detached) = meta.detached.take() {
            detached.store(true, Ordering::Release);
        }
        meta.requires_grad = false;
        meta.output_nr = 0;
        meta.grad_fn.take()
    };
    drop(previous);
}
