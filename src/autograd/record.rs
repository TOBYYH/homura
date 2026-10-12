//! Connecting tensors to the backward graph.

use std::sync::atomic::Ordering;

use super::accumulate::grad_accumulator;
use super::formulas::NotImplemented;
use super::node::{Edge, InputMeta, Node, NodeBuilder, NodeRef};
use super::{is_grad_enabled, meta};
use crate::Tensor;

/// Whether an operation on `inputs` is recorded: grad mode is on and a tensor input requires
/// grad. `None` stands for a scalar operand, which never takes part in autograd.
pub(crate) fn needs_grad(inputs: &[Option<&Tensor>]) -> bool {
    is_grad_enabled() && inputs.iter().flatten().any(|t| t.requires_grad())
}

/// Records the backward node of an operation, the one place where operations connect their
/// results to the graph. Does nothing unless [`needs_grad`] holds for `inputs` and every one of
/// `outputs` is floating point. Then the outputs become the outputs of a new node built by
/// `formula`, with one edge per entry of `inputs` (through [`gradient_edge`], `None` for scalar
/// operands); `formula` saves tensors through the builder, which tells which inputs have an
/// edge.
///
/// An input can require grad without having an edge, a view created in no-grad mode for
/// example. The node is recorded anyway, so the outputs require grad, while the gradients for
/// that input are dropped, as in PyTorch.
pub(crate) fn record<N: Node>(
    inputs: &[Option<&Tensor>],
    outputs: &[&Tensor],
    formula: impl FnOnce(&mut NodeBuilder) -> N,
) {
    if !is_grad_enabled() || !outputs.iter().all(|t| t.is_floating_point()) || !needs_grad(inputs) {
        return;
    }
    let edges = inputs.iter().map(|t| t.and_then(gradient_edge)).collect();
    let mut builder = NodeBuilder::new(edges);
    let formula = formula(&mut builder);
    let input_meta = outputs.iter().map(|t| InputMeta::of(t)).collect();
    let node = builder.build(formula, input_meta);
    for (output_nr, t) in outputs.iter().enumerate() {
        set_history(t, node.clone(), output_nr as u32);
    }
}

/// [`record`] for a differentiable operation whose derivative isn't implemented yet: the outputs
/// require grad as they should, and a backward pass that reaches the node fails with PyTorch's
/// "the derivative for 'op' is not implemented." instead of dropping the gradients silently.
pub(crate) fn record_unimplemented(op: &'static str, inputs: &[Option<&Tensor>], output: &Tensor) {
    record(inputs, &[output], |_| NotImplemented { op });
}

/// Checks an in-place operation on `target` with the tensor inputs `inputs` before anything is
/// computed or written, so that a rejected call changes neither data, version nor history.
///
/// In grad mode, writing into a floating point `target` has to be recorded when `target` or an
/// input requires grad. Rewriting the history of the target is not implemented yet, so such
/// calls panic: on a leaf that requires grad with PyTorch's message (PyTorch rejects these
/// too), on a view with homura's (views will keep it until in-place operations on views are
/// supported), and otherwise with a message that the operation isn't supported yet. In no-grad
/// mode every in-place operation is allowed.
///
/// # Panics
///
/// Panics as described above.
pub(crate) fn check_inplace(target: &Tensor, inputs: &[Option<&Tensor>]) {
    if !is_grad_enabled() || !target.is_floating_point() {
        return;
    }
    let target_requires_grad = target.requires_grad();
    if !target_requires_grad && !inputs.iter().flatten().any(|t| t.requires_grad()) {
        return;
    }
    if target.view_info().is_some() {
        panic!(
            "in-place operations on a view are not supported yet when the view or one of the \
             inputs requires grad; use the out-of-place operation, or deep_clone() the view first"
        );
    }
    if target_requires_grad && meta::is_leaf(target) {
        panic!("a leaf Variable that requires grad is being used in an in-place operation.");
    }
    panic!(
        "in-place operations that need to be recorded for autograd are not supported yet: the \
         tensor written to or one of the inputs requires grad; use the out-of-place operation"
    );
}

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
// In-place operations start rewriting histories in M1 step 7; until then only tests do.
#[cfg_attr(not(test), expect(dead_code))]
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
