//! Connecting tensors to the backward graph.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::accumulate::grad_accumulator;
use super::node::{Edge, InputMeta, Node, NodeBuilder, NodeRef};
use super::{is_grad_enabled, meta};
use crate::Tensor;

/// Whether an operation on `inputs` is recorded: grad mode is on and a tensor input requires
/// grad. `None` stands for a scalar operand, which never takes part in autograd.
pub(crate) fn needs_grad(inputs: &[Option<&Tensor>]) -> bool {
    is_grad_enabled() && inputs.iter().flatten().any(|t| t.requires_grad())
}

/// The checks [`record`] makes on `inputs` when it looks up their edges, for operations that
/// have side effects before they record (`nn::functional::batch_norm` writes its running
/// statistics): if the operation will be recorded, panics like [`gradient_edge`] for a stale
/// input view, so that it fails before anything is written.
pub(crate) fn check_recordable(inputs: &[Option<&Tensor>]) {
    if needs_grad(inputs) {
        inputs.iter().flatten().for_each(|t| check_not_stale(t));
    }
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
///
/// Inside [`record_in_place`], an output that is the target of the in-place operation is
/// recorded in place: the edges are looked up and the inputs saved while the target still holds
/// its old value and history, then the result is written, the target's saved copies are moved to
/// the version after the write, and the new node replaces the target's history.
pub(crate) fn record<N: Node>(
    inputs: &[Option<&Tensor>],
    outputs: &[&Tensor],
    formula: impl FnOnce(&mut NodeBuilder) -> N,
) {
    if !needs_grad(inputs) || !outputs.iter().all(|t| t.is_floating_point()) {
        return;
    }
    let in_place = take_in_place_write(outputs);
    let edges = inputs.iter().map(|t| t.and_then(gradient_edge)).collect();
    let mut builder = NodeBuilder::new(edges);
    let formula = formula(&mut builder);
    let target_nr = in_place.map(|(output_nr, write)| {
        write();
        builder.refresh_output(output_nr as u32, outputs[output_nr].version());
        output_nr
    });
    let input_meta = outputs.iter().map(|t| InputMeta::of(t)).collect();
    let node = builder.build(formula, input_meta);
    for (output_nr, t) in outputs.iter().enumerate() {
        if target_nr == Some(output_nr) {
            rebase_history(t, node.clone(), output_nr as u32);
        } else {
            set_history(t, node.clone(), output_nr as u32);
        }
    }
}

/// The in-place operation being recorded on this thread: the tensor it writes into and the
/// write, until [`record`] runs it.
struct InPlace {
    target: Tensor,
    write: Option<Box<dyn FnOnce()>>,
}

thread_local! {
    static IN_PLACE: RefCell<Option<InPlace>> = const { RefCell::new(None) };
}

/// Restores the enclosing in-place context, if any, when an in-place recording ends, also when
/// it panics.
struct InPlaceGuard(Option<InPlace>);

impl Drop for InPlaceGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        let current = IN_PLACE.with(|c| c.replace(previous));
        // Dropped after the context is restored: dropping tensors runs no recording code, but
        // keep the thread-local free while it happens anyway.
        drop(current);
    }
}

/// Takes the pending write of the in-place operation being recorded if it targets one of
/// `outputs`, and that output's position.
///
/// # Panics
///
/// Panics if the write already ran: only one node may be recorded for the target.
fn take_in_place_write(outputs: &[&Tensor]) -> Option<(usize, Box<dyn FnOnce()>)> {
    IN_PLACE.with(|c| {
        let mut context = c.borrow_mut();
        let context = context.as_mut()?;
        let output_nr = outputs.iter().position(|t| t.is_same(&context.target))?;
        let write = context
            .write
            .take()
            .expect("internal error: an in-place operation recorded its target twice");
        Some((output_nr, write))
    })
}

/// Records the in-place operation that `write` performs on `target`: `recorder` records the
/// operation's node with `target` as its output (through [`record`]), which runs `write` exactly
/// once, between saving the inputs and replacing the target's history. Contexts nest: a
/// recording inside `recorder` for another target restores this one when it ends.
///
/// Call it only when [`check_in_place`] returned true, and with `write` unable to fail: then a
/// panic before the write (an input that is a stale view, say) leaves data, version and history
/// unchanged.
///
/// # Panics
///
/// Panics if `recorder` didn't record a node for `target`, an internal error. Nothing has been
/// written then.
pub(crate) fn record_in_place(target: &Tensor, write: Box<dyn FnOnce()>, recorder: impl FnOnce()) {
    let context = InPlace {
        target: target.clone(),
        write: Some(write),
    };
    let guard = InPlaceGuard(IN_PLACE.with(|c| c.replace(Some(context))));
    recorder();
    let written = IN_PLACE.with(|c| c.borrow().as_ref().is_some_and(|c| c.write.is_none()));
    assert!(
        written,
        "internal error: an in-place operation didn't record its target"
    );
    drop(guard);
}

/// A copy of `t` in new storage that has the history of `t`, for in-place operations whose
/// backward formula saves the value of the target before the write (PyTorch's `original_self`):
/// gradients through the copy reach the old history directly, and higher-order graphs through the
/// saved value too. A target without a history (a tensor that doesn't require grad) gives a copy
/// without one.
pub(crate) fn original_value(t: &Tensor) -> Tensor {
    let data = crate::ops::clone_data(&t.data());
    match meta::grad_fn(t) {
        Some((node, output_nr)) => meta::with_history(data, node, output_nr),
        None => Tensor::from_data(data),
    }
}

/// Checks an in-place operation on `target` with the tensor inputs `inputs` before anything is
/// computed or written, so that a rejected call changes neither data, version nor history, and
/// returns whether it has to be recorded: in grad mode, a write into a floating point `target`
/// is recorded when `target` or an input requires grad.
///
/// Such a write is rejected on a view: on a view of a leaf whose own flag is set with PyTorch's
/// message, on other views with homura's, since in-place operations on views aren't supported
/// yet (D7). Views are told apart first, since a view without a history is a leaf too. A write
/// into a leaf that requires grad is rejected with PyTorch's message. Everything else is
/// recorded: a tensor computed by a recorded operation, and a tensor that doesn't require grad
/// (a buffer), which joins the graph.
///
/// # Panics
///
/// Panics as described above.
pub(crate) fn check_in_place(target: &Tensor, inputs: &[Option<&Tensor>]) -> bool {
    if !is_grad_enabled() || !target.is_floating_point() {
        return false;
    }
    if !target.requires_grad() && !inputs.iter().flatten().any(|t| t.requires_grad()) {
        return false;
    }
    if let Some(view) = target.view_info() {
        if has_own_flag(&view.base) && meta::is_leaf(&view.base) {
            panic!(
                "a view of a leaf Variable that requires grad is being used in an in-place \
                 operation."
            );
        }
        panic!(
            "in-place operations on a view are not supported yet when the view or one of the \
             inputs requires grad; use the out-of-place operation, or deep_clone() the view first"
        );
    }
    if has_own_flag(target) && meta::is_leaf(target) {
        panic!("a leaf Variable that requires grad is being used in an in-place operation.");
    }
    true
}

fn has_own_flag(t: &Tensor) -> bool {
    t.autograd_meta().as_ref().is_some_and(|m| m.requires_grad)
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
    check_not_stale(t);
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

/// Panics if `t` is a view whose history is stale: its base had its history rewritten by an
/// in-place operation in grad mode after the view was created (replaying the view on the new
/// history is not supported yet), or its base's data was replaced by `Module::to`, so that the
/// view no longer shares it.
pub(crate) fn check_not_stale(t: &Tensor) {
    match staleness(t) {
        None => {}
        Some(Stale::History) => panic!(
            "a view whose base was modified by an in-place operation in grad mode after the view \
             was created can't be used for gradient computation yet; create the view again \
             after the in-place operation"
        ),
        Some(Stale::Data) => panic!(
            "a view whose base had its data replaced (by Module::to) after the view was created \
             can't be used for gradient computation: it still refers to the old data; create \
             the view again"
        ),
    }
}

/// Whether `t` is a view whose history is stale, see [`check_not_stale`].
pub(crate) fn is_stale(t: &Tensor) -> bool {
    staleness(t).is_some()
}

enum Stale {
    History,
    Data,
}

fn staleness(t: &Tensor) -> Option<Stale> {
    let view = t.view_info()?;
    if meta::history_gen(&view.base) != view.base_history_gen {
        return Some(Stale::History);
    }
    // Views share their root base's storage until `set_data` replaces the base's.
    let shared = Arc::ptr_eq(&t.data().storage, &view.base.data().storage);
    (!shared).then_some(Stale::Data)
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
///
/// If `t` called `retain_grad`, its registration moves to the new node, as in PyTorch: the old
/// history no longer stores into `t.grad`. A tensor that called `retain_grad` and was then
/// detached in place has no old history here (PyTorch fails an internal assertion); it is
/// registered with the new node, which keeps `retains_grad` true to its word, while the
/// registration on its old history stays, as `detach_` left it.
pub(crate) fn rebase_history(t: &Tensor, node: NodeRef, output_nr: u32) {
    debug_assert!(
        t.view_info().is_none(),
        "the history of a view can't be rebased yet"
    );
    let previous = {
        let mut guard = t.autograd_meta();
        let meta = guard.get_or_insert_with(meta::AutogradMeta::default);
        if meta.retains_grad {
            if let Some(old) = &meta.grad_fn {
                old.remove_retained(meta.output_nr, t);
            }
            node.add_retained(output_nr, t);
        }
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
