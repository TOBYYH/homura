//! Contract tests for the backward graph interfaces: formulas run without locks, saved tensors
//! (including later flag changes and in-place detaching), iterative destruction, the view rules
//! and gradient accumulation. Operations don't record yet, so the tests build nodes directly.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use super::meta::{self, view_of};
use super::node::{ApplyContext, Edge, InputMeta, Node, NodeBuilder, NodeRef};
use super::record::{detach_history, gradient_edge, rebase_history, set_history};
use super::saved::{SavedId, SavedTensor};
use crate::Tensor;

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_vec()
}

/// A leaf that requires grad.
fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    meta::set_requires_grad(&x, true);
    x
}

fn view(t: &Tensor) -> Tensor {
    view_of(t, t.layout())
}

fn edge_to(node: &NodeRef) -> Option<Edge> {
    Some(Edge {
        node: node.clone(),
        input_nr: 0,
    })
}

/// Multiplies the incoming gradient by a saved tensor, if any, and sends it to every edge.
struct Scale {
    saved: Option<SavedId>,
    inputs: usize,
}

impl Node for Scale {
    fn name(&self) -> &'static str {
        "ScaleBackward0"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let mut g = grads[0].clone().expect("one gradient");
        if let Some(id) = self.saved {
            g = g.mul(ctx.unpack(id));
        }
        (0..self.inputs)
            .map(|i| ctx.needs_input_grad(i).then(|| g.clone()))
            .collect()
    }
}

/// A node with `edges` computing a tensor like `like`, optionally saving `save`.
fn node(edges: Vec<Option<Edge>>, save: Option<&Tensor>, like: &Tensor) -> NodeRef {
    let inputs = edges.len();
    let mut builder = NodeBuilder::new(edges);
    let saved = save.map(|s| builder.save(s));
    builder.build(Scale { saved, inputs }, vec![InputMeta::of(like)])
}

/// Panics the first time it runs.
struct FailsOnce {
    saved: SavedId,
    fail: AtomicBool,
}

impl Node for FailsOnce {
    fn name(&self) -> &'static str {
        "FailsOnceBackward"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        if self.fail.swap(false, Ordering::SeqCst) {
            panic!("formula failed");
        }
        vec![Some(grads[0].as_ref().unwrap().mul(ctx.unpack(self.saved)))]
    }
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// Runs `f` on another thread and fails, instead of hanging, if it deadlocks.
fn finishes<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(catch_unwind(AssertUnwindSafe(f)));
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(Ok(value)) => value,
        Ok(Err(payload)) => std::panic::resume_unwind(payload),
        Err(_) => panic!("deadlock: the formula did not finish"),
    }
}

/// Runs `f` on a thread with a small stack, which recursive destruction of a deep graph would
/// overflow.
fn on_small_stack(f: impl FnOnce() + Send + 'static) {
    thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

/// Releases the saved tensors of its own node while running.
struct ReleasesItself {
    saved: SavedId,
}

impl Node for ReleasesItself {
    fn name(&self) -> &'static str {
        "ReleasesItselfBackward"
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let x = ctx.unpack(self.saved);
        // Locks every saved slot of the running node: this deadlocks if the node or a slot is
        // locked while the formula runs.
        ctx.node().release_saved();
        vec![Some(grads[0].as_ref().unwrap().mul(&x))]
    }
}

#[test]
fn formulas_run_without_holding_any_node_lock() {
    let w = leaf(&[1., 2., 3.]);
    let mut builder = NodeBuilder::new(vec![gradient_edge(&w)]);
    let saved = builder.save(&t(&[4., 5., 6.]));
    let node = builder.build(ReleasesItself { saved }, vec![InputMeta::of(&w)]);
    let running = node.clone();
    let grads = finishes(move || running.call(vec![Some(t(&[1., 1., 2.]))]));
    // The unpacked tensor stays valid after its slot is released.
    assert_eq!(vals(grads[0].as_ref().unwrap()), [4., 5., 12.]);
    let message = panic_message(|| {
        node.call(vec![Some(t(&[1., 1., 1.]))]);
    });
    assert!(
        message.starts_with("Trying to backward through the graph a second time"),
        "{message}"
    );
}

#[test]
fn only_nodes_that_need_released_values_fail_to_run_again() {
    let w = leaf(&[1., 2.]);
    // Like `w + 1`, which saves nothing, and `w * c`, which saves `c`.
    let add = node(vec![gradient_edge(&w)], None, &w);
    let mul = node(vec![gradient_edge(&w)], Some(&t(&[3., 4.])), &w);
    for node in [&add, &mul] {
        node.call(vec![Some(t(&[1., 1.]))]);
        node.release_saved();
    }
    assert_eq!(
        vals(add.call(vec![Some(t(&[1., 1.]))])[0].as_ref().unwrap()),
        [1., 1.]
    );
    let message = panic_message(|| {
        mul.call(vec![Some(t(&[1., 1.]))]);
    });
    assert!(
        message.starts_with("Trying to backward through the graph a second time"),
        "{message}"
    );
}

#[test]
fn in_place_changes_to_saved_tensors_are_reported_like_pytorch() {
    let w = leaf(&[1., 2., 3.]);
    let producer = node(vec![gradient_edge(&w)], None, &w);
    let plain = t(&[1., 2., 3.]);
    let non_leaf = meta::with_history(t(&[1., 2., 3.]).data(), producer, 0);
    for (saved, described) in [
        (&plain, "[f32 [3]]"),
        (&w, "[f32 [3]]"),
        (&non_leaf, "[f32 [3]], which is output 0 of ScaleBackward0,"),
    ] {
        let consumer = node(vec![gradient_edge(&w)], Some(saved), &w);
        // In no-grad mode, where in-place operations are allowed on leaves that require grad.
        {
            let _no_grad = crate::no_grad();
            saved.add_(1.0);
        }
        let message = panic_message(|| {
            consumer.call(vec![Some(t(&[1., 1., 1.]))]);
        });
        assert_eq!(
            message,
            format!(
                "one of the variables needed for gradient computation has been modified by an \
                 inplace operation: {described} is at version 1; expected version 0 instead."
            )
        );
    }
}

#[test]
fn calls_check_arity_and_skip_formulas_without_gradients() {
    struct ReturnsNothing;
    impl Node for ReturnsNothing {
        fn name(&self) -> &'static str {
            "ReturnsNothingBackward"
        }
        fn apply(&self, _: &ApplyContext<'_>, _: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
            vec![]
        }
    }
    let w = leaf(&[1.]);
    let node =
        NodeBuilder::new(vec![gradient_edge(&w)]).build(ReturnsNothing, vec![InputMeta::of(&w)]);
    // Without any gradient the formula doesn't run.
    assert!(node.call(vec![None])[0].is_none());
    assert_eq!(
        panic_message(|| {
            node.call(vec![Some(t(&[1.]))]);
        }),
        "function ReturnsNothingBackward returned an incorrect number of gradients (expected 1, \
         got 0)"
    );
}

#[test]
fn a_panicking_formula_leaves_the_node_usable_and_destroyable() {
    let w = leaf(&[1., 2.]);
    let mut builder = NodeBuilder::new(vec![gradient_edge(&w)]);
    let saved = builder.save(&t(&[3., 4.]));
    let fails_once = FailsOnce {
        saved,
        fail: AtomicBool::new(true),
    };
    let node = builder.build(fails_once, vec![InputMeta::of(&w)]);
    let message = panic_message(|| {
        node.call(vec![Some(t(&[1., 1.]))]);
    });
    assert_eq!(message, "formula failed");
    assert_eq!(
        vals(node.call(vec![Some(t(&[1., 1.]))])[0].as_ref().unwrap()),
        [3., 4.]
    );
    let weak = node.downgrade();
    drop(node);
    assert!(weak.upgrade().is_none());
}

#[test]
fn saved_leaves_unpack_to_their_original_handle_and_accumulator() {
    let w = leaf(&[2., 3.]);
    // The graph's edge to the leaf, which keeps its accumulator alive.
    let accumulator = gradient_edge(&w).unwrap().node;
    let saved = SavedTensor::input(&w);
    let owner = node(vec![], None, &w);
    let original = w.clone();
    drop(w);
    let unpacked = saved.unpack(&owner);
    assert!(unpacked.is_same(&original));
    assert!(gradient_edge(&unpacked).unwrap().node.ptr_eq(&accumulator));
}

#[test]
fn saved_non_leaves_reconnect_to_the_node_that_computed_them() {
    let w = leaf(&[1.]);
    let producer = node(vec![gradient_edge(&w)], None, &w);
    let y = meta::with_history(t(&[5.]).data(), producer.clone(), 1);
    let saved = SavedTensor::input(&y);
    drop(y);
    let owner = node(vec![], None, &w);
    let unpacked = saved.unpack(&owner);
    let (grad_fn, output_nr) = meta::grad_fn(&unpacked).unwrap();
    assert!(grad_fn.ptr_eq(&producer));
    assert_eq!(output_nr, 1);
    assert_eq!(vals(&unpacked), [5.]);
}

#[test]
fn saved_leaves_follow_later_changes_of_their_flag() {
    let x = t(&[2.]);
    let owner = node(vec![], None, &x);
    let saved = SavedTensor::input(&x);
    assert!(
        x.autograd_meta().is_none(),
        "saving creates no autograd state"
    );
    // Like `x.requires_grad_(True)` after `y = x * z`, before a higher-order pass uses `x`.
    meta::set_requires_grad(&x, true);
    let unpacked = saved.unpack(&owner);
    assert!(unpacked.is_same(&x) && unpacked.requires_grad());
    let accumulator = gradient_edge(&unpacked).unwrap().node;
    assert!(accumulator.ptr_eq(&gradient_edge(&x).unwrap().node));
    {
        let _no_grad = crate::no_grad();
        accumulator.call(vec![Some(t(&[1.]))]);
    }
    assert_eq!(vals(&meta::grad(&x).unwrap()), [1.]);
    // And the other way.
    meta::set_requires_grad(&x, false);
    let unpacked = saved.unpack(&owner);
    assert!(!unpacked.requires_grad());
    assert!(gradient_edge(&unpacked).is_none());
}

#[test]
fn saved_non_leaves_detached_in_place_refuse_to_be_used() {
    const DETACHED: &str = "Trying to use a saved tensor that has been detached in-place, i.e. \
                            with .detach_().This is not supported, please use out-of-place \
                            `.detach()` instead";
    let w = leaf(&[1.]);
    let owner = node(vec![], None, &w);
    let non_leaf =
        || meta::with_history(t(&[2.]).data(), node(vec![gradient_edge(&w)], None, &w), 0);
    let unpack_message = |saved: &SavedTensor| {
        panic_message(|| {
            saved.unpack(&owner);
        })
    };
    // Detached while alive, then also dropped.
    let a = non_leaf();
    let saved = SavedTensor::input(&a);
    detach_history(&a);
    assert!(meta::is_leaf(&a) && !a.requires_grad());
    assert_eq!(unpack_message(&saved), DETACHED);
    drop(a);
    assert_eq!(unpack_message(&saved), DETACHED);
    // Only dropped: the snapshot keeps its history.
    let b = non_leaf();
    let (producer, _) = meta::grad_fn(&b).unwrap();
    let saved = SavedTensor::input(&b);
    drop(b);
    let (grad_fn, _) = meta::grad_fn(&saved.unpack(&owner)).unwrap();
    assert!(grad_fn.ptr_eq(&producer));
    // Given a new history after detaching: copies saved before stay invalid, new ones are valid.
    let c = non_leaf();
    let before = SavedTensor::input(&c);
    detach_history(&c);
    rebase_history(&c, node(vec![gradient_edge(&w)], None, &w), 0);
    let after = SavedTensor::input(&c);
    assert_eq!(unpack_message(&before), DETACHED);
    assert!(meta::grad_fn(&after.unpack(&owner)).is_some());
    // Modified in place, then detached: the detach error comes first, as in PyTorch.
    let d = non_leaf();
    let saved = SavedTensor::input(&d);
    {
        let _no_grad = crate::no_grad();
        d.add_(1.0);
    }
    rebase_history(&d, node(vec![gradient_edge(&w)], None, &w), 0);
    assert!(unpack_message(&saved).starts_with("one of the variables needed"));
    detach_history(&d);
    assert_eq!(unpack_message(&saved), DETACHED);
    // Unpacked and saved again by a higher-order graph: a later `detach_` of the original is
    // still seen, while only dropping the original leaves both copies usable.
    for detach in [true, false] {
        let e = non_leaf();
        let first = SavedTensor::input(&e);
        let second = SavedTensor::input(&first.unpack(&owner));
        first.release();
        if detach {
            detach_history(&e);
        }
        drop(e);
        if detach {
            assert_eq!(unpack_message(&second), DETACHED);
        } else {
            assert!(meta::grad_fn(&second.unpack(&owner)).is_some());
        }
    }
}

#[test]
fn saved_outputs_reconnect_to_the_running_node_without_a_cycle() {
    struct UsesOutput {
        output: SavedId,
    }
    impl Node for UsesOutput {
        fn name(&self) -> &'static str {
            "UsesOutputBackward"
        }
        fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
            let out = ctx.unpack(self.output);
            let (grad_fn, output_nr) = meta::grad_fn(&out).expect("a saved output has history");
            assert!(grad_fn.ptr_eq(ctx.node()));
            assert_eq!(output_nr, 0);
            vec![Some(grads[0].as_ref().unwrap().mul(&out))]
        }
    }
    // Like `out = exp(w)`, whose formula needs `out`.
    let w = leaf(&[2.]);
    let out = t(&[3.]);
    let mut builder = NodeBuilder::new(vec![gradient_edge(&w)]);
    let output = builder.save_output(&out, 0);
    let node = builder.build(UsesOutput { output }, vec![InputMeta::of(&out)]);
    set_history(&out, node.clone(), 0);
    assert_eq!(
        vals(node.call(vec![Some(t(&[1.]))])[0].as_ref().unwrap()),
        [3.]
    );
    // Detaching the output in place doesn't affect what its node saved, as in PyTorch.
    detach_history(&out);
    assert_eq!(
        vals(node.call(vec![Some(t(&[1.]))])[0].as_ref().unwrap()),
        [3.]
    );
    let weak = node.downgrade();
    drop(node);
    drop(out);
    assert!(
        weak.upgrade().is_none(),
        "a saved output must not keep its node alive"
    );
}

const DEPTH: usize = 100_000;

/// Builds `DEPTH` nodes on top of `first`, each with `branches` edges to the previous node and,
/// if `save`, the previous node's output saved. Returns the last node.
fn chain(first: &NodeRef, branches: usize, save: bool) -> NodeRef {
    let like = t(&[1.]);
    let data = like.data();
    let mut previous = first.clone();
    for _ in 0..DEPTH {
        let mut builder = NodeBuilder::new((0..branches).map(|_| edge_to(&previous)).collect());
        let saved =
            save.then(|| builder.save(&meta::with_history(data.clone(), previous.clone(), 0)));
        let formula = Scale {
            saved,
            inputs: branches,
        };
        previous = builder.build(formula, vec![InputMeta::of(&like)]);
    }
    previous
}

#[test]
fn deep_graphs_are_destroyed_without_recursion() {
    // A plain chain, a chain whose nodes also save their predecessor's output, and a chain of
    // shared branches.
    for (branches, save) in [(1, false), (1, true), (2, true)] {
        on_small_stack(move || {
            let w = leaf(&[1.]);
            let first = node(vec![gradient_edge(&w)], None, &w);
            let last = chain(&first, branches, save);
            let first_weak = first.downgrade();
            drop(first);
            drop(last);
            assert!(
                first_weak.upgrade().is_none(),
                "branches {branches}, save {save}"
            );
        });
    }
}

#[test]
fn deep_graphs_are_destroyed_after_calls_and_failures() {
    on_small_stack(|| {
        let w = leaf(&[1.]);
        let first = node(vec![gradient_edge(&w)], None, &w);
        let last = chain(&first, 1, true);
        let first_weak = first.downgrade();
        drop(first);
        // A call that succeeds, as in a retained backward pass.
        last.call(vec![Some(t(&[1.]))]);
        let mut builder = NodeBuilder::new(vec![edge_to(&last)]);
        let saved = builder.save(&t(&[1.]));
        let fails_once = FailsOnce {
            saved,
            fail: AtomicBool::new(true),
        };
        let failing = builder.build(fails_once, vec![InputMeta::of(&w)]);
        drop(last);
        let result = catch_unwind(AssertUnwindSafe(|| failing.call(vec![Some(t(&[1.]))])));
        assert!(result.is_err());
        drop(failing);
        assert!(first_weak.upgrade().is_none());
    });
}

#[test]
fn tensors_without_autograd_state_stay_without_it() {
    let x = t(&[1.]);
    let v = view(&x);
    assert!(!x.requires_grad() && !v.requires_grad());
    assert!(gradient_edge(&v).is_none());
    assert!(x.autograd_meta().is_none() && v.autograd_meta().is_none());
}

#[test]
fn a_view_with_its_own_history_keeps_requiring_grad() {
    let base = leaf(&[2.]);
    let v = view(&base);
    // What a recorded view operation does.
    set_history(&v, node(vec![gradient_edge(&base)], None, &base), 0);
    meta::set_requires_grad(&base, false);
    assert!(v.requires_grad());
    assert!(!meta::is_leaf(&v));
}

#[test]
fn an_explicit_leaf_view_accumulates_into_itself_not_into_its_base() {
    let base = t(&[2.]);
    let v = view(&base);
    meta::set_requires_grad(&v, true);
    let w = view(&v);
    assert!(
        w.view_info().unwrap().base.is_same(&base),
        "views record the root base"
    );
    // A recorded view operation creating `w` connects it to `v`'s own accumulator.
    let edge = gradient_edge(&v).unwrap();
    {
        let _no_grad = crate::no_grad();
        edge.node.call(vec![Some(t(&[2.]))]);
    }
    assert_eq!(vals(&meta::grad(&v).unwrap()), [2.]);
    assert!(meta::grad(&base).is_none());
}

#[test]
fn views_without_their_own_history_follow_the_base_but_drop_gradients() {
    // Created in no-grad mode from a base that requires grad, so no view node was recorded.
    let base = leaf(&[2.]);
    let quiet = {
        let _no_grad = crate::no_grad();
        view(&base)
    };
    // Created before the base required grad.
    let late = t(&[3.]);
    let early = view(&late);
    meta::set_requires_grad(&late, true);
    for v in [&quiet, &early] {
        assert!(v.requires_grad());
        assert!(meta::is_leaf(v));
        // Gradients reaching it are dropped. As a backward root it has no edge, which the
        // engine reports as "does not require grad and does not have a grad_fn".
        assert!(gradient_edge(v).is_none());
    }
    // The properties follow later changes of the base's flag.
    meta::set_requires_grad(&base, false);
    assert!(!quiet.requires_grad());
}

#[test]
fn stale_views_are_rejected_wherever_an_edge_is_looked_up() {
    let w = leaf(&[1.]);
    let y = meta::with_history(t(&[2.]).data(), node(vec![gradient_edge(&w)], None, &w), 0);
    let v = view(&y);
    set_history(&v, node(vec![gradient_edge(&y)], None, &y), 0);
    assert!(gradient_edge(&v).is_some());
    // An in-place operation in grad mode rewrites the history of `y`.
    rebase_history(&y, node(vec![gradient_edge(&w)], None, &w), 0);
    // Created after the rebase, but from a stale parent.
    let stale_child = view(&v);
    for stale in [&v, &stale_child] {
        let message = panic_message(|| {
            gradient_edge(stale);
        });
        assert!(
            message.starts_with("a view whose base was modified by an in-place operation"),
            "{message}"
        );
    }
    // A view created from the base after the rebase is current.
    let fresh = view(&y);
    assert!(gradient_edge(&fresh).is_none());
}

#[test]
fn the_first_gradient_is_copied_and_later_ones_are_added_in_place() {
    let w = leaf(&[1., 2., 3.]);
    let accumulator = gradient_edge(&w).unwrap().node;
    // First-order backward runs in no-grad mode.
    let _no_grad = crate::no_grad();
    let expanded = t(&[1.]).expand([3i64]);
    accumulator.call(vec![Some(expanded.clone())]);
    let first = meta::grad(&w).unwrap();
    assert!(first.is_contiguous());
    assert!(!Arc::ptr_eq(
        &first.data().storage,
        &expanded.data().storage
    ));
    accumulator.call(vec![Some(t(&[1., 2., 3.]))]);
    let second = meta::grad(&w).unwrap();
    assert!(second.is_same(&first), "accumulated in place");
    assert_eq!(vals(&second), [2., 3., 4.]);
}

#[test]
fn concurrent_accumulation_into_one_leaf_loses_nothing() {
    let w = leaf(&[0.]);
    let accumulator = gradient_edge(&w).unwrap().node;
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let w = w.clone();
            thread::spawn(move || {
                let node = gradient_edge(&w).unwrap().node;
                let _no_grad = crate::no_grad();
                for _ in 0..100 {
                    node.call(vec![Some(t(&[1.]))]);
                }
                node
            })
        })
        .collect();
    for worker in workers {
        assert!(worker.join().unwrap().ptr_eq(&accumulator));
    }
    assert_eq!(vals(&meta::grad(&w).unwrap()), [800.]);
}

#[test]
fn accumulation_follows_the_leaf_flag_at_backward_time() {
    let w = leaf(&[2.]);
    let accumulator = gradient_edge(&w).unwrap().node;
    let _no_grad = crate::no_grad();
    // Like `y = w * w; w.requires_grad_(False); y.backward()`: nothing is stored.
    meta::set_requires_grad(&w, false);
    accumulator.call(vec![Some(t(&[4.]))]);
    assert!(meta::grad(&w).is_none());
    // Enabled again, the leaf accumulates.
    meta::set_requires_grad(&w, true);
    accumulator.call(vec![Some(t(&[4.]))]);
    assert_eq!(vals(&meta::grad(&w).unwrap()), [4.]);
    // Disabling it keeps the gradient it has.
    meta::set_requires_grad(&w, false);
    accumulator.call(vec![Some(t(&[4.]))]);
    assert_eq!(vals(&meta::grad(&w).unwrap()), [4.]);
}

#[test]
fn a_leaf_moved_into_the_graph_is_rejected_by_its_accumulator() {
    let x = leaf(&[2.]);
    let z = leaf(&[3.]);
    // Like `y = x + 1; x.requires_grad_(False); x.add_(z); y.backward()`.
    let accumulator = gradient_edge(&x).unwrap().node;
    meta::set_requires_grad(&x, false);
    rebase_history(&x, node(vec![gradient_edge(&z)], None, &z), 0);
    let message = panic_message(|| {
        let _no_grad = crate::no_grad();
        accumulator.call(vec![Some(t(&[1.]))]);
    });
    assert_eq!(
        message,
        "leaf variable has been moved into the graph interior"
    );
    assert!(meta::grad(&x).is_none());
}
