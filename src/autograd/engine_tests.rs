//! Engine tests that need the crate internals: no autograd state on the fast path, freeing of
//! saved tensors, destruction of deep recorded graphs on a small stack, the lifetime of graphs
//! stored in `.grad`, and the engine's checks of the gradients formulas return.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread;

use super::node::{ApplyContext, InputMeta, Node, NodeBuilder, NodeRef, WeakNodeRef};
use super::record::{gradient_edge, set_history};
use super::{BackwardOptions, GradOptions, meta};
use crate::{DType, Tensor};

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn has_state(t: &Tensor) -> bool {
    t.autograd_meta().is_some()
}

fn node_of(t: &Tensor) -> NodeRef {
    meta::grad_fn(t).expect("a recorded tensor").0
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// Runs `f` on a thread with a small stack, which recursion over a deep graph would overflow.
fn on_small_stack(f: impl FnOnce() + Send + 'static) {
    thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn operations_on_tensors_that_dont_require_grad_create_no_autograd_state() {
    let x = Tensor::from_slice(&[1f32, 2., 3., 4.], [2, 2]);
    let w = homura_like(&x);
    let mut outputs = vec![
        &x + &w,
        &x * 2.0,
        -&x,
        x.exp(),
        x.log(),
        x.sum(),
        x.sum_to_size([2]),
        x.expand([3, 2, 2]),
        x.view([4]),
        x.transpose(0, 1).reshape([4]),
        x.deep_clone(),
        x.to(DType::F64),
        x.matmul(&w),
        x.index((.., 0)),
        x.repeat([2, 1]),
        x.clamp(0.0, 2.0),
    ];
    outputs.extend(x.chunk(2, 0));
    for (i, out) in outputs.iter().enumerate() {
        assert!(!has_state(out), "output {i}");
    }
    assert!(!has_state(&x) && !has_state(&w));
    // In no-grad mode, inputs that require grad don't give their outputs any state either. A
    // view still requires grad through its base.
    let r = leaf(&[1., 2.]);
    let _no_grad = crate::no_grad();
    for out in [&r * &r, r.exp(), r.sum(), r.deep_clone()] {
        assert!(!has_state(&out) && !out.requires_grad());
    }
    let v = r.view([2, 1]);
    assert!(!has_state(&v) && v.requires_grad());
}

/// A tensor of the same shape with other values.
fn homura_like(x: &Tensor) -> Tensor {
    crate::full_like(x, 0.5)
}

#[test]
fn a_backward_pass_frees_saved_tensors_but_not_the_nodes() {
    let x = leaf(&[1., 2.]);
    let a = x.exp();
    let storage = a.data().storage;
    let y = (&a * &x).sum();
    let mul = node_of(&y).next_edges()[0]
        .as_ref()
        .unwrap()
        .node
        .downgrade();
    let before = Arc::strong_count(&storage);
    let retain = BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    super::backward(&[&y], &[], retain);
    assert_eq!(Arc::strong_count(&storage), before);
    y.backward();
    // `exp` saved its output and `mul` saved `a`: both are freed.
    assert_eq!(Arc::strong_count(&storage), before - 2);
    assert!(
        mul.upgrade().is_some(),
        "the nodes live as long as the output"
    );
    drop(y);
    assert!(mul.upgrade().is_none());
}

const DEPTH: usize = 100_000;

/// Builds a recorded chain of `DEPTH` operations on `first`.
fn chain(first: &Tensor, kind: &str) -> Tensor {
    let w = t(&[1.]);
    let mut y = first.clone();
    for _ in 0..DEPTH {
        y = match kind {
            "add" => &y + 1.0,
            "saved" => &y * &w,
            "view" => y.view([1]),
            "shared" => &y + &y,
            _ => unreachable!(),
        };
    }
    y
}

#[test]
fn deep_recorded_graphs_are_destroyed_without_recursion() {
    for kind in ["add", "saved", "view", "shared"] {
        for scenario in ["dropped", "retained backward", "failed backward"] {
            on_small_stack(move || {
                let x = leaf(&[1.]);
                let start = if scenario == "failed backward" {
                    // A node whose derivative isn't implemented, reached last.
                    x.softmax(0)
                } else {
                    &x * 1.0
                };
                let first = node_of(&start).downgrade();
                let y = chain(&start, kind);
                drop(start);
                match scenario {
                    "retained backward" => {
                        let retain = BackwardOptions {
                            retain_graph: Some(true),
                            ..Default::default()
                        };
                        super::backward(&[&y], &[], retain);
                        assert!(x.grad().is_some(), "{kind}");
                    }
                    "failed backward" => {
                        let message = panic_message(|| y.backward());
                        assert_eq!(message, "the derivative for 'softmax' is not implemented.");
                    }
                    _ => {}
                }
                assert!(first.upgrade().is_some());
                drop(y);
                assert!(first.upgrade().is_none(), "{kind}, {scenario}");
            });
        }
    }
}

#[test]
fn create_graph_gradients_and_their_leaves_are_freed_once_the_gradient_is_cleared() {
    // A `.grad` with a graph that reaches the leaf again, cleared before the handles are dropped.
    let x = leaf(&[2.]);
    let (weak_x, graph) = {
        let options = BackwardOptions {
            create_graph: true,
            ..Default::default()
        };
        super::backward(&[&(&x * &x).sum()], &[], options);
        let g = x.grad().unwrap();
        assert!(g.requires_grad());
        (x.downgrade(), node_of(&g).downgrade())
    };
    x.set_grad(None);
    assert!(
        graph.upgrade().is_none(),
        "clearing the gradient frees its graph"
    );
    drop(x);
    assert!(weak_x.upgrade().is_none());
}

#[test]
fn an_uncleared_create_graph_gradient_keeps_its_leaf_alive() {
    let (weak_x, graph): (_, WeakNodeRef) = {
        let x = leaf(&[2.]);
        let options = BackwardOptions {
            create_graph: true,
            ..Default::default()
        };
        super::backward(&[&(&x * &x).sum()], &[], options);
        let graph = node_of(&x.grad().unwrap()).downgrade();
        (x.downgrade(), graph)
    };
    // The cycle leaf -> .grad -> graph -> leaf, as in PyTorch.
    let x = weak_x
        .upgrade()
        .expect("kept alive by its gradient's graph");
    assert!(graph.upgrade().is_some());
    x.set_grad(None);
    drop(x);
    assert!(weak_x.upgrade().is_none() && graph.upgrade().is_none());
}

#[test]
fn create_graph_results_of_grad_are_freed_with_their_handles() {
    let x = leaf(&[2.]);
    let options = GradOptions {
        create_graph: true,
        ..Default::default()
    };
    let g = super::grad(&[&(&x * &x).sum()], &[&x], options)
        .remove(0)
        .unwrap();
    assert!(x.grad().is_none());
    let graph = node_of(&g).downgrade();
    let weak_x = x.downgrade();
    drop(x);
    assert!(
        weak_x.upgrade().is_some(),
        "the graph of `g` holds the leaf"
    );
    drop(g);
    assert!(graph.upgrade().is_none() && weak_x.upgrade().is_none());
}

/// Returns a gradient of the wrong shape.
struct WrongShape;

impl Node for WrongShape {
    fn name(&self) -> &'static str {
        "WrongShapeBackward"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, _grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        vec![Some(t(&[1., 1.]))]
    }
}

#[test]
fn the_engine_checks_and_converts_the_gradients_formulas_return() {
    let x = leaf(&[1., 2., 3.]);
    let y = t(&[0.]);
    let node = NodeBuilder::new(vec![gradient_edge(&x)]).build(WrongShape, vec![InputMeta::of(&y)]);
    set_history(&y, node, 0);
    assert_eq!(
        panic_message(|| y.backward()),
        "Function WrongShapeBackward returned an invalid gradient at index 0 - got [2] but \
         expected shape compatible with [3]"
    );
}

#[test]
fn pruning_for_grad_leaves_the_graph_unchanged() {
    let x = leaf(&[1., 2.]);
    let w = leaf(&[3., 4.]);
    let y = (&x * &w).sum();
    let retain = GradOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    let g = super::grad(&[&y], &[&x], retain).remove(0).unwrap();
    assert_eq!(g.to_vec::<f32>(), [3., 4.]);
    y.backward();
    assert_eq!(x.grad().unwrap().to_vec::<f32>(), [3., 4.]);
    assert_eq!(w.grad().unwrap().to_vec::<f32>(), [1., 2.]);
}
