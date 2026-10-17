//! Tests of `retain_grad` that the public API can't observe: registrations don't keep tensors
//! alive, the reference cycles that stored gradients can and can't form (documented in the
//! `autograd` module), and outputs of one node told apart by `output_nr`.

use super::meta;
use super::node::{ApplyContext, InputMeta, Node, NodeBuilder};
use super::record::{gradient_edge, set_history};
use crate::Tensor;
use crate::autograd::{self, GradOptions};

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_vec()
}

fn grad_creating_graph(out: &Tensor, inputs: &[&Tensor], grad_output: Option<Tensor>) {
    let options = GradOptions {
        grad_outputs: vec![grad_output],
        create_graph: true,
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options);
}

#[test]
fn registrations_do_not_keep_tensors_alive() {
    let x = leaf(&[1., 2.]);
    let y = &x * 2.0;
    y.retain_grad();
    let z = (&y * 3.0).sum();
    let (weak, node) = (y.downgrade(), meta::grad_fn(&y).unwrap().0);
    drop(y);
    assert!(weak.upgrade().is_none());
    assert_eq!(node.retained_outputs().len(), 1, "lapsed, not removed");
    // The pass runs through the node and stores nothing.
    z.backward();
    assert_eq!(vals(&x.grad().unwrap()), [6., 6.]);
}

#[test]
fn gradients_stored_with_create_graph_are_freed_with_their_tensor() {
    let x = leaf(&[1., 2.]);
    let y = &x * &x;
    y.retain_grad();
    grad_creating_graph(&(&y * &y).sum(), &[&x], None);
    let stored = y.grad().unwrap();
    assert_eq!(stored.grad_fn().unwrap().name(), "CloneBackward0");
    let weak_stored = stored.downgrade();
    let weak_node = meta::grad_fn(&y).unwrap().0.downgrade();
    let weak = y.downgrade();
    // The stored gradient's graph saved y as a data copy, which holds y's node, not y.
    drop((y, stored));
    assert!(weak.upgrade().is_none());
    assert!(weak_stored.upgrade().is_none());
    assert!(weak_node.upgrade().is_none());
}

#[test]
fn a_stored_gradient_that_reaches_its_tensor_keeps_it_alive_until_cleared() {
    // Round 34: the leaf view `v` of y, used in the gradient's computation, holds y as its base,
    // and the stored gradient's graph holds `v` through its accumulator.
    let x = leaf(&[1.]);
    let y = &x * 2.0;
    y.retain_grad();
    let v = {
        let _no_grad = crate::no_grad();
        y.view([1])
    };
    v.requires_grad_(true);
    let u = &v * 3.0;
    grad_creating_graph(&y, &[&x], Some(u.clone()));
    let weak = y.downgrade();
    drop((y, v, u));
    let y = weak.upgrade().expect("kept alive by its stored gradient");
    y.set_grad(None);
    drop(y);
    assert!(weak.upgrade().is_none());
}

/// A node with two outputs, both computed from one input, that sends the sum of the gradients it
/// receives (none counting as zero) to its input.
struct TwoOutputs;

impl Node for TwoOutputs {
    fn name(&self) -> &'static str {
        "TwoOutputsBackward0"
    }

    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let sum = grads.into_iter().flatten().reduce(|a, b| a.add(&b));
        vec![sum]
    }
}

/// Outputs 0 and 1 of a [`TwoOutputs`] node on `x`.
fn two_outputs(x: &Tensor) -> (Tensor, Tensor) {
    let (a, b) = (x.detach().deep_clone(), x.detach().deep_clone());
    let node = NodeBuilder::new(vec![gradient_edge(x)])
        .build(TwoOutputs, vec![InputMeta::of(&a), InputMeta::of(&b)]);
    set_history(&a, node.clone(), 0);
    set_history(&b, node, 1);
    (a, b)
}

#[test]
fn outputs_of_one_node_store_their_own_gradients() {
    let x = leaf(&[1., 2.]);
    let (a, b) = two_outputs(&x);
    let node = meta::grad_fn(&a).unwrap().0;
    assert!(node.ptr_eq(&meta::grad_fn(&b).unwrap().0));
    // Another tensor that is output 0 too, as the unpacked copy of a saved output is.
    let twin = meta::with_history(a.data(), node.clone(), 0);
    b.retain_grad();
    twin.retain_grad();
    a.retain_grad();
    // Only one output receives a gradient: the other keeps None, not zeros.
    (&a * 2.0).sum().backward();
    assert_eq!(vals(&a.grad().unwrap()), [2., 2.]);
    assert_eq!(vals(&twin.grad().unwrap()), [2., 2.]);
    assert!(b.grad().is_none());
    (&(&a * 2.0) + &(&b * 3.0)).sum().backward();
    assert_eq!(vals(&a.grad().unwrap()), [4., 4.]);
    assert_eq!(vals(&twin.grad().unwrap()), [4., 4.]);
    assert_eq!(vals(&b.grad().unwrap()), [3., 3.]);

    // An in-place operation on one output moves only its own registration.
    let existing = a.grad().unwrap();
    a.add_(1.0);
    let retained = node.retained_outputs();
    assert_eq!(retained.len(), 2);
    assert!(
        retained
            .iter()
            .any(|r| r.output_nr == 1 && r.tensor.points_to(&b))
    );
    assert!(
        retained
            .iter()
            .any(|r| r.output_nr == 0 && r.tensor.points_to(&twin))
    );
    let new = meta::grad_fn(&a).unwrap().0;
    let moved = new.retained_outputs();
    assert_eq!(moved.len(), 1);
    assert_eq!(moved[0].output_nr, 0);
    assert!(moved[0].tensor.points_to(&a));
    assert!(
        a.grad().unwrap().is_same(&existing),
        "the stored gradient is kept"
    );
    // The gradient of the new history reaches the old one, where the others are still stored.
    (&(&a * 5.0) + &(&b * 7.0)).sum().backward();
    assert_eq!(vals(&a.grad().unwrap()), [9., 9.]);
    assert_eq!(vals(&twin.grad().unwrap()), [9., 9.]);
    assert_eq!(vals(&b.grad().unwrap()), [10., 10.]);
}
