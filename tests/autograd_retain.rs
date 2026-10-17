//! `retain_grad` (M1 step 8a): non-leaves that call it store the gradient that backward passes
//! compute for them, in `grad` passes too; stored gradients are summed out of place, keep a graph
//! with `create_graph`, follow the tensor through in-place operations and stay with the old
//! history after `detach_`. tools/golden/check_m1_retain.py checks the reference behavior that
//! these tests encode; the lifetime tests are in src/autograd/retain_tests.rs.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Barrier;
use std::thread;

use homura::autograd::{self, BackwardOptions, GradOptions};
use homura::{DType, Tensor};

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to(DType::F32).to_vec()
}

fn grad_vals(t: &Tensor) -> Vec<f32> {
    vals(&t.grad().expect("a stored gradient"))
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

fn backward_keeping_graph(out: &Tensor) {
    let options = BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    autograd::backward(&[out], &[], options);
}

fn grad_keeping_graph(out: &Tensor, inputs: &[&Tensor]) -> Vec<Option<Tensor>> {
    let options = GradOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
}

fn grad_creating_graph(out: &Tensor, inputs: &[&Tensor]) -> Vec<Option<Tensor>> {
    let options = GradOptions {
        create_graph: true,
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
}

const NO_GRAD: &str = "can't retain_grad on Tensor that has requires_grad=False";

#[test]
fn retain_grad_is_a_no_op_on_leaves_and_rejects_tensors_without_grad() {
    let x = leaf(&[1., 2.]);
    assert!(x.retain_grad().is_same(&x));
    assert!(!x.retains_grad());
    assert_eq!(
        panic_message(|| {
            homura::ones([2]).retain_grad();
        }),
        NO_GRAD
    );
    let z = {
        let _no_grad = homura::no_grad();
        &x * 2.0
    };
    assert_eq!(
        panic_message(|| {
            z.retain_grad();
        }),
        NO_GRAD
    );
    // A view created in no-grad mode requires grad through its base but is a leaf.
    let v = {
        let _no_grad = homura::no_grad();
        x.view([2])
    };
    assert!(v.requires_grad() && v.is_leaf());
    v.retain_grad();
    assert!(!v.retains_grad());
    // Calling it twice registers once.
    let y = &x * 2.0;
    assert!(!y.retains_grad());
    y.retain_grad().retain_grad();
    assert!(y.retains_grad() && y.grad().is_none());
    (&y * 3.0).sum().backward();
    assert_eq!(grad_vals(&y), [3., 3.]);
}

#[test]
fn the_total_gradient_is_stored_and_accumulated_out_of_place() {
    let x = leaf(&[1., 2.]);
    let y = &x + 1.0;
    y.retain_grad();
    let out = (&(&y * 3.0) + &(&(&y * 1.0) * 4.0)).sum();
    backward_keeping_graph(&out);
    let (first, first_x) = (y.grad().unwrap(), x.grad().unwrap());
    assert_eq!(vals(&first), [7., 7.]);
    assert!(first.grad_fn().is_none() && !first.requires_grad());
    backward_keeping_graph(&out);
    assert_eq!(grad_vals(&y), [14., 14.]);
    assert_eq!(vals(&first), [7., 7.], "an earlier gradient doesn't change");
    assert!(!y.grad().unwrap().is_same(&first));
    // Leaves keep accumulating in place: the earlier handle follows.
    assert!(x.grad().unwrap().is_same(&first_x));
    assert_eq!(vals(&first_x), [14., 14.]);
    // Cleared, the next pass starts again.
    y.set_grad(None);
    backward_keeping_graph(&out);
    assert_eq!(grad_vals(&y), [7., 7.]);
    // A gradient set by hand is added to, not modified.
    let preset = homura::full([2], 5.0);
    y.set_grad(Some(preset.clone()));
    out.backward();
    assert_eq!(grad_vals(&y), [12., 12.]);
    assert_eq!(vals(&preset), [5., 5.]);
}

#[test]
fn autograd_grad_stores_retained_gradients_on_its_paths() {
    let x = leaf(&[1., 2.]);
    let y = &x * 2.0;
    y.retain_grad();
    let w = &x * 3.0;
    w.retain_grad();
    let out = (&(&y * &y) + &w).sum();
    grad_keeping_graph(&out, &[&x]);
    assert_eq!(grad_vals(&y), [4., 8.]);
    assert_eq!(grad_vals(&w), [1., 1.]);
    assert!(x.grad().is_none(), "grad doesn't accumulate into leaves");
    // A requested input that retains its gradient stores it too: its node is captured.
    grad_keeping_graph(&out, &[&y]);
    assert_eq!(grad_vals(&y), [8., 16.]);
    assert_eq!(grad_vals(&w), [1., 1.]);
    // Off the path to the requested input, nothing is stored.
    autograd::grad(&[&out], &[&w], GradOptions::default());
    assert_eq!(grad_vals(&y), [8., 16.]);
    assert_eq!(grad_vals(&w), [2., 2.]);
}

#[test]
fn gradients_stored_with_create_graph_can_be_differentiated() {
    let x = leaf(&[1., 2.]);
    let y = &x * &x;
    y.retain_grad();
    grad_creating_graph(&(&y * &y).sum(), &[&x]);
    let first = y.grad().unwrap();
    assert_eq!(first.grad_fn().unwrap().name(), "CloneBackward0");
    assert_eq!(vals(&first), [2., 8.]);
    let d = grad_keeping_graph(&first.sum(), &[&x]).remove(0).unwrap();
    assert_eq!(vals(&d), [4., 8.]);
    // That pass reached the node of y through the y that the gradient's graph saved, and stored
    // there too, as in PyTorch.
    assert_eq!(grad_vals(&y), [4., 10.]);
    assert_eq!(vals(&first), [2., 8.]);
    grad_creating_graph(&(&y * &y).sum(), &[&x]);
    assert_eq!(y.grad().unwrap().grad_fn().unwrap().name(), "AddBackward0");
    assert_eq!(grad_vals(&y), [6., 18.]);

    // Round 33: a second pass through a saved output stores again.
    let x = Tensor::from_slice(&[0.5f64], [1]);
    x.requires_grad_(true);
    let y = x.exp();
    y.retain_grad();
    let g = grad_creating_graph(&y, &[&x]).remove(0).unwrap();
    let first = y.grad().unwrap();
    assert_eq!(first.to_vec::<f64>(), [1.0]);
    autograd::grad(&[&g], &[&x], GradOptions::default());
    assert_eq!(y.grad().unwrap().to_vec::<f64>(), [2.0]);
    assert_eq!(first.to_vec::<f64>(), [1.0]);

    // Round 35: create_graph records only what depends on tensors that require grad, so a
    // constant incoming gradient is stored without a graph.
    let x = leaf(&[1., 2.]);
    let y = &x * 2.0;
    y.retain_grad();
    grad_creating_graph(&y.sum(), &[&x]);
    let stored = y.grad().unwrap();
    assert_eq!(vals(&stored), [1., 1.]);
    assert!(!stored.requires_grad() && stored.grad_fn().is_none());
}

#[test]
fn stored_gradients_are_contiguous_and_keep_the_dtype() {
    let x = homura::zeros([3, 2]);
    x.requires_grad_(true);
    let y = &x * 1.0;
    y.retain_grad();
    let g = homura::arange(6.0).view([2, 3]).transpose(0, 1);
    assert!(!g.is_contiguous());
    y.backward_with(&g);
    let stored = y.grad().unwrap();
    assert!(stored.is_contiguous());
    assert_eq!(vals(&stored), vals(&g.contiguous()));

    let x = homura::tensor_with([1.0, 2.0], DType::F16);
    x.requires_grad_(true);
    let y = &x * 2.0;
    y.retain_grad();
    y.sum().backward();
    assert_eq!(y.grad().unwrap().dtype(), DType::F16);
    assert_eq!(grad_vals(&y), [1., 1.]);
}

#[test]
fn views_store_their_own_gradient() {
    let x = leaf(&[1., 2., 3., 4., 5., 6.]);
    let b = &x * 1.0;
    let v = b.view([2, 3]);
    v.retain_grad();
    (&v * 5.0).sum().backward();
    assert_eq!(grad_vals(&v), [5.; 6]);
    assert_eq!(v.grad().unwrap().shape(), v.shape());
    assert!(b.grad().is_none());
}

#[test]
fn in_place_operations_move_the_registration() {
    let x = leaf(&[1., 2.]);
    let y = &x + 1.0;
    y.retain_grad();
    let old = (&y * 3.0).sum();
    y.set_grad(Some(homura::ones([2])));
    y.add_(1.0);
    // The stored gradient is kept, and the new history stores into it.
    assert!(y.retains_grad());
    assert_eq!(grad_vals(&y), [1., 1.]);
    old.backward();
    assert_eq!(grad_vals(&y), [1., 1.], "the old history no longer stores");
    (&y * 7.0).sum().backward();
    assert_eq!(grad_vals(&y), [8., 8.]);

    // detach_ leaves the registration on the old history, and the flag set.
    let x = leaf(&[1., 2.]);
    let y = &x + 1.0;
    y.retain_grad();
    let old = (&y * 3.0).sum();
    y.detach_();
    assert!(y.retains_grad() && y.is_leaf() && !y.requires_grad());
    old.backward();
    assert_eq!(grad_vals(&y), [3., 3.]);

    // Joining a graph again in place: PyTorch fails an internal assertion and the new history
    // never stores; homura registers the tensor with its new node.
    let x = leaf(&[1., 2.]);
    let y = &x + 1.0;
    y.retain_grad();
    y.detach_();
    y.add_(&x);
    assert_eq!(y.grad_fn().unwrap().name(), "AddBackward0");
    assert!(y.retains_grad());
    (&y * 2.0).sum().backward();
    assert_eq!(grad_vals(&y), [2., 2.]);
}

#[test]
fn stale_views_can_not_start_retaining() {
    let x = leaf(&[1., 2.]);
    let b = &x * 1.0;
    let v = b.view([2]);
    let w = b.view([2]);
    w.retain_grad();
    b.add_(1.0);
    // PyTorch replays the view on the new history of its base; homura doesn't yet.
    let message = panic_message(|| {
        v.retain_grad();
    });
    assert!(message.starts_with("a view whose base was modified by an in-place operation"));
    assert!(!v.retains_grad());
    // A view that retained before keeps its flag, and calling again does nothing.
    assert!(w.retain_grad().retains_grad());
}

/// Stricter than PyTorch, which serializes accumulating into leaves but not storing retained
/// gradients, and loses updates here (see tools/golden/check_m1_retain.py).
#[test]
fn concurrent_backward_passes_lose_no_retained_gradient() {
    let x = homura::zeros([1000]);
    x.requires_grad_(true);
    let y = &x + 1.0;
    y.retain_grad();
    let outs = [(&y * 1.0).sum(), (&y * 2.0).sum()];
    let workers: Vec<_> = outs
        .into_iter()
        .map(|out| {
            thread::spawn(move || {
                for _ in 0..100 {
                    backward_keeping_graph(&out);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(grad_vals(&y), [300.; 1000]);
    assert_eq!(grad_vals(&x), [300.; 1000]);
}

#[test]
fn a_tensor_detached_in_place_takes_both_kinds_of_concurrent_writes() {
    // Round 35: after detach_, the old graph still stores into y (out of place) while y, a leaf
    // now, accumulates the gradients of its new graph (in place). Both writes take y's lock.
    let x = homura::zeros([1000]);
    x.requires_grad_(true);
    let y = &x + 1.0;
    y.retain_grad();
    let old = (&y * 3.0).sum();
    y.detach_();
    y.requires_grad_(true);
    let new = (&y * 5.0).sum();
    let workers: Vec<_> = [old, new]
        .into_iter()
        .map(|out| {
            thread::spawn(move || {
                for _ in 0..100 {
                    backward_keeping_graph(&out);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(grad_vals(&y), [800.; 1000]);
    assert_eq!(grad_vals(&x), [300.; 1000]);
}

#[test]
fn concurrent_calls_register_once() {
    // Round 35: retain_grad checks again under the lock, so threads that call it at the same time
    // register y once, and a backward pass stores its gradient once.
    let x = leaf(&[1., 2.]);
    let y = &x * 2.0;
    let barrier = Barrier::new(8);
    thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                barrier.wait();
                y.retain_grad();
            });
        }
    });
    assert!(y.retains_grad());
    (&y * 3.0).sum().backward();
    assert_eq!(grad_vals(&y), [3., 3.]);
}
