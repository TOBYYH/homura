//! Tests of custom functions that the public API can't observe: node names, and what the saved
//! tensors and the node keep alive. Each test says which handles it keeps: a leaf input's storage
//! stays alive through the leaf and its accumulator, and an output's through the caller's handle,
//! so neither tells whether a saved slot was freed.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, Weak};

use super::function::short_type_name;
use super::meta;
use super::{BackwardContext, BackwardOptions, ForwardContext, Function};
use crate::Tensor;
use crate::storage::Storage;

fn leaf(values: &[f32]) -> Tensor {
    let x = Tensor::from_slice(values, [values.len()]);
    x.requires_grad_(true);
    x
}

#[test]
fn names_are_the_last_path_segment_without_generics() {
    for (full, short) in [
        ("my_crate::ops::MyRelu", "MyRelu"),
        ("my_crate::ops::MyRelu<f32>", "MyRelu"),
        ("a::Wrap<b::Inner<c::D>, e::F>", "Wrap"),
        ("Plain", "Plain"),
    ] {
        assert_eq!(short_type_name(full), short);
    }
}

/// `2x`, saving a hidden intermediate (`3x`) whose storage it lets the test watch; holds a token
/// to show when the function value is dropped.
struct Hidden {
    storage: Arc<Mutex<Weak<Storage>>>,
    _token: Arc<()>,
}

impl Function for Hidden {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let h = inputs[0] * 3.0;
        *self.storage.lock().unwrap() = Arc::downgrade(&h.data().storage);
        ctx.save_for_backward(&[&h, inputs[0]]);
        vec![inputs[0] * 2.0]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        assert_eq!(ctx.saved_tensors().len(), 2);
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

fn hidden(token: &Arc<()>) -> (Hidden, Arc<Mutex<Weak<Storage>>>) {
    let storage = Arc::new(Mutex::new(Weak::new()));
    let f = Hidden {
        storage: storage.clone(),
        _token: token.clone(),
    };
    (f, storage)
}

#[test]
fn names_are_created_once_per_type() {
    let x = leaf(&[1.0]);
    let token = Arc::new(());
    let a = hidden(&token).0.apply(&[&x]).remove(0);
    let b = hidden(&token).0.apply(&[&x]).remove(0);
    let (a, b) = (a.grad_fn().unwrap(), b.grad_fn().unwrap());
    assert_eq!(a.name(), "HiddenBackward");
    assert!(std::ptr::eq(a.name(), b.name()));
}

#[test]
fn released_slots_refuse_to_unpack() {
    // Keeps the output and its node.
    let x = leaf(&[1.0, 2.0]);
    let token = Arc::new(());
    let y = hidden(&token).0.apply(&[&x]).remove(0);
    let (node, _) = meta::grad_fn(&y).unwrap();
    y.sum().backward();
    let payload = catch_unwind(AssertUnwindSafe(|| {
        node.call(vec![Some(crate::ones([2]))]);
    }))
    .expect_err("the saved tensors were freed");
    let message = payload.downcast_ref::<&str>().copied().unwrap_or_default();
    assert!(message.starts_with("Trying to backward through the graph a second time"));
}

#[test]
fn hidden_intermediates_live_only_in_their_slot() {
    // Keeps the output; the hidden tensor has no other handle.
    let x = leaf(&[1.0, 2.0]);
    let token = Arc::new(());
    let (f, storage) = hidden(&token);
    let y = f.apply(&[&x]).remove(0);
    assert!(storage.lock().unwrap().upgrade().is_some());
    y.sum().backward();
    assert!(storage.lock().unwrap().upgrade().is_none());
    // With `retain_graph`, it lives as long as the node.
    let (f, storage) = hidden(&token);
    let y = f.apply(&[&x]).remove(0);
    let options = BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    super::backward(&[&y.sum()], &[], options);
    assert!(storage.lock().unwrap().upgrade().is_some());
    drop(y);
    assert!(storage.lock().unwrap().upgrade().is_none());
}

#[test]
fn nodes_go_with_their_last_reference() {
    // Keeps the output, a `GradFn` and a later node with an edge to the function's node.
    let x = leaf(&[1.0, 2.0]);
    let token = Arc::new(());
    let (f, storage) = hidden(&token);
    let y = f.apply(&[&x]).remove(0);
    let grad_fn = y.grad_fn().unwrap();
    let z = &y * 2.0;
    let node = meta::grad_fn(&y).unwrap().0.downgrade();
    drop(y);
    assert!(node.upgrade().is_some());
    drop(grad_fn);
    assert!(node.upgrade().is_some());
    assert_eq!(Arc::strong_count(&token), 2);
    drop(z);
    assert!(node.upgrade().is_none());
    assert_eq!(Arc::strong_count(&token), 1);
    assert!(storage.lock().unwrap().upgrade().is_none());
}

/// `exp(x)`, saving its output; holds a token to show when the function value is dropped.
struct SavedOutput {
    _token: Arc<()>,
}

impl Function for SavedOutput {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let y = inputs[0].exp();
        ctx.save_for_backward(&[&y]);
        vec![y]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let y = &ctx.saved_tensors()[0];
        vec![grads[0].as_ref().map(|g| g * y)]
    }
}

#[test]
fn saved_outputs_dont_keep_their_node_alive() {
    // Keeps nothing but the output: the saved copy of it must not hold the output itself, whose
    // `grad_fn` is the node holding the copy.
    let x = leaf(&[1.0]);
    let token = Arc::new(());
    let y = SavedOutput {
        _token: token.clone(),
    }
    .apply(&[&x])
    .remove(0);
    let node = meta::grad_fn(&y).unwrap().0.downgrade();
    assert_eq!(Arc::strong_count(&token), 2);
    drop(y);
    assert!(node.upgrade().is_none());
    assert_eq!(Arc::strong_count(&token), 1);
}
