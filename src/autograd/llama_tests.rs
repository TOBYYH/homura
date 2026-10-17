//! The release check of the Llama decoder layer's graph, which needs crate-internal weak handles:
//! after a backward pass that doesn't keep the graph, dropping the last outside handles frees
//! every node, every tensor of the layer and every gradient. The values are checked against
//! PyTorch in `tests/llama_layer.rs`, with the same layer.

use std::collections::HashSet;

use super::meta;
use super::node::{NodeKey, NodeRef, WeakNodeRef};
use crate::autograd::{self, BackwardOptions};
use crate::tensor::WeakTensor;
use crate::{DType, Tensor};

#[path = "test_support/llama.rs"]
mod llama;

use llama::{BATCH, HIDDEN, Layer, SEQ};

/// Weak handles of every node reachable from `roots`, each once.
fn reachable_nodes(roots: &[&Tensor]) -> Vec<WeakNodeRef> {
    let mut seen: HashSet<NodeKey> = HashSet::new();
    let mut stack: Vec<NodeRef> = roots
        .iter()
        .filter_map(|t| meta::grad_fn(t).map(|(node, _)| node))
        .collect();
    let mut found = Vec::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node.key()) {
            continue;
        }
        found.push(node.downgrade());
        stack.extend(node.next_edges().iter().flatten().map(|e| e.node.clone()));
    }
    found
}

fn input(seed: f64) -> Tensor {
    let n = (BATCH * SEQ * HIDDEN) as i64;
    let shape = [BATCH as i64, SEQ as i64, HIDDEN as i64];
    (crate::arange_with(n, DType::F32) * 0.21 + seed)
        .sin()
        .view(shape)
        .deep_clone()
}

#[test]
fn the_layer_graph_is_freed_after_a_backward_pass() {
    let layer = Layer::deterministic();
    let x = input(0.0);
    for t in std::iter::once(&x).chain(layer.weights()) {
        t.requires_grad_(true);
    }
    let f = layer.forward(&x);
    f.h.retain_grad();
    f.probs.retain_grad();
    let (grad_h, grad_out) = (input(1.0), input(2.0));
    autograd::backward(
        &[&f.h, &f.out],
        &[Some(&grad_h), Some(&grad_out)],
        BackwardOptions::default(),
    );

    let nodes = reachable_nodes(&[&f.h, &f.out]);
    // The layer's operations (77 when written) and the accumulators of the ten leaves.
    assert!(nodes.len() > 50, "{} nodes", nodes.len());
    let mut tensors: Vec<(String, WeakTensor)> = Vec::new();
    let mut keep = |name: &str, t: &Tensor| tensors.push((name.to_string(), t.downgrade()));
    for (name, t) in [("h", &f.h), ("probs", &f.probs), ("out", &f.out), ("x", &x)] {
        keep(name, t);
        if let Some(g) = t.grad() {
            keep(&format!("{name}.grad"), &g);
        }
    }
    for (name, w) in llama::WEIGHT_NAMES.iter().zip(layer.weights()) {
        keep(name, w);
        keep(&format!("{name}.grad"), &w.grad().expect("a gradient"));
    }
    assert_eq!(tensors.len(), 2 * 9 + 7, "out has no gradient");

    drop((layer, x, f, grad_h, grad_out));
    for (name, weak) in &tensors {
        assert!(weak.upgrade().is_none(), "{name} is still alive");
    }
    let alive: Vec<&str> = nodes
        .iter()
        .filter_map(|n| n.upgrade().map(|n| n.name()))
        .collect();
    assert!(alive.is_empty(), "nodes still alive: {alive:?}");
}
