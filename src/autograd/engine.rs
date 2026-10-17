//! The backward engine: runs a graph from its roots to its leaves.
//!
//! Like PyTorch's engine, it first counts for every node reachable from the roots how many
//! gradients it will receive, then runs nodes once all of them have arrived, latest created
//! first (accumulators as soon as they are ready). Gradients for the same input of a node are
//! summed in its input buffer. Everything is iterative, so deep graphs don't overflow the stack.
//! Once a node has all its gradients, those of its outputs that called `retain_grad` are stored
//! in their `.grad`.
//!
//! The engine runs on the calling thread. Grad mode is set to `create_graph` for the whole pass,
//! so with `create_graph` the formulas, the conversions of gradients and the sums in the input
//! buffers are recorded in turn.

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet};

use super::accumulate::store_retained;
use super::node::{Edge, InputMeta, NodeKey, NodeRef};
use super::set_grad_enabled;
use crate::Tensor;

/// Runs a backward pass from `roots`, the edges of the differentiated outputs with their
/// gradients.
///
/// With `capture` (`autograd::grad`), the gradients flowing into those edges are returned, one
/// per entry, `None` for entries that are `None` or that no gradient reaches; only the nodes on
/// a path to them run, so no leaf accumulates its gradient. Without it (`backward`), every
/// reachable node runs and the accumulators write the gradients of the leaves. Either way, the
/// gradients of retained non-leaves are stored as soon as their node has received all of its
/// gradients, also when the node is captured without running, as in PyTorch.
///
/// Unless `keep_graph`, each node frees its saved tensors once it has run.
///
/// # Panics
///
/// Panics if a formula panics, or returns a gradient whose shape doesn't match the input it is
/// for.
pub(crate) fn run(
    roots: Vec<(Edge, Tensor)>,
    capture: Option<&[Option<Edge>]>,
    keep_graph: bool,
    create_graph: bool,
) -> Vec<Option<Tensor>> {
    let _grad_mode = set_grad_enabled(create_graph);
    let mut pass = Pass {
        dependencies: count_dependencies(roots.iter().map(|(edge, _)| &edge.node)),
        exec: capture.map(|edges| ExecInfo::new(&roots, edges)),
        buffers: HashMap::new(),
        ready: BinaryHeap::new(),
    };
    let mut captured = vec![None; capture.map_or(0, <[_]>::len)];
    // The gradients of the roots, as PyTorch's graph root node passes them on.
    for (i, (edge, grad)) in roots.into_iter().enumerate() {
        let grad = validate("GraphRoot", i, grad, &meta_of(&edge));
        pass.deliver(&edge, Some(grad));
    }
    while let Some(Ready(node)) = pass.ready.pop() {
        let key = node.key();
        let grads = pass
            .buffers
            .remove(&key)
            .expect("a ready node has a buffer");
        store_retained_outputs(&node, &grads);
        let mut mask = None;
        if let Some(exec) = &pass.exec {
            for &(input_nr, i) in exec.captures.get(&key).into_iter().flatten() {
                captured[i] = grads[input_nr].clone();
            }
            if !exec.needed.contains(&key) {
                continue;
            }
            mask = Some(
                node.next_edges()
                    .iter()
                    .map(|e| e.as_ref().is_some_and(|e| exec.should_run(e.node.key())))
                    .collect::<Vec<_>>(),
            );
        }
        let outputs = node.call_masked(grads, mask.as_deref());
        if !keep_graph {
            node.release_saved();
        }
        for (i, (grad, edge)) in outputs.into_iter().zip(node.next_edges()).enumerate() {
            let Some(edge) = edge else { continue };
            let grad = grad.map(|g| validate(node.name(), i, g, &meta_of(edge)));
            pass.deliver(edge, grad);
        }
    }
    captured
}

/// Stores the gradients that `node` received into the `.grad` of its outputs that called
/// `retain_grad` and are still alive. Outputs that received no gradient keep their `.grad`.
fn store_retained_outputs(node: &NodeRef, grads: &[Option<Tensor>]) {
    for retained in node.retained_outputs() {
        if let Some(grad) = &grads[retained.output_nr as usize]
            && let Some(t) = retained.tensor.upgrade()
        {
            store_retained(&t, grad);
        }
    }
}

/// The state of one backward pass.
struct Pass {
    /// How many gradients each node still waits for.
    dependencies: HashMap<NodeKey, usize>,
    /// Which nodes run, for `autograd::grad`; every node runs for `backward`.
    exec: Option<ExecInfo>,
    /// The gradients received so far by each node that waits for more, one slot per output of
    /// its forward operation.
    buffers: HashMap<NodeKey, Vec<Option<Tensor>>>,
    ready: BinaryHeap<Ready>,
}

impl Pass {
    /// Hands `grad` (`None` if the sender computed none) to input `edge.input_nr` of
    /// `edge.node`, which becomes ready when it has received all its gradients.
    fn deliver(&mut self, edge: &Edge, grad: Option<Tensor>) {
        let key = edge.node.key();
        let remaining = self
            .dependencies
            .get_mut(&key)
            .expect("the dependencies of every reachable node are counted");
        *remaining -= 1;
        let ready = *remaining == 0;
        if self.exec.as_ref().is_some_and(|e| !e.should_run(key)) {
            return;
        }
        let buffer = self
            .buffers
            .entry(key)
            .or_insert_with(|| vec![None; edge.node.input_meta().len()]);
        if let Some(grad) = grad {
            let slot = &mut buffer[edge.input_nr as usize];
            *slot = Some(match slot.take() {
                Some(previous) => previous.add(&grad),
                None => grad,
            });
        }
        if ready {
            self.ready.push(Ready(edge.node.clone()));
        }
    }
}

/// Counts, for every node reachable from `roots`, the edges that lead to it (including one per
/// root).
fn count_dependencies<'a>(roots: impl Iterator<Item = &'a NodeRef>) -> HashMap<NodeKey, usize> {
    let mut dependencies = HashMap::new();
    let mut stack = vec![];
    let mut visit = |node: &NodeRef, stack: &mut Vec<NodeRef>| match dependencies.entry(node.key())
    {
        Entry::Occupied(mut e) => *e.get_mut() += 1,
        Entry::Vacant(e) => {
            e.insert(1);
            stack.push(node.clone());
        }
    };
    for root in roots {
        visit(root, &mut stack);
    }
    while let Some(node) = stack.pop() {
        for edge in node.next_edges().iter().flatten() {
            visit(&edge.node, &mut stack);
        }
    }
    dependencies
}

/// Which nodes an `autograd::grad` pass runs (PyTorch's `exec_info`).
struct ExecInfo {
    /// Nodes with a path to a captured edge, which run.
    needed: HashSet<NodeKey>,
    /// The nodes whose incoming gradients are captured: which of their inputs, and for which
    /// requested input. A captured node runs only if it is also needed.
    captures: HashMap<NodeKey, Vec<(usize, usize)>>,
}

impl ExecInfo {
    fn new(roots: &[(Edge, Tensor)], capture: &[Option<Edge>]) -> ExecInfo {
        let mut captures: HashMap<NodeKey, Vec<(usize, usize)>> = HashMap::new();
        for (i, edge) in capture.iter().enumerate() {
            if let Some(edge) = edge {
                captures
                    .entry(edge.node.key())
                    .or_default()
                    .push((edge.input_nr as usize, i));
            }
        }
        // A node is needed if one of its edges leads to a captured or needed node: a post-order
        // walk, with the index of the next edge to look at kept on the stack.
        let mut needed = HashSet::new();
        let mut visited = HashSet::new();
        let mut stack: Vec<(NodeRef, usize)> = vec![];
        for (edge, _) in roots {
            if visited.insert(edge.node.key()) {
                stack.push((edge.node.clone(), 0));
            }
            while let Some((node, next)) = stack.last_mut() {
                let edges = node.next_edges();
                if let Some(edge) = edges.get(*next) {
                    *next += 1;
                    if let Some(edge) = edge
                        && visited.insert(edge.node.key())
                    {
                        let child = edge.node.clone();
                        stack.push((child, 0));
                    }
                    continue;
                }
                let leads_to_capture = edges.iter().flatten().any(|e| {
                    let key = e.node.key();
                    captures.contains_key(&key) || needed.contains(&key)
                });
                if leads_to_capture {
                    needed.insert(node.key());
                }
                stack.pop();
            }
        }
        ExecInfo { needed, captures }
    }

    /// Whether gradients for `key` are collected: it runs or a gradient is captured there.
    fn should_run(&self, key: NodeKey) -> bool {
        self.needed.contains(&key) || self.captures.contains_key(&key)
    }
}

/// What the gradient for `edge` must look like: the forward output of `edge.node` it is for.
fn meta_of(edge: &Edge) -> InputMeta {
    edge.node.input_meta()[edge.input_nr as usize].clone()
}

/// Checks the gradient that node `name` computed for its edge `index` against what the receiving
/// input expects (PyTorch's `validate_outputs`) and converts it to the input's dtype and device,
/// so that formulas don't have to undo type promotion.
fn validate(name: &str, index: usize, grad: Tensor, meta: &InputMeta) -> Tensor {
    let shape = grad.shape();
    if shape != meta.shape {
        panic!(
            "Function {name} returned an invalid gradient at index {index} - got {shape:?} but \
             expected shape compatible with {:?}",
            meta.shape
        );
    }
    if grad.dtype() == meta.dtype && grad.device() == meta.device {
        grad
    } else {
        grad.to((meta.dtype, meta.device))
    }
}

/// A node in the ready queue, ordered by sequence number: later nodes run first.
struct Ready(NodeRef);

impl PartialEq for Ready {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ready {}

impl PartialOrd for Ready {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ready {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.sequence_nr().cmp(&other.0.sequence_nr())
    }
}
