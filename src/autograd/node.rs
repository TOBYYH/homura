//! Backward graph nodes and how they are owned.
//!
//! A node is shared through [`NodeRef`]: edges, the `grad_fn` of tensors and saved non-leaf
//! tensors each hold one. Dropping the last `NodeRef` of a node destroys it and everything only
//! it kept alive with an explicit stack, so a chain of 100k operations doesn't overflow the stack
//! when it is freed. That guarantee covers the nodes themselves: a graph that a node reaches
//! only through a tensor it holds (the `.grad` of a saved leaf or of an accumulator's leaf, or the
//! base of a saved view) is destroyed one level deeper. Arbitrarily deep chains of such tensors
//! are outside the guarantee.
//!
//! Formulas are immutable once recorded, and saved tensors guard their own slots with locks that
//! are held only while copying a handle out. [`NodeRef::call`] therefore runs a formula without
//! holding any lock, so formulas are free to run tensor operations, including recorded ones when
//! the engine builds a higher-order graph.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use super::saved::{SavedId, SavedTensor};
use crate::{DType, Device, Shape, Tensor};

/// A backward formula: the gradients of an operation's inputs given the gradients of its
/// outputs.
///
/// A formula holds only plain values (shapes, scalars, dtypes) and the [`SavedId`]s of tensors
/// saved through [`NodeBuilder`], never a `Tensor` or a [`NodeRef`]: keeping every saved tensor in
/// the node lets [`NodeRef::release_saved`] and graph destruction reach all of them.
/// `AccumulateGrad`, which holds its leaf like PyTorch's, is the only exception.
pub(crate) trait Node: Send + Sync + 'static {
    /// The name PyTorch gives this kind of node, such as `"MulBackward0"`.
    fn name(&self) -> &'static str;

    /// Returns one gradient per input of the forward operation (the node's edges), `None` where
    /// there is no contribution to that input: the input needs no gradient, or the operation
    /// doesn't depend on it. Where the gradient is observably zero (the overwritten value of
    /// `copy_` or `fill_`, a `beta = 0` term), the formula returns a zero tensor instead.
    ///
    /// `grads` has one entry per output of the forward operation, `None` for outputs that
    /// received no gradient; at least one entry is `Some`. Runs without any lock held.
    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>>;
}

/// The shape, dtype and device of one output of the forward operation, which the gradient the
/// node receives for that output must match.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InputMeta {
    pub shape: Shape,
    pub dtype: DType,
    pub device: Device,
}

impl InputMeta {
    pub fn of(t: &Tensor) -> InputMeta {
        let data = t.data();
        InputMeta {
            shape: data.layout.shape.clone(),
            dtype: data.dtype,
            device: data.device(),
        }
    }
}

/// Where a gradient goes: input `input_nr` of `node`.
#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub node: NodeRef,
    pub input_nr: u32,
}

/// A node with its edges and saved tensors, only reachable through [`NodeRef`].
pub(crate) struct NodeCell {
    /// Creation order; the engine runs later nodes first.
    sequence_nr: u64,
    formula: Box<dyn Node>,
    /// One per input of the forward operation, `None` where that input needs no gradient.
    next_edges: Vec<Option<Edge>>,
    /// One per output of the forward operation.
    input_meta: Vec<InputMeta>,
    saved: Box<[SavedTensor]>,
}

static NEXT_SEQUENCE_NR: AtomicU64 = AtomicU64::new(0);

impl NodeCell {
    /// Moves out every node reference this cell owns. References that were the last ones yield
    /// their cells, which the caller destroys in turn.
    fn release_into(&mut self, pending: &mut Vec<NodeCell>) {
        for edge in self.next_edges.drain(..).flatten() {
            pending.extend(edge.node.into_last());
        }
        for saved in self.saved.iter_mut() {
            if let Some(node) = saved.take_node() {
                pending.extend(node.into_last());
            }
        }
    }
}

/// A strong reference to a node.
pub(crate) struct NodeRef(
    /// Always `Some` except while the reference is being dropped.
    Option<Arc<NodeCell>>,
);

/// A weak reference to a node, for example from a leaf to its accumulator.
#[derive(Clone)]
pub(crate) struct WeakNodeRef(Weak<NodeCell>);

impl NodeRef {
    fn arc(&self) -> &Arc<NodeCell> {
        self.0
            .as_ref()
            .expect("node reference used while being dropped")
    }

    fn cell(&self) -> &NodeCell {
        self.arc()
    }

    /// The cell, if this was the last reference to it.
    fn into_last(mut self) -> Option<NodeCell> {
        self.0.take().and_then(Arc::into_inner)
    }

    pub fn ptr_eq(&self, other: &NodeRef) -> bool {
        Arc::ptr_eq(self.arc(), other.arc())
    }

    pub fn downgrade(&self) -> WeakNodeRef {
        WeakNodeRef(Arc::downgrade(self.arc()))
    }

    pub fn name(&self) -> &'static str {
        self.cell().formula.name()
    }

    pub fn sequence_nr(&self) -> u64 {
        self.cell().sequence_nr
    }

    pub fn next_edges(&self) -> &[Option<Edge>] {
        &self.cell().next_edges
    }

    pub fn input_meta(&self) -> &[InputMeta] {
        &self.cell().input_meta
    }

    /// Frees the saved tensors (`retain_graph = false`). The node and its edges stay alive as
    /// long as references to it remain; running a formula that needs a released tensor fails
    /// with PyTorch's "backward through the graph a second time" error.
    pub fn release_saved(&self) {
        for saved in self.cell().saved.iter() {
            saved.release();
        }
    }

    /// Runs the formula on the gradients of the forward outputs and returns the gradients for
    /// the edges, with `None` wherever there is no edge.
    ///
    /// No lock is held while the formula runs. If it panics, the node stays structurally valid
    /// and is destroyed normally; it can run again unless its saved tensors have been released.
    ///
    /// # Panics
    ///
    /// Panics if `grads` doesn't have one entry per forward output, or if the formula returns
    /// the wrong number of gradients.
    pub fn call(&self, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let cell = self.cell();
        assert_eq!(
            grads.len(),
            cell.input_meta.len(),
            "{} received {} gradients for {} outputs",
            self.name(),
            grads.len(),
            cell.input_meta.len()
        );
        if grads.iter().all(Option::is_none) {
            return vec![None; cell.next_edges.len()];
        }
        let mut out = cell.formula.apply(&ApplyContext { node: self }, grads);
        if out.len() != cell.next_edges.len() {
            panic!(
                "function {} returned an incorrect number of gradients (expected {}, got {})",
                self.name(),
                cell.next_edges.len(),
                out.len()
            );
        }
        for (grad, edge) in out.iter_mut().zip(&cell.next_edges) {
            if edge.is_none() {
                *grad = None;
            }
        }
        out
    }
}

impl Clone for NodeRef {
    fn clone(&self) -> Self {
        NodeRef(Some(self.arc().clone()))
    }
}

impl Drop for NodeRef {
    /// Destroys the node if this was its last reference, then every node only it kept alive,
    /// iteratively. `Arc::into_inner` guarantees that exactly one of several threads dropping
    /// the last references concurrently gets the cell.
    fn drop(&mut self) {
        let Some(cell) = self.0.take().and_then(Arc::into_inner) else {
            return;
        };
        let mut pending = vec![cell];
        while let Some(mut cell) = pending.pop() {
            cell.release_into(&mut pending);
            // `cell` is dropped here without any node references left in it.
        }
    }
}

impl fmt::Debug for NodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(cell) => write!(f, "<{} #{}>", cell.formula.name(), cell.sequence_nr),
            None => f.write_str("<dropped node>"),
        }
    }
}

impl WeakNodeRef {
    pub fn upgrade(&self) -> Option<NodeRef> {
        self.0.upgrade().map(|cell| NodeRef(Some(cell)))
    }
}

impl fmt::Debug for WeakNodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.upgrade() {
            Some(node) => write!(f, "Weak({node:?})"),
            None => f.write_str("Weak(<dropped node>)"),
        }
    }
}

/// What a running formula can reach: its saved tensors and which inputs need gradients.
pub(crate) struct ApplyContext<'a> {
    node: &'a NodeRef,
}

impl ApplyContext<'_> {
    /// The tensor saved under `id`, reconnected to the graph (see [`SavedTensor`]).
    ///
    /// # Panics
    ///
    /// Panics if the saved tensors were released or the tensor was modified in place after it
    /// was saved, with PyTorch's error messages.
    pub fn unpack(&self, id: SavedId) -> Tensor {
        self.node.cell().saved[id.0].unpack(self.node)
    }

    /// Whether input `i` of the forward operation needs a gradient. Formulas skip the work for
    /// inputs that don't.
    pub fn needs_input_grad(&self, i: usize) -> bool {
        self.node.cell().next_edges[i].is_some()
    }

    /// The running node. Formulas don't get it, so that they can't release saved tensors.
    #[cfg(test)]
    pub fn node(&self) -> &NodeRef {
        self.node
    }
}

/// Collects the edges and saved tensors of a node while it is recorded.
pub(crate) struct NodeBuilder {
    next_edges: Vec<Option<Edge>>,
    saved: Vec<SavedTensor>,
}

impl NodeBuilder {
    /// Starts a node with one edge per input of the forward operation.
    pub fn new(next_edges: Vec<Option<Edge>>) -> NodeBuilder {
        NodeBuilder {
            next_edges,
            saved: vec![],
        }
    }

    /// Saves a tensor that is not an output of this node, usually an input.
    pub fn save(&mut self, t: &Tensor) -> SavedId {
        self.push(SavedTensor::input(t))
    }

    /// Saves output `output_nr` of this node. The node is not referenced from the saved tensor,
    /// which would be a cycle; unpacking reconnects it to the running node instead.
    pub fn save_output(&mut self, t: &Tensor, output_nr: u32) -> SavedId {
        self.push(SavedTensor::output(t, output_nr))
    }

    fn push(&mut self, saved: SavedTensor) -> SavedId {
        self.saved.push(saved);
        SavedId(self.saved.len() - 1)
    }

    /// Finishes the node; `input_meta` describes each output of the forward operation.
    pub fn build(self, formula: impl Node, input_meta: Vec<InputMeta>) -> NodeRef {
        NodeRef(Some(Arc::new(NodeCell {
            sequence_nr: NEXT_SEQUENCE_NR.fetch_add(1, Ordering::Relaxed),
            formula: Box::new(formula),
            next_edges: self.next_edges,
            input_meta,
            saved: self.saved.into_boxed_slice(),
        })))
    }
}
