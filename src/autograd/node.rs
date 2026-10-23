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
use std::sync::{Arc, Mutex, Weak};

use super::saved::{SavedId, SavedTensor};
use crate::tensor::WeakTensor;
use crate::{DType, Device, Shape, Tensor};

/// A backward formula: the gradients of an operation's inputs given the gradients of its
/// outputs.
///
/// A built-in formula holds only plain values (shapes, scalars, dtypes) and the [`SavedId`]s of tensors
/// saved through [`NodeBuilder`], never a `Tensor` or a [`NodeRef`]: keeping every saved tensor in
/// the node lets [`NodeRef::release_saved`] and graph destruction reach all of them.
/// `AccumulateGrad` holds its leaf like PyTorch's. A custom [`Function`](super::Function) node
/// also owns the caller's function value: tensor fields in that value are outside the managed
/// saved slots, and their ownership and lifetime are the implementation's responsibility.
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
    /// The outputs whose gradient is stored in their `.grad` (PyTorch's `retains_grad_hooks_`),
    /// registered by `retain_grad`. The lock is held only to copy or change the list.
    retained: Mutex<Vec<Retained>>,
}

/// A tensor that called `retain_grad` while it was output `output_nr` of the node. Weak, so that
/// the registration doesn't keep the tensor alive; it lapses when the tensor is dropped.
#[derive(Clone)]
pub(crate) struct Retained {
    pub output_nr: u32,
    pub tensor: WeakTensor,
}

static NEXT_SEQUENCE_NR: AtomicU64 = AtomicU64::new(0);

/// Identifies a node while a backward pass holds references to it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct NodeKey(usize);

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

    /// A key that identifies the node for as long as this reference keeps it alive.
    pub fn key(&self) -> NodeKey {
        NodeKey(Arc::as_ptr(self.arc()) as usize)
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

    /// Registers `t`, output `output_nr` of this node, to have its gradient stored in its
    /// `.grad`.
    pub fn add_retained(&self, output_nr: u32, t: &Tensor) {
        self.retained().push(Retained {
            output_nr,
            tensor: t.downgrade(),
        });
    }

    /// Removes the registration of `t` as output `output_nr`, if there is one, leaving those of
    /// other tensors and outputs.
    pub fn remove_retained(&self, output_nr: u32, t: &Tensor) {
        let mut retained = self.retained();
        if let Some(i) = retained
            .iter()
            .position(|r| r.output_nr == output_nr && r.tensor.points_to(t))
        {
            retained.swap_remove(i);
        }
    }

    /// The registered outputs, copied out so that no lock is held while their gradients are
    /// stored.
    pub fn retained_outputs(&self) -> Vec<Retained> {
        self.retained().clone()
    }

    fn retained(&self) -> std::sync::MutexGuard<'_, Vec<Retained>> {
        self.cell()
            .retained
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Frees the saved tensors (`retain_graph = false`). The node and its edges stay alive as
    /// long as references to it remain; running a formula that needs a released tensor fails
    /// with PyTorch's "backward through the graph a second time" error.
    pub fn release_saved(&self) {
        for saved in self.cell().saved.iter() {
            saved.release();
        }
    }

    /// [`call_masked`](NodeRef::call_masked) computing the gradients of all edges.
    #[cfg(test)]
    pub fn call(&self, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        self.call_masked(grads, None)
    }

    /// Runs the formula on the gradients of the forward outputs and returns the gradients for
    /// the edges, with `None` wherever there is no edge.
    ///
    /// With a `mask`, only the gradients of the edges whose entry is true are computed: the
    /// formula sees the others as not needing a gradient ([`ApplyContext::needs_input_grad`])
    /// and they come back as `None`. The engine uses this to skip the branches that don't lead
    /// to the inputs `autograd::grad` asked for, without changing the node, which other graphs
    /// and passes may share.
    ///
    /// No lock is held while the formula runs. If it panics, the node stays structurally valid
    /// and is destroyed normally; it can run again unless its saved tensors have been released.
    ///
    /// # Panics
    ///
    /// Panics if `grads` doesn't have one entry per forward output, if `mask` doesn't have one
    /// entry per edge, or if the formula returns the wrong number of gradients.
    pub fn call_masked(
        &self,
        grads: Vec<Option<Tensor>>,
        mask: Option<&[bool]>,
    ) -> Vec<Option<Tensor>> {
        let cell = self.cell();
        if let Some(mask) = mask {
            assert_eq!(mask.len(), cell.next_edges.len(), "one mask entry per edge");
        }
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
        let mut out = cell
            .formula
            .apply(&ApplyContext { node: self, mask }, grads);
        if out.len() != cell.next_edges.len() {
            panic!(
                "function {} returned an incorrect number of gradients (expected {}, got {})",
                self.name(),
                cell.next_edges.len(),
                out.len()
            );
        }
        for (i, (grad, edge)) in out.iter_mut().zip(&cell.next_edges).enumerate() {
            if edge.is_none() || mask.is_some_and(|m| !m[i]) {
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
    /// The edges this call computes, `None` for all of them.
    mask: Option<&'a [bool]>,
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

    /// Whether input `i` of the forward operation needs a gradient in this call: it has an edge
    /// and the engine wants its gradient. Formulas skip the work for inputs that don't.
    pub fn needs_input_grad(&self, i: usize) -> bool {
        self.node.cell().next_edges[i].is_some() && self.mask.is_none_or(|m| m[i])
    }

    /// The metadata of the forward outputs whose gradients this node receives, as recorded.
    pub fn input_meta(&self) -> &[InputMeta] {
        self.node.input_meta()
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

    /// Whether input `i` has an edge, so that the formula may have to compute its gradient.
    /// Recording code saves the tensors a gradient needs only when this is true.
    pub fn needs_grad(&self, i: usize) -> bool {
        self.next_edges[i].is_some()
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

    /// Moves the saved copies of output `output_nr` to `version`, the version after an in-place
    /// operation that is being recorded has written that output, which saved them before the
    /// write. Saved inputs keep their versions, so an input that shares the target's storage
    /// (`y.mul_(&y)`) is detected as modified.
    pub fn refresh_output(&mut self, output_nr: u32, version: u64) {
        for saved in &mut self.saved {
            saved.refresh_output(output_nr, version);
        }
    }

    fn push(&mut self, saved: SavedTensor) -> SavedId {
        self.saved.push(saved);
        SavedId(self.saved.len() - 1)
    }

    /// Finishes the node; `input_meta` describes each output of the forward operation.
    pub fn build(self, formula: impl Node, input_meta: Vec<InputMeta>) -> NodeRef {
        let sequence_nr = NEXT_SEQUENCE_NR.fetch_add(1, Ordering::Relaxed);
        self.build_with(sequence_nr, formula, input_meta)
    }

    /// Finishes the accumulator of a leaf. Like PyTorch's, it gets the largest sequence number,
    /// so that the engine runs it, and frees the gradient it receives, as soon as it is ready.
    pub fn build_accumulator(self, formula: impl Node, input_meta: InputMeta) -> NodeRef {
        self.build_with(u64::MAX, formula, vec![input_meta])
    }

    fn build_with(
        self,
        sequence_nr: u64,
        formula: impl Node,
        input_meta: Vec<InputMeta>,
    ) -> NodeRef {
        NodeRef(Some(Arc::new(NodeCell {
            sequence_nr,
            formula: Box::new(formula),
            next_edges: self.next_edges,
            input_meta,
            saved: self.saved.into_boxed_slice(),
            retained: Mutex::new(vec![]),
        })))
    }
}

/// The backward node that computed a tensor, PyTorch's `grad_fn`, or the accumulator of a leaf
/// reached through [`next_functions`](GradFn::next_functions). An opaque handle for inspecting
/// the graph; nodes can't be run through it.
///
/// ```
/// let x = homura::ones([2]);
/// x.requires_grad_(true);
/// let y = (&x * 2.0).sum();
/// let grad_fn = y.grad_fn().unwrap();
/// assert_eq!(grad_fn.name(), "SumBackward0");
/// let (mul, _) = grad_fn.next_functions()[0].clone().unwrap();
/// assert_eq!(mul.name(), "MulBackward0");
/// ```
#[derive(Clone)]
pub struct GradFn(pub(crate) NodeRef);

impl GradFn {
    /// The name PyTorch gives this kind of node, such as `"MulBackward0"` or `"AccumulateGrad"`.
    #[track_caller]
    pub fn name(&self) -> &'static str {
        self.0.name()
    }

    /// Where the gradients computed by this node go, one entry per input of the forward
    /// operation: the next node and which of its inputs the gradient is, or `None` where the
    /// input needs no gradient.
    #[track_caller]
    pub fn next_functions(&self) -> Vec<Option<(GradFn, usize)>> {
        self.0
            .next_edges()
            .iter()
            .map(|e| {
                e.as_ref()
                    .map(|e| (GradFn(e.node.clone()), e.input_nr as usize))
            })
            .collect()
    }

    /// Whether `self` and `other` are handles to the same node (Python's `a is b`).
    #[track_caller]
    pub fn is_same(&self, other: &GradFn) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl fmt::Debug for GradFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.name())
    }
}
