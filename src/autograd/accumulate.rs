//! Accumulating gradients into leaves.

use std::sync::Mutex;

use super::is_grad_enabled;
use super::meta::{self, AutogradMeta};
use super::node::{ApplyContext, InputMeta, Node, NodeBuilder, NodeRef, WeakNodeRef};
use crate::Tensor;

/// Adds the gradients that reach a leaf to its `.grad`.
///
/// Holds the leaf, like PyTorch's `AccumulateGrad`, while the leaf refers to it only weakly.
///
/// Lifetime: `backward` with `create_graph = true` stores a `.grad` that has a graph, and that
/// graph usually reaches the leaf again (through this node or a saved leaf). The leaf and its
/// gradient then keep each other alive until the gradient is cleared with `set_grad(None)`,
/// exactly as in PyTorch. `autograd::grad` never writes `.grad` and has no such cycle.
///
/// Destroying a graph doesn't recurse into nodes, but dropping a node's leaf can drop that
/// leaf's `.grad`, whose graph is then destroyed one level deeper. A chain of leaves whose
/// gradients own graphs that reach the next leaf nests once per leaf; arbitrarily deep chains
/// of this kind are outside the guarantee.
pub(crate) struct AccumulateGrad {
    leaf: Tensor,
    /// Serializes accumulation from concurrent backward passes into this leaf. It is taken
    /// before the leaf's state is read and never while another autograd lock is held. It is the
    /// only autograd lock held while tensor operations run, and nothing those operations do
    /// takes it.
    serial: Mutex<()>,
}

impl Node for AccumulateGrad {
    fn name(&self) -> &'static str {
        "AccumulateGrad"
    }

    /// Checks the leaf's current state, not the state it had when the graph was recorded: a leaf
    /// that has since stopped requiring grad keeps its `.grad` unchanged, and one that an
    /// in-place operation has given a history is rejected, as in PyTorch.
    ///
    /// # Panics
    ///
    /// Panics if the leaf now has a `grad_fn`.
    fn apply(&self, _ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let [Some(grad)] = <[Option<Tensor>; 1]>::try_from(grads).expect("one gradient") else {
            return vec![];
        };
        let _serial = self.serial.lock().unwrap_or_else(|e| e.into_inner());
        if meta::grad_fn(&self.leaf).is_some() {
            panic!("leaf variable has been moved into the graph interior");
        }
        if !meta::requires_grad(&self.leaf) {
            return vec![];
        }
        let new = match meta::grad(&self.leaf) {
            // Never store the incoming gradient itself: it may be an expanded view, which can't
            // be updated in place, or a tensor that the caller still holds.
            None => grad.deep_clone(),
            // With `create_graph`, keep the result connected to the higher-order graph.
            Some(old) if is_grad_enabled() => old.add(&grad),
            Some(old) => {
                old.add_(&grad);
                old
            }
        };
        meta::set_grad(&self.leaf, Some(new));
        vec![]
    }
}

/// The accumulator of the leaf `t`, created on first use. It stays the same node for as long as
/// any graph refers to it, so every edge to `t` (including those of saved copies of `t`) leads
/// to the same `.grad`.
pub(crate) fn grad_accumulator(t: &Tensor) -> NodeRef {
    // Reads the tensor's data lock; done before taking its autograd lock.
    let input_meta = InputMeta::of(t);
    let mut guard = t.autograd_meta();
    let meta = guard.get_or_insert_with(AutogradMeta::default);
    if let Some(node) = meta
        .grad_accumulator
        .as_ref()
        .and_then(WeakNodeRef::upgrade)
    {
        return node;
    }
    let node = NodeBuilder::new(vec![]).build_accumulator(
        AccumulateGrad {
            leaf: t.clone(),
            serial: Mutex::new(()),
        },
        input_meta,
    );
    meta.grad_accumulator = Some(node.downgrade());
    node
}
