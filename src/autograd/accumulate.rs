//! Accumulating gradients into leaves, and storing those of non-leaves that called
//! `retain_grad`.

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
/// exactly as in PyTorch. `autograd::grad` never accumulates into leaves and has no such cycle
/// (the gradients it stores into retained non-leaves can form another one, see
/// [`store_retained`]).
///
/// Destroying a graph doesn't recurse into nodes, but dropping a node's leaf can drop that
/// leaf's `.grad`, whose graph is then destroyed one level deeper. A chain of leaves whose
/// gradients own graphs that reach the next leaf nests once per leaf; arbitrarily deep chains
/// of this kind are outside the guarantee.
pub(crate) struct AccumulateGrad {
    leaf: Tensor,
}

impl Node for AccumulateGrad {
    fn name(&self) -> &'static str {
        "AccumulateGrad"
    }

    /// Checks the leaf's current state, not the state it had when the graph was recorded: a leaf
    /// that has since stopped requiring grad keeps its `.grad` unchanged, and one that an
    /// in-place operation has given a history is rejected, as in PyTorch. So is a leaf whose
    /// data `Module::to` has replaced with another dtype or device since this accumulator was
    /// created (`Tensor::set_data`), which PyTorch accepts: the gradient would no longer match
    /// the leaf. A replacement that went back to the recorded dtype and device is accepted.
    /// A `.grad` whose metadata no longer match the leaf's is rejected too, so that a backward
    /// pass never writes a gradient that doesn't match its leaf.
    ///
    /// # Panics
    ///
    /// Panics if the leaf now has a `grad_fn` or other metadata, or if its `.grad` doesn't match
    /// it.
    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let [Some(grad)] = <[Option<Tensor>; 1]>::try_from(grads).expect("one gradient") else {
            return vec![];
        };
        meta::with_grad_lock(&self.leaf, || {
            if meta::grad_fn(&self.leaf).is_some() {
                panic!("leaf variable has been moved into the graph interior");
            }
            let (recorded, current) = (&ctx.input_meta()[0], InputMeta::of(&self.leaf));
            if *recorded != current {
                panic!(
                    "the data of a leaf was replaced after the graph was recorded (it was {} \
                     {:?} on {}, it is {} {:?} on {}); run the forward pass again",
                    recorded.dtype,
                    recorded.shape,
                    recorded.device,
                    current.dtype,
                    current.shape,
                    current.device
                );
            }
            if !meta::requires_grad(&self.leaf) {
                return;
            }
            let old = meta::grad(&self.leaf);
            // `set_grad` checks the gradient, but it can be a tensor that `Module::to` converted
            // since, as a member of a module.
            if let Some(old) = &old
                && InputMeta::of(old) != current
            {
                let old = InputMeta::of(old);
                panic!(
                    "the .grad of a leaf no longer matches it (the gradient is {} {:?} on {}, the \
                     leaf is {} {:?} on {}); clear it with set_grad(None)",
                    old.dtype, old.shape, old.device, current.dtype, current.shape, current.device
                );
            }
            let new = match old {
                // Never store the incoming gradient itself: it may be an expanded view, which
                // can't be updated in place, or a tensor that the caller still holds.
                None => grad.deep_clone(),
                // With `create_graph`, keep the result connected to the higher-order graph.
                Some(old) if is_grad_enabled() => old.add(&grad),
                Some(old) => {
                    old.add_(&grad);
                    old
                }
            };
            meta::set_grad(&self.leaf, Some(new));
        });
        vec![]
    }
}

/// Stores `grad`, the gradient that a backward pass computed for the non-leaf `t`, which called
/// `retain_grad`, into its `.grad` (PyTorch's retains-grad hook): a copy the first time, then
/// the sum with the previous gradient. Unlike a leaf's, the sum is never computed in place, so
/// that gradients handed out earlier don't change. With `create_graph` both are recorded when
/// their inputs require grad; a constant incoming gradient can still be stored without a graph.
///
/// Lifetime: the node holds `t` only weakly and saves non-leaves as data copies, so storing
/// doesn't make the node keep `t` alive. The stored gradient's graph can still reach `t`
/// through a tensor that holds it, such as a leaf view of `t` (its base) used in the gradient's
/// computation; `t` and its gradient then keep each other alive until the gradient is cleared,
/// with [`autograd::grad`](super::grad) as well.
pub(crate) fn store_retained(t: &Tensor, grad: &Tensor) {
    meta::with_grad_lock(t, || {
        let new = match meta::grad(t) {
            None => grad.deep_clone(),
            Some(old) => old.add(grad),
        };
        meta::set_grad(t, Some(new));
    });
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
    let node =
        NodeBuilder::new(vec![]).build_accumulator(AccumulateGrad { leaf: t.clone() }, input_meta);
    meta.grad_accumulator = Some(node.downgrade());
    node
}
