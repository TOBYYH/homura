//! Tensors saved for the backward pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::meta;
use super::node::NodeRef;
use crate::Tensor;
use crate::tensor::TensorData;

/// The index of a saved tensor in its node, handed out by `NodeBuilder::save`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SavedId(pub(super) usize);

/// A tensor that a node keeps for its backward formula.
///
/// The slot has its own lock, held only while a handle is copied out of it or it is emptied,
/// never while a tensor is built or an operation runs.
pub(crate) struct SavedTensor {
    /// `None` once released.
    slot: Mutex<Option<Packed>>,
}

#[derive(Clone)]
enum Packed {
    /// A leaf, kept as its original handle. Its flag can change after it is saved, for example
    /// be set so that a higher-order graph reaches it, and such a graph must reach the leaf's own
    /// accumulator. A leaf owns no node, but its `.grad` can own a graph (under
    /// `backward(create_graph = true)`, one that saves the leaf again: a reference cycle that only
    /// clearing the `.grad` breaks, as in PyTorch).
    Leaf { tensor: Tensor, version: u64 },
    /// An input with a history: a snapshot of its data without autograd state, its history, and
    /// the flag that `detach_` sets on the original. Not holding the original tensor lets graph
    /// destruction reach `grad_fn` without recursion.
    NonLeaf {
        data: TensorData,
        version: u64,
        grad_fn: NodeRef,
        output_nr: u32,
        detached: Arc<AtomicBool>,
    },
    /// Output `output_nr` of the node that owns this saved tensor. Referencing that node would
    /// be a cycle, so the running node supplies itself when unpacking.
    Output {
        data: TensorData,
        version: u64,
        output_nr: u32,
    },
}

impl SavedTensor {
    /// Saves a tensor that is not an output of the node being recorded.
    pub(super) fn input(t: &Tensor) -> SavedTensor {
        let version = t.version();
        let packed = match meta::history_for_saving(t) {
            None => Packed::Leaf {
                tensor: t.clone(),
                version,
            },
            Some((grad_fn, output_nr, detached)) => Packed::NonLeaf {
                data: t.data(),
                version,
                grad_fn,
                output_nr,
                detached,
            },
        };
        SavedTensor::new(packed)
    }

    /// Saves output `output_nr` of the node being recorded.
    pub(super) fn output(t: &Tensor, output_nr: u32) -> SavedTensor {
        SavedTensor::new(Packed::Output {
            data: t.data(),
            version: t.version(),
            output_nr,
        })
    }

    fn new(packed: Packed) -> SavedTensor {
        SavedTensor {
            slot: Mutex::new(Some(packed)),
        }
    }

    /// The saved tensor, connected to the graph again: a saved leaf is its original handle, and
    /// any other saved tensor is rebuilt from its data snapshot with its history. `owner` is the
    /// node this tensor is saved in.
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's messages if the tensor was released, detached in place after it
    /// was saved (checked even if the original was dropped since), or modified in place after
    /// it was saved.
    pub(super) fn unpack(&self, owner: &NodeRef) -> Tensor {
        // Copy handles out under the lock; build the tensor after releasing it.
        let packed = self.slot.lock().unwrap_or_else(|e| e.into_inner()).clone();
        match packed {
            None => panic!(
                "Trying to backward through the graph a second time (or directly access saved \
                 tensors after they have already been freed). Saved intermediate values of the \
                 graph are freed when you call backward() or autograd::grad(). Set retain_graph \
                 if you need to backward through the graph a second time or if you need to \
                 access saved tensors after calling backward."
            ),
            Some(Packed::Leaf { tensor, version }) => {
                let history = meta::grad_fn(&tensor);
                check_version(&tensor.data(), version, history.as_ref());
                tensor
            }
            Some(Packed::NonLeaf {
                data,
                version,
                grad_fn,
                output_nr,
                detached,
            }) => {
                if detached.load(Ordering::Acquire) {
                    panic!(
                        "Trying to use a saved tensor that has been detached in-place, i.e. with \
                         .detach_().This is not supported, please use out-of-place `.detach()` \
                         instead"
                    );
                }
                let history = (grad_fn, output_nr);
                check_version(&data, version, Some(&history));
                meta::with_saved_history(data, history.0, output_nr, detached)
            }
            Some(Packed::Output {
                data,
                version,
                output_nr,
            }) => {
                let history = (owner.clone(), output_nr);
                check_version(&data, version, Some(&history));
                meta::with_history(data, history.0, output_nr)
            }
        }
    }

    /// Frees the saved tensor. Its handles are dropped after the lock is released.
    pub(super) fn release(&self) {
        let packed = self.slot.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(packed);
    }

    /// Takes the earlier node this saved tensor keeps alive, if any, for iterative graph
    /// destruction. Needs exclusive access, which the node being destroyed has.
    pub(super) fn take_node(&mut self) -> Option<NodeRef> {
        let slot = self.slot.get_mut().unwrap_or_else(|e| e.into_inner());
        match slot.take()? {
            Packed::NonLeaf { grad_fn, .. } => Some(grad_fn),
            _ => None,
        }
    }
}

/// Panics with PyTorch's message unless the storage of `data` is still at `version`.
fn check_version(data: &TensorData, version: u64, history: Option<&(NodeRef, u32)>) {
    let current = data.storage.version();
    if current != version {
        let output = history.map_or(String::new(), |(grad_fn, output_nr)| {
            format!(", which is output {output_nr} of {},", grad_fn.name())
        });
        panic!(
            "one of the variables needed for gradient computation has been modified by an \
             inplace operation: [{} {:?}]{output} is at version {current}; expected version \
             {version} instead.",
            data.dtype, data.layout.shape
        );
    }
}
