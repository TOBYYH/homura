//! Tensors saved for the backward pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::meta;
use super::node::NodeRef;
use crate::tensor::{TensorData, WeakTensor};
use crate::{DType, Device, Tensor};

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
    /// clearing the `.grad` breaks, as in PyTorch). Its data generation when saved tells a later
    /// replacement of its data (`Tensor::set_data`, `Tensor::assign_data`) apart from an in-place
    /// modification, also when the new storage happens to have the saved version; the dtype and
    /// device describe a replacement in the message.
    Leaf {
        tensor: Tensor,
        version: u64,
        data_gen: u64,
        dtype: DType,
        device: Device,
    },
    /// An input with a history: a snapshot of its data without autograd state, its history, and
    /// the flag that `detach_` sets on the original. Not holding the original tensor lets graph
    /// destruction reach `grad_fn` without recursion. The weak handle to the original only names
    /// its current history when the version check fails, as PyTorch's message does.
    NonLeaf {
        data: TensorData,
        version: u64,
        grad_fn: NodeRef,
        output_nr: u32,
        detached: Arc<AtomicBool>,
        original: WeakTensor,
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
                data_gen: t.data_gen(),
                dtype: t.dtype(),
                device: t.device(),
            },
            Some((grad_fn, output_nr, detached)) => Packed::NonLeaf {
                data: t.data(),
                version,
                grad_fn,
                output_nr,
                detached,
                original: t.downgrade(),
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
    /// it was saved; also if it is a leaf whose data was replaced since (see `Tensor::set_data`
    /// and `Tensor::assign_data`).
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
            Some(Packed::Leaf {
                tensor,
                version,
                data_gen,
                dtype,
                device,
            }) => {
                let data = tensor.data();
                if tensor.data_gen() != data_gen {
                    panic!(
                        "the data of a leaf saved for the backward pass was replaced after it was \
                         saved (it was {dtype} on {device}, it is {} on {}); run the forward pass \
                         again",
                        data.dtype,
                        data.device()
                    );
                }
                check_version(&data, version, || meta::grad_fn(&tensor));
                tensor
            }
            Some(Packed::NonLeaf {
                data,
                version,
                grad_fn,
                output_nr,
                detached,
                original,
            }) => {
                if detached.load(Ordering::Acquire) {
                    panic!(
                        "Trying to use a saved tensor that has been detached in-place, i.e. with \
                         .detach_().This is not supported, please use out-of-place `.detach()` \
                         instead"
                    );
                }
                // An in-place operation that modified the original since may have given it a new
                // history, which PyTorch's message names.
                check_version(&data, version, || {
                    let current = original.upgrade().and_then(|t| meta::grad_fn(&t));
                    current.or_else(|| Some((grad_fn.clone(), output_nr)))
                });
                meta::with_saved_history(data, grad_fn, output_nr, detached)
            }
            Some(Packed::Output {
                data,
                version,
                output_nr,
            }) => {
                check_version(&data, version, || Some((owner.clone(), output_nr)));
                meta::with_history(data, owner.clone(), output_nr)
            }
        }
    }

    /// Sets the version of a saved copy of output `output_nr` (see `NodeBuilder::refresh_output`).
    pub(super) fn refresh_output(&mut self, nr: u32, new_version: u64) {
        let slot = self.slot.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(Packed::Output {
            version, output_nr, ..
        }) = slot
            && *output_nr == nr
        {
            *version = new_version;
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

/// Panics with PyTorch's message unless the storage of `data` is still at `version`. A tensor
/// with a `history` is described as an output of the operation that computed it.
fn check_version(
    data: &TensorData,
    version: u64,
    history: impl FnOnce() -> Option<(NodeRef, u32)>,
) {
    let current = data.storage.version();
    if current != version {
        let output = history().map_or(String::new(), |(grad_fn, output_nr)| {
            let operation = operation_name(grad_fn.name());
            format!(", which is output {output_nr} of {operation},")
        });
        panic!(
            "one of the variables needed for gradient computation has been modified by an \
             inplace operation: [{} {:?}]{output} is at version {current}; expected version \
             {version} instead.",
            data.dtype, data.layout.shape
        );
    }
}

/// The name PyTorch's messages give the operation that a node belongs to: the name of a node
/// generated from a formula without `Backward` and a version number of 0, such as `Mul` for
/// `MulBackward0` and `Sum1` for `SumBackward1`; other names, such as `CopyBackwards`, as they
/// are.
fn operation_name(node: &str) -> String {
    match node.rsplit_once("Backward") {
        Some((operation, "0")) => operation.to_string(),
        Some((operation, version)) if version.bytes().all(|b| b.is_ascii_digit()) => {
            format!("{operation}{version}")
        }
        _ => node.to_string(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn operations_are_named_like_pytorch() {
        for (node, operation) in [
            ("MulBackward0", "Mul"),
            ("SumBackward1", "Sum1"),
            ("FillBackward2", "Fill2"),
            ("CopyBackwards", "CopyBackwards"),
            ("AccumulateGrad", "AccumulateGrad"),
        ] {
            assert_eq!(super::operation_name(node), operation);
        }
    }
}
