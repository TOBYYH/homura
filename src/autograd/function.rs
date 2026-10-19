//! Custom differentiable functions (PyTorch's `torch.autograd.Function`).
//!
//! [`Function::apply`] runs the user's forward pass without recording, then records one node for
//! the call with [`NodeBuilder`], like a built-in operation: one edge per input, one slot per
//! output. The tensors the forward pass asked to save are classified by identity once it has
//! returned, so that the node holds only [`SavedId`]s and never its own outputs (which would be a
//! reference cycle): inputs are saved as inputs, differentiable outputs in the node's output
//! slots, non-differentiable outputs without history, and everything else as it is.

use std::any::{TypeId, type_name};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use super::node::{ApplyContext, InputMeta, Node, NodeBuilder};
use super::record::{gradient_edge, set_history};
use super::saved::SavedId;
use super::{is_grad_enabled, no_grad};
use crate::Tensor;

/// A differentiable operation with a hand-written backward pass, PyTorch's
/// `torch.autograd.Function`.
///
/// Implement [`forward`](Function::forward) and [`backward`](Function::backward) and call the
/// provided [`apply`](Function::apply), which consumes the value: its fields carry the
/// non-tensor arguments of the operation (PyTorch's extra `forward` arguments), and `forward`
/// can store values that `backward` needs in them (PyTorch's `ctx.attr = ...`). Tensors that
/// `backward` needs go through [`ForwardContext::save_for_backward`].
///
/// ```
/// use homura::Tensor;
/// use homura::autograd::{BackwardContext, ForwardContext, Function};
///
/// struct Exp;
///
/// impl Function for Exp {
///     fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
///         let y = inputs[0].exp();
///         ctx.save_for_backward(&[&y]);
///         vec![y]
///     }
///
///     fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
///         let y = &ctx.saved_tensors()[0];
///         vec![grads[0].as_ref().map(|g| g * y)]
///     }
/// }
///
/// let x = homura::tensor([0.0, 1.0]);
/// x.requires_grad_(true);
/// let y = Exp.apply(&[&x]).remove(0);
/// assert_eq!(y.grad_fn().unwrap().name(), "ExpBackward");
/// y.sum().backward();
/// assert_eq!(x.grad().unwrap().to_vec::<f32>(), y.detach().to_vec::<f32>());
/// ```
///
/// See [Custom functions](crate::autograd#custom-functions) for the rules: what the outputs may
/// be, how saved tensors behave in higher-order passes, and how this differs from PyTorch.
pub trait Function: Send + Sync + Sized + 'static {
    /// Computes the outputs from `inputs`. Starts in no-grad mode; operations are recorded only
    /// if it explicitly enables grad mode. The call is recorded as one node once it returns.
    ///
    /// # Panics
    ///
    /// [`apply`](Function::apply) panics after this returns if it modified an input in place,
    /// returned a tensor that shares memory with an input or another output, or marked a tensor
    /// that it doesn't return as non-differentiable.
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor>;

    /// Computes the gradient of each input from the gradients of the outputs: `grads` has one
    /// entry per output, `None` for outputs that received no gradient (at least one is `Some`).
    /// Returns one entry per input, `None` for no gradient; more entries are accepted if the extra
    /// ones are all `None`. A gradient for an input that needs none is dropped.
    ///
    /// A returned gradient is checked against its input like the gradients of built-in
    /// operations: a larger shape that the input's shape broadcasts to is summed to it, another
    /// dtype is converted, and another device is accepted only for a zero-dimensional gradient,
    /// which is moved. Runs in grad mode when the backward pass has `create_graph`, so that its
    /// operations are recorded for higher-order derivatives.
    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>>;

    /// Runs [`forward`](Function::forward) on `inputs` and returns its outputs, recorded as one
    /// node named after the type (`MyReluBackward` for `MyRelu`) when grad mode is on and an
    /// input requires grad. Then the floating point outputs that were not marked
    /// non-differentiable require grad and have the node as their `grad_fn`.
    ///
    /// The outputs are new tensors sharing the data and layout of those `forward` returned,
    /// never views, without any history `forward` gave them.
    ///
    /// # Panics
    ///
    /// Panics if `forward` panics, and in the cases listed there. The panic leaves grad mode as it
    /// was and records nothing.
    fn apply(self, inputs: &[&Tensor]) -> Vec<Tensor> {
        apply(self, inputs)
    }
}

/// What [`Function::forward`] can reach besides its inputs: which inputs need gradients, and the
/// tensors to save for [`Function::backward`].
pub struct ForwardContext {
    needs_input_grad: Vec<bool>,
    saved: Vec<Tensor>,
    non_differentiable: Vec<Tensor>,
}

impl ForwardContext {
    /// Whether input `i` requires grad (PyTorch's `ctx.needs_input_grad[i]`). The same in
    /// `backward`, and also when the call isn't recorded because grad mode is off.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range.
    pub fn needs_input_grad(&self, i: usize) -> bool {
        self.needs_input_grad[i]
    }

    /// Saves `tensors` for [`BackwardContext::saved_tensors`]: inputs, outputs, or tensors
    /// computed in between. Call it once; a later call replaces the tensors of an earlier one,
    /// as in PyTorch.
    ///
    /// Saved inputs and outputs are checked for in-place modifications when they are used,
    /// freed after a backward pass without `retain_graph`, and connected to the graph in
    /// higher-order passes. Other tensors are saved as they are, with the same checks. Those
    /// computed with grad mode off have no history: they are constants, so a `backward` that
    /// uses them is correct for the first derivative only. If `forward` explicitly enables grad
    /// mode, other saved tensors retain the history it records.
    pub fn save_for_backward(&mut self, tensors: &[&Tensor]) {
        self.saved = tensors.iter().map(|&t| t.clone()).collect();
    }

    /// Marks outputs as non-differentiable: they won't require grad, and `backward` receives
    /// `None` for them. Call it once; a later call replaces the outputs of an earlier one, as in
    /// PyTorch. Integer and `Bool` outputs never require grad.
    ///
    /// [`Function::apply`] panics if a marked tensor (of the last call) is not one of the tensors
    /// `forward` returns; PyTorch ignores such tensors.
    pub fn mark_non_differentiable(&mut self, outputs: &[&Tensor]) {
        self.non_differentiable = outputs.iter().map(|&t| t.clone()).collect();
    }
}

impl fmt::Debug for ForwardContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardContext")
            .field("needs_input_grad", &self.needs_input_grad)
            .finish_non_exhaustive()
    }
}

/// What [`Function::backward`] can reach: the saved tensors, and which inputs need gradients.
pub struct BackwardContext<'a> {
    ctx: &'a ApplyContext<'a>,
    saved: &'a [SavedId],
    needs_input_grad: &'a [bool],
}

impl BackwardContext<'_> {
    /// Whether input `i` required grad when the function was applied, the same flag
    /// [`ForwardContext::needs_input_grad`] returned. As in PyTorch, it doesn't change when a
    /// backward pass computes the gradients of only some inputs; those of the others are
    /// dropped.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range.
    pub fn needs_input_grad(&self, i: usize) -> bool {
        self.needs_input_grad[i]
    }

    /// The tensors saved by [`ForwardContext::save_for_backward`], in order (PyTorch's
    /// `ctx.saved_tensors`). Each call unpacks them again.
    ///
    /// # Panics
    ///
    /// Panics with PyTorch's messages if the saved tensors were freed by an earlier backward
    /// pass without `retain_graph`, or if one of them was modified in place since it was saved.
    pub fn saved_tensors(&self) -> Vec<Tensor> {
        self.saved.iter().map(|&id| self.ctx.unpack(id)).collect()
    }
}

impl fmt::Debug for BackwardContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackwardContext")
            .field("needs_input_grad", &self.needs_input_grad)
            .finish_non_exhaustive()
    }
}

/// The node recorded for one call of a [`Function`].
struct FunctionNode<F> {
    function: F,
    saved: Vec<SavedId>,
    needs_input_grad: Vec<bool>,
    name: &'static str,
}

impl<F: Function> Node for FunctionNode<F> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, ctx: &ApplyContext<'_>, grads: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let ctx = BackwardContext {
            ctx,
            saved: &self.saved,
            needs_input_grad: &self.needs_input_grad,
        };
        let mut out = self.function.backward(&ctx, &grads);
        // Extra entries are fine as long as they are all `None`, as in PyTorch; anything else is
        // left for the node's count check.
        let n = self.needs_input_grad.len();
        if out.len() > n && out[n..].iter().all(Option::is_none) {
            out.truncate(n);
        }
        out
    }
}

fn apply<F: Function>(mut function: F, inputs: &[&Tensor]) -> Vec<Tensor> {
    let needs_input_grad: Vec<bool> = inputs.iter().map(|t| t.requires_grad()).collect();
    let record = is_grad_enabled() && needs_input_grad.iter().any(|&b| b);
    // Before the user's code runs, so that a stale input view fails without side effects.
    let edges: Vec<_> = if record {
        inputs.iter().map(|t| gradient_edge(t)).collect()
    } else {
        vec![]
    };
    let versions: Vec<u64> = inputs.iter().map(|t| t.version()).collect();
    let mut ctx = ForwardContext {
        needs_input_grad,
        saved: vec![],
        non_differentiable: vec![],
    };
    let outputs = {
        let _guard = no_grad();
        function.forward(&mut ctx, inputs)
    };
    check_forward::<F>(inputs, &versions, &outputs, &ctx.non_differentiable);

    let results: Vec<Tensor> = outputs
        .iter()
        .map(|t| Tensor::from_data(t.data()))
        .collect();
    let differentiable: Vec<bool> = outputs
        .iter()
        .map(|t| {
            record && t.is_floating_point() && !ctx.non_differentiable.iter().any(|m| m.is_same(t))
        })
        .collect();
    if !differentiable.contains(&true) {
        return results;
    }
    let mut builder = NodeBuilder::new(edges);
    let saved = ctx
        .saved
        .iter()
        .map(|t| {
            if let Some(input) = inputs.iter().find(|x| x.is_same(t)) {
                builder.save(input)
            } else if let Some(k) = outputs.iter().position(|o| o.is_same(t)) {
                if differentiable[k] {
                    builder.save_output(&results[k], k as u32)
                } else {
                    builder.save(&results[k])
                }
            } else {
                builder.save(t)
            }
        })
        .collect();
    let input_meta = results.iter().map(InputMeta::of).collect();
    let node = builder.build(
        FunctionNode {
            function,
            saved,
            needs_input_grad: ctx.needs_input_grad,
            name: node_name::<F>(),
        },
        input_meta,
    );
    for (k, t) in results.iter().enumerate() {
        if differentiable[k] {
            set_history(t, node.clone(), k as u32);
        }
    }
    results
}

/// Checks what `forward` did once it has returned (whether or not the call is recorded, so that
/// it behaves the same in every grad mode): it must not have modified an input in place, its
/// outputs must not share memory with an input or each other, and it must not have marked a
/// tensor it doesn't return as non-differentiable.
fn check_forward<F: Function>(
    inputs: &[&Tensor],
    versions: &[u64],
    outputs: &[Tensor],
    non_differentiable: &[Tensor],
) {
    let name = function_name::<F>();
    for (i, (t, &version)) in inputs.iter().zip(versions).enumerate() {
        if t.version() != version {
            panic!(
                "{name}::forward modified input {i} in place, which custom functions don't \
                 support (PyTorch's mark_dirty isn't available): modify a deep_clone() of it \
                 instead"
            );
        }
    }
    for (k, out) in outputs.iter().enumerate() {
        let storage = out.data().storage;
        if let Some(i) = inputs
            .iter()
            .position(|t| Arc::ptr_eq(&t.data().storage, &storage))
        {
            panic!(
                "output {k} of {name}::forward shares memory with input {i}, which custom \
                 functions don't support: return a deep_clone() of it instead"
            );
        }
        if let Some(j) = outputs[..k]
            .iter()
            .position(|t| Arc::ptr_eq(&t.data().storage, &storage))
        {
            panic!(
                "output {k} of {name}::forward shares memory with output {j}, which custom \
                 functions don't support: return a deep_clone() of one of them instead"
            );
        }
    }
    if non_differentiable
        .iter()
        .any(|m| !outputs.iter().any(|o| o.is_same(m)))
    {
        panic!(
            "{name}::forward marked a tensor as non-differentiable that it doesn't return; mark \
             only the tensors it returns"
        );
    }
}

/// The name of `F` in messages: the last segment of its path, without generic arguments
/// (`MyRelu` for `my_crate::ops::MyRelu<f32>`). `type_name` doesn't promise a stable spelling,
/// so the name is for display only.
fn function_name<F: 'static>() -> &'static str {
    short_type_name(type_name::<F>())
}

pub(super) fn short_type_name(full: &str) -> &str {
    let path = full.split('<').next().unwrap_or(full);
    path.rsplit("::").next().unwrap_or(path)
}

/// The name of the nodes of `F`, `{name}Backward` as in PyTorch, created once per type.
fn node_name<F: 'static>() -> &'static str {
    static NAMES: OnceLock<Mutex<HashMap<TypeId, &'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    names.entry(TypeId::of::<F>()).or_insert_with(|| {
        let name = format!("{}Backward", function_name::<F>());
        Box::leak(name.into_boxed_str())
    })
}
