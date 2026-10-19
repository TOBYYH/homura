//! Automatic differentiation, following PyTorch's design: operations on tensors that require
//! grad record a backward graph as they run (define-by-run), and [`backward`] or [`grad`] runs it
//! from the outputs to the inputs.
//!
//! ```
//! use homura::autograd;
//!
//! let x = homura::tensor([1.0, 2.0, 3.0]);
//! x.requires_grad_(true);
//! let y = (&x * &x).sum();
//! y.backward();
//! assert_eq!(x.grad().unwrap().to_vec::<f32>(), [2.0, 4.0, 6.0]);
//!
//! // `grad` returns the gradients instead of accumulating them into the leaves' `.grad`.
//! let y = (&x * &x).sum();
//! let g = autograd::grad(&[&y], &[&x], autograd::GradOptions::default());
//! assert_eq!(g[0].as_ref().unwrap().to_vec::<f32>(), [2.0, 4.0, 6.0]);
//! ```
//!
//! With `create_graph`, the backward pass is recorded too, so gradients can be differentiated
//! again (higher-order derivatives).
//!
//! Only leaves accumulate their gradients into `.grad`; a tensor computed by a recorded operation
//! stores its own after [`Tensor::retain_grad`], in [`grad`] passes too.
//!
//! # Coming from PyTorch
//!
//! | PyTorch | homura |
//! | --- | --- |
//! | `x.requires_grad_()` | [`x.requires_grad_(true)`](Tensor::requires_grad_) |
//! | `y.backward()`, `y.backward(g)` | [`y.backward()`](Tensor::backward), [`y.backward_with(&g)`](Tensor::backward_with) |
//! | `torch.autograd.backward`, `torch.autograd.grad` | [`backward`], [`grad`], with [`BackwardOptions`] and [`GradOptions`] |
//! | `x.grad`, `x.grad = None` | [`x.grad()`](Tensor::grad), [`x.set_grad(None)`](Tensor::set_grad) |
//! | `y.grad_fn`, `y.is_leaf`, `y.retain_grad()` | [`Tensor::grad_fn`], [`Tensor::is_leaf`], [`Tensor::retain_grad`] |
//! | `x.detach()`, `x.detach_()` | [`Tensor::detach`], [`Tensor::detach_`] |
//! | `with torch.no_grad():` | `let _guard = homura::no_grad();` |
//! | `x.clone()` | [`x.deep_clone()`](Tensor::deep_clone) |
//! | `repr(y)` | `format!("{y:?}")`, with the same `grad_fn=<MulBackward0>` or `requires_grad=True` |
//! | `class F(torch.autograd.Function)`, `F.apply(x)` | `impl` [`Function`] `for F`, [`F.apply(&[&x])`](Function::apply) |
//! | `ctx.save_for_backward(x)`, `ctx.saved_tensors` | [`ForwardContext::save_for_backward`], [`BackwardContext::saved_tensors`] |
//! | `ctx.needs_input_grad[i]` | [`ForwardContext::needs_input_grad`], [`BackwardContext::needs_input_grad`] |
//! | `ctx.mark_non_differentiable(y)` | [`ForwardContext::mark_non_differentiable`] |
//! | `ctx.shape = x.shape` in `forward` | a field of the function value, set in [`Function::forward`] |
//!
//! Mind `clone`: [`Tensor`]'s `Clone` copies the handle, like `y = x` in Python. The clone *is*
//! `x`, with its data, history and `.grad`, so an in-place operation through it changes `x` and
//! its history. Port `x.clone()` to `deep_clone`, which copies the data and is recorded as
//! `CloneBackward0`, like PyTorch's.
//!
//! # Current state
//!
//! Every differentiable operation records its node with a backward formula: `add`, `sub`, `mul`,
//! `div`, `pow`, `neg`, `abs`, `sign`, `exp`, `log`, `sqrt`, `rsqrt`, `reciprocal`, `sin`, `cos`,
//! `tanh`, `sigmoid`, `relu`, `silu`, `gelu`, `erf`, `maximum`, `minimum`, `clamp` (and
//! `clamp_min`, `clamp_max`), `where_`, `lerp`, `addcmul`, `addcdiv`, `softmax`, `log_softmax`,
//! `matmul`, `addmm`, `addmv`, `addr`, `baddbmm`, `addbmm`, the reductions `sum`, `sum_dim`,
//! `sum_to_size`, `mean`, `mean_dim`, `prod`, `prod_dim`, `max`, `min`, `max_dim`, `min_dim`,
//! `var`, `var_dim`, `std` and `std_dim`, the views `expand`, `view`, `reshape`, `permute`,
//! `transpose`, `squeeze`, `squeeze_dim`, `unsqueeze`, `select`, `narrow`, slicing and
//! `as_strided`, `index_select`, indexing with an integer tensor, `index_add`, `gather`,
//! `scatter_add`, `select_scatter`, `slice_scatter`, the copies `deep_clone`, `to`, `concat`,
//! `flip`, `triu` and `tril`, and what is built from them, such as `flatten`, `contiguous`,
//! `repeat`, `stack`, `split`, `chunk` and `masked_fill`. Comparisons, logical operations,
//! integer results, creation and random functions and `detach` record nothing.
//!
//! The gradient of `prod` for an input that contains zeros can't be differentiated again yet: a
//! backward pass with `create_graph` that computes it panics.
//!
//! Backward passes run wherever their tensors live. On the meta device they compute only the
//! gradients' shapes, dtypes and devices; a backward pass from a CPU leaf copied to the meta
//! device fails when it reaches the copy, since meta tensors have no data to copy back, as in
//! PyTorch.
//!
//! # In-place operations
//!
//! In grad mode, an in-place operation whose target or inputs require grad is recorded like the
//! out-of-place operation, with the target as its output: the target gets the operation's node
//! as its new history, whose first edge leads to the old one, as in PyTorch. A formula that needs
//! the target's value from before the write (`mul_` for the other factor's gradient, `pow_`,
//! `clamp_`, `lerp_` with a tensor weight) saves a copy of it, with its old history. Writes that
//! replace the value give the old value a zero gradient, as in PyTorch: `copy_` (recorded as
//! `CopyBackwards`, which gives `src` the gradient converted to its dtype and summed to its
//! shape), `fill_`, `zero_`, `uniform_` and `normal_`. A tensor that doesn't require grad
//! becomes part of the graph when it is written from one that does.
//!
//! ```
//! let x = homura::tensor([1.0, 2.0]);
//! x.requires_grad_(true);
//! let y = &x * 3.0;
//! y.mul_(&x);
//! assert_eq!(y.grad_fn().unwrap().name(), "MulBackward0");
//! y.sum().backward();
//! assert_eq!(x.grad().unwrap().to_vec::<f32>(), [6.0, 12.0]);
//! ```
//!
//! Recorded writes are rejected before anything is written, so that the data, version and
//! history of the target don't change: on a leaf that requires grad and on a view of one, with
//! PyTorch's errors, and on any other view, which homura doesn't support yet. In no-grad mode
//! every in-place operation is allowed, which is how optimizers update parameters; the history
//! doesn't change.
//!
//! Every in-place operation bumps the version of the target's storage once. A backward pass that
//! needs a tensor saved before a write panics with PyTorch's error, naming the tensor's current
//! history. A view created before a recorded write into its base (or from such a view) is stale:
//! using its history (as an input to a recorded operation, a root or requested input of a
//! backward pass, the parent of a new view, or through [`Tensor::grad_fn`]) panics, while
//! no-grad operations on its data work. PyTorch replays such views on the new history; create
//! the view again after the write instead. A view whose base
//! [`nn::Module::to`](crate::nn::Module::to) has converted since is stale in the same way: it
//! still refers to the old data.
//!
//! # Views
//!
//! A view of a tensor that requires grad records its own node, which sends gradients to the
//! tensor it was created from. A view created in no-grad mode, or before its base required
//! grad, has no node: it requires grad as long as its base does, but gradients reaching it are
//! dropped, and it can't be the root of a backward pass. Calling `requires_grad_(true)` on such
//! a view makes it a leaf of its own.
//!
//! | View | `requires_grad` | `is_leaf` | `grad_fn` | Gradients reaching it |
//! | --- | --- | --- | --- | --- |
//! | Created in grad mode from a tensor that requires grad | `true` | `false` | its own node, such as `ViewBackward0` | go to its base |
//! | Created in no-grad mode, or before its base required grad | as its base | `true` | `None` | are dropped |
//! | Such a view after `requires_grad_(true)` | `true` | `true` | `None` | accumulate into its own `.grad` |
//! | Stale: its base's history was rewritten in place, or its base was converted by `Module::to`, after it was created | unchanged | unchanged | panics | panic, like every use of its history |
//!
//! In grad mode, any view rejects in-place operations that would be recorded, see
//! [In-place operations](#in-place-operations); in no-grad mode they are allowed.
//!
//! # Custom functions
//!
//! A [`Function`] is an operation with a hand-written backward pass. [`Function::apply`] runs its
//! `forward` in no-grad mode, then, if grad mode is on and an input requires grad, records one
//! node for the call, named after the type (`ExpBackward` for `Exp`), with an edge per input.
//! The outputs are new tensors that share the data and layout of the tensors `forward` returned;
//! the floating point ones that weren't marked with
//! [`mark_non_differentiable`](ForwardContext::mark_non_differentiable) require grad and have
//! the node as their `grad_fn`. Integer, `Bool` and marked outputs never require grad, and
//! `backward` receives `None` for them, as for any output that no gradient reached; when none
//! did, the backward pass doesn't call `backward` at all.
//!
//! [`save_for_backward`](ForwardContext::save_for_backward) tells apart, by identity, the
//! inputs, the outputs and everything else, as PyTorch does. Saved inputs and differentiable
//! outputs behave like the tensors built-in operations save: modifications are detected, they
//! are freed by a backward pass without `retain_graph`, and in a pass with `create_graph` they
//! come back connected to the graph, so a `backward` written with tensor operations on them is
//! differentiable again. Any other saved tensor is saved as it is, with the same checks. An
//! intermediate value computed with grad mode off has no history: it is a constant, correct
//! for the first derivative only. If `forward` explicitly enables grad mode, other saved tensors
//! retain the history it records. Saving the output of `exp`
//! gives every order:
//!
//! ```
//! use homura::Tensor;
//! use homura::autograd::{self, BackwardContext, ForwardContext, Function, GradOptions};
//!
//! struct Exp;
//!
//! impl Function for Exp {
//!     fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
//!         let y = inputs[0].exp();
//!         ctx.save_for_backward(&[&y]);
//!         vec![y]
//!     }
//!
//!     fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
//!         let y = &ctx.saved_tensors()[0];
//!         vec![grads[0].as_ref().map(|g| g * y)]
//!     }
//! }
//!
//! let x = homura::tensor([0.0, 1.0]);
//! x.requires_grad_(true);
//! let y = Exp.apply(&[&x]).remove(0);
//! let options = GradOptions { create_graph: true, ..Default::default() };
//! let dx = autograd::grad(&[&y.sum()], &[&x], options).remove(0).unwrap();
//! let d2x = autograd::grad(&[&dx.sum()], &[&x], GradOptions::default()).remove(0).unwrap();
//! assert_eq!(d2x.to_vec::<f32>(), x.detach().exp().to_vec::<f32>());
//! ```
//!
//! Saving the exponential as an intermediate instead (returning `&e * 1.0`, a separate tensor)
//! gives the same first derivative, but it is a constant, so `dx` above wouldn't require grad.
//! Recomputing it in `backward` from the saved input (`saved_tensors()[0].exp()`) is correct
//! again for every order.
//!
//! The function value is moved into the node, and lives as long as it. Its fields carry the
//! operation's non-tensor arguments and whatever `forward` computes for `backward` besides
//! tensors. Tensors in its fields are the caller's responsibility: they are not inputs, so they
//! get no edges, and not saved tensors, so they get none of the checks above; recorded
//! operations on them in `backward` build graphs through their own history.
//!
//! `forward` must not modify its inputs in place, return a tensor that shares memory with an input
//! or another output, or mark a tensor it doesn't return: [`Function::apply`] panics after it
//! returns, in every grad mode, and the modification isn't undone. Return a `deep_clone`
//! instead. A gradient that `backward` returns is checked like those of built-in operations: a
//! larger shape that the input's shape broadcasts to is summed to it, another dtype is
//! converted, and only a zero-dimensional gradient may be on another device, which it is moved
//! to. `backward` may run nested backward passes on graphs of its own.
//!
//! Calls on different threads are independent. Concurrent backward passes through the same node
//! are safe, but each pass that doesn't `retain_graph` frees the saved tensors when it is done
//! with the node, so the others then fail: give all of them `retain_graph`. Mutable state in the
//! function value is up to its implementation. Differentiating, inside `backward`, a graph that
//! contains the running node is not supported.
//!
//! # Differences from PyTorch
//!
//! homura doesn't support yet some cases that PyTorch handles. They panic with an error that
//! says so, instead of computing something else:
//!
//! - Recorded in-place operations on views, which PyTorch records through the view's base:
//!   write into the base, or into a `deep_clone` of the view.
//! - Stale views, which PyTorch replays on their base's new history: create the view again.
//! - Differentiating the gradient of `prod` for an input that contains zeros.
//! - Computing the gradient of [`Tensor::as_strided`] when its input's elements overlap in
//!   memory: use a `deep_clone` of the input first.
//!
//! The CPU reference backend accumulates floating point reductions and matrix inner products
//! in f64. This differs from some PyTorch kernels: extreme cancellation can change finite,
//! infinity and NaN results, even for finite inputs. Scaled matrix products also use rules
//! independent of size and layout; see [`Tensor::addmm`]. The sums at repeated positions of
//! advanced indexing's backward pass and of `index_put` with `accumulate` are computed the same
//! way, so they can differ from PyTorch's, which round after each addition, by more than the
//! last bit.
//!
//! With `create_graph`, the gradient of advanced indexing ([`Tensor::index`] with index tensors)
//! is recorded as `IndexPutBackward0`, where PyTorch records `IndexPutImplBackward0`.
//! `IndexPutBackward0` sums the gradient of the values of [`Tensor::index_put`] to their shape
//! itself; PyTorch's engine does, recording `SumBackward1` and `ViewBackward0`.
//!
//! [Custom functions](#custom-functions) differ from PyTorch's `torch.autograd.Function` on
//! purpose:
//!
//! - Outputs that share memory with an input or another output, in-place modifications of the
//!   inputs in `forward` and marking a tensor that isn't an output as non-differentiable panic.
//!   PyTorch accepts all three: the first through views or new tensors that share the data, the
//!   second silently (without `mark_dirty`, the gradients may be wrong), and ignores the third.
//! - Outputs that received no gradient give `backward` `None`, where PyTorch gives zeros by
//!   default. When no output received one, `backward` isn't called, and [`grad`] with
//!   `allow_unused` returns `None` for the inputs; PyTorch calls `backward` anyway.
//! - The outputs are always new tensors without any history `forward` gave them, recorded or
//!   not; PyTorch returns the tensors `forward` returned, replacing their history when it
//!   records the call. So an output is never a view: one that `forward` computed as a view of an
//!   intermediate value can be modified in place, recorded after the node, where PyTorch
//!   rejects that.
//! - Nodes are named after the last segment of the type's path, without generic arguments. The
//!   spelling of type names isn't guaranteed to be stable between compiler versions; the names
//!   are for display only.
//! - `mark_dirty`, `set_materialize_grads`, `once_differentiable`, `setup_context`, `jvp` and
//!   `vmap` are not available.
//!
//! Hooks, anomaly detection, checkpointing and forward-mode differentiation are not available
//! yet.
//!
//! # Lifetimes and threads
//!
//! The graph lives as long as the tensors computed in it: each holds its node, and nodes hold
//! the nodes before them. Without `retain_graph`, a backward pass frees the tensors the nodes
//! saved, so a second pass through a node that needs them panics. Destroying a graph doesn't
//! recurse, so graphs of any depth can be dropped; a chain of leaves whose `.grad` tensors hold
//! graphs that reach the next leaf is the exception and is destroyed recursively.
//! `backward` with `create_graph` stores gradients that have a graph which usually reaches the
//! leaf again: the leaf and its gradient then keep each other alive until the gradient is
//! cleared with `set_grad(None)`, as in PyTorch. [`grad`] doesn't accumulate into leaves, so it
//! has no such cycle.
//!
//! [`Tensor::retain_grad`] registers the tensor with its node only weakly, so it doesn't keep
//! the tensor alive, and the gradient stored without `create_graph` has no graph. With
//! `create_graph` (in [`grad`] passes too), the copy and additions are recorded when their inputs
//! require grad; a constant gradient can still have no graph. When the stored gradient has a
//! graph, it usually reaches the tensor only through data copies the graph saved, so dropping
//! the tensor frees it. It can still form a cycle when the gradient's graph holds the tensor
//! itself, for example as the base of a view that is a leaf of its own and was used in the
//! gradient's computation: clear such a gradient with `set_grad(None)` when done with it.
//!
//! A tensor that doesn't require grad and joins the graph through an in-place operation can form
//! another cycle, with or without a backward pass, when a node of its new history saved the
//! tensor itself as an input: `b.addcmul_(&b, &z, 1.0)`, or `let y = &b * &z; b.add_(&y)`, whose
//! earlier product saved `b`. Clearing `.grad` or using [`grad`] doesn't help here: the tensor and
//! its history then live until [`Tensor::detach_`] removes the history, as in PyTorch.
//!
//! Backward passes run on the calling thread. Passes on different threads may share leaves:
//! accumulating into the same `.grad` is serialized and loses nothing.
//!
//! # Checking gradients
//!
//! [`gradcheck`] compares the gradients that backward passes compute with central differences
//! of the function, and [`gradgradcheck`] does the same for the gradients of the gradients, like
//! their PyTorch namesakes. Both work on `F64` inputs at points where the function is smooth.
//!
//! # Grad mode
//!
//! [`no_grad`], [`enable_grad`] and [`set_grad_enabled`] turn graph recording off or on for the
//! current thread until the returned guard is dropped. They are also re-exported at the crate
//! root.
//!
//! ```
//! use homura::autograd;
//!
//! {
//!     let _guard = autograd::no_grad();
//!     assert!(!autograd::is_grad_enabled());
//! }
//! assert!(autograd::is_grad_enabled());
//! ```

use std::cell::Cell;
use std::marker::PhantomData;

use crate::Tensor;

mod accumulate;
mod engine;
#[cfg(test)]
mod engine_tests;
pub(crate) mod formulas;
mod function;
#[cfg(test)]
mod function_tests;
mod gradcheck;
#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod inplace_tests;
#[cfg(test)]
mod llama_tests;
mod meta;
mod node;
mod record;
#[cfg(test)]
mod retain_tests;
mod saved;

pub use function::{BackwardContext, ForwardContext, Function};
pub use gradcheck::{GradcheckError, GradcheckOptions, gradcheck, gradgradcheck};
pub(crate) use meta::{
    AutogradMeta, ViewInfo, forget_accumulator, replace_leaf_data, repr_suffix, view_of,
};
pub use node::GradFn;
pub(crate) use record::{check_in_place, original_value, record_in_place};

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether operations on this thread record the backward graph.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Enables or disables graph recording on this thread until the returned guard is dropped.
pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let previous = GRAD_ENABLED.with(|g| g.replace(enabled));
    GradModeGuard {
        previous,
        _not_send: PhantomData,
    }
}

/// Disables graph recording on this thread until the returned guard is dropped, like
/// `with torch.no_grad():`.
///
/// ```
/// let _guard = homura::no_grad();
/// assert!(!homura::is_grad_enabled());
/// ```
pub fn no_grad() -> GradModeGuard {
    set_grad_enabled(false)
}

/// Enables graph recording on this thread until the returned guard is dropped, like
/// `with torch.enable_grad():`.
pub fn enable_grad() -> GradModeGuard {
    set_grad_enabled(true)
}

/// Guard returned by [`no_grad`], [`enable_grad`] and [`set_grad_enabled`]; restores the
/// previous grad mode when dropped.
#[must_use = "grad mode is restored when the guard is dropped"]
#[derive(Debug)]
pub struct GradModeGuard {
    previous: bool,
    _not_send: PhantomData<*const ()>,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.previous));
    }
}

/// Options of [`backward`], with PyTorch's defaults.
#[derive(Clone, Debug, Default)]
pub struct BackwardOptions {
    /// Keep the tensors the graph saved, so that it can be run again. `None`, the default,
    /// means the same as `create_graph`.
    pub retain_graph: Option<bool>,
    /// Record the backward pass, so that the gradients can be differentiated again.
    pub create_graph: bool,
}

/// Options of [`grad`], with PyTorch's defaults.
#[derive(Clone, Debug, Default)]
pub struct GradOptions {
    /// The gradient of each output, the vector of the vector-Jacobian product: empty, or one
    /// entry per output. Missing entries are created as ones for outputs with a single element.
    pub grad_outputs: Vec<Option<Tensor>>,
    /// Keep the tensors the graph saved, so that it can be run again. `None`, the default,
    /// means the same as `create_graph`.
    pub retain_graph: Option<bool>,
    /// Record the backward pass, so that the gradients can be differentiated again.
    pub create_graph: bool,
    /// Return `None` for inputs the outputs don't depend on, instead of panicking.
    pub allow_unused: bool,
}

/// Computes the gradients of `tensors` with respect to the leaves of their graph and adds them
/// to the leaves' `.grad` (`torch.autograd.backward`).
///
/// `grad_tensors` is empty or has one entry per tensor: the gradient of that tensor, which must
/// have its shape and is converted to its dtype; it must be on its device unless it is
/// zero-dimensional, then it is moved there. `None` entries are created as ones, which needs a
/// tensor with a single element. See [`Tensor::backward`] for the common case.
///
/// # Panics
///
/// Panics with PyTorch's messages if a tensor doesn't require grad, if a gradient can't be
/// created, has the wrong shape or is on the wrong device, if `grad_tensors` has the wrong
/// length, or if the backward
/// pass fails (a saved tensor was freed or modified in place, a derivative isn't implemented).
pub fn backward(tensors: &[&Tensor], grad_tensors: &[Option<&Tensor>], options: BackwardOptions) {
    let roots = root_edges(tensors, make_grads(tensors, grad_tensors));
    let keep_graph = options.retain_graph.unwrap_or(options.create_graph);
    engine::run(roots, None, keep_graph, options.create_graph);
}

/// Computes and returns the gradients of `outputs` with respect to `inputs`
/// (`torch.autograd.grad`), without accumulating them into the `.grad` of the leaves. Only the
/// parts of the graph that lead to `inputs` run. Tensors on those paths (or among `inputs`) that
/// called [`Tensor::retain_grad`] store their gradients, as in PyTorch.
///
/// The result has one entry per input; it is `None` only for inputs that the outputs don't
/// depend on, which panics unless `allow_unused` is set.
///
/// ```
/// use homura::autograd::{self, GradOptions};
///
/// let x = homura::tensor([3.0]);
/// x.requires_grad_(true);
/// let y = (&x * &x * &x).sum();
/// let options = GradOptions { create_graph: true, ..Default::default() };
/// let dy = autograd::grad(&[&y], &[&x], options).remove(0).unwrap();
/// let d2y = autograd::grad(&[&dy.sum()], &[&x], GradOptions::default()).remove(0).unwrap();
/// assert_eq!(dy.item::<f32>(), 27.0);
/// assert_eq!(d2y.item::<f32>(), 18.0);
/// ```
///
/// # Panics
///
/// Like [`backward`]; also if `inputs` is empty, if an input doesn't require grad, or if an
/// input is unused and `allow_unused` is not set.
pub fn grad(outputs: &[&Tensor], inputs: &[&Tensor], options: GradOptions) -> Vec<Option<Tensor>> {
    if inputs.is_empty() {
        panic!("`inputs` argument to `grad()` cannot be empty.");
    }
    let grad_outputs: Vec<Option<&Tensor>> =
        options.grad_outputs.iter().map(Option::as_ref).collect();
    let roots = root_edges(outputs, make_grads(outputs, &grad_outputs));
    let input_edges: Vec<_> = inputs
        .iter()
        .map(|t| {
            if !t.requires_grad() {
                panic!("One of the differentiated Tensors does not require grad");
            }
            record::gradient_edge(t)
        })
        .collect();
    let keep_graph = options.retain_graph.unwrap_or(options.create_graph);
    let grads = engine::run(roots, Some(&input_edges), keep_graph, options.create_graph);
    if !options.allow_unused
        && let Some(i) = grads.iter().position(Option::is_none)
    {
        panic!(
            "The differentiated Tensor at index {i} appears to not have been used in the graph. \
             Set allow_unused if this is the desired behavior."
        );
    }
    grads
}

/// The gradient of each output (PyTorch's `_make_grads`): the given one, checked against the
/// output's shape, or ones for an output that requires grad and has a single element. As in
/// PyTorch, the pairs are checked before the numbers of outputs and gradients are compared.
fn make_grads(outputs: &[&Tensor], grads: &[Option<&Tensor>]) -> Vec<Option<Tensor>> {
    let implicit = vec![None; outputs.len()];
    let given = if grads.is_empty() { &implicit } else { grads };
    let made: Vec<_> = outputs
        .iter()
        .zip(given)
        .enumerate()
        .map(|(i, (out, grad))| match *grad {
            Some(grad) => {
                let (grad_shape, out_shape) = (grad.shape(), out.shape());
                if grad_shape != out_shape {
                    panic!(
                        "Mismatch in shape: grad_output[{i}] has a shape of {grad_shape:?} and \
                         output[{i}] has a shape of {out_shape:?}."
                    );
                }
                Some(grad.clone())
            }
            None if out.requires_grad() => {
                if out.num_elements() != 1 {
                    panic!("grad can be implicitly created only for scalar outputs");
                }
                Some(crate::ones_like(out))
            }
            None => None,
        })
        .collect();
    if given.len() != outputs.len() {
        panic!(
            "got {} tensors and {} gradients",
            outputs.len(),
            given.len()
        );
    }
    made
}

/// The edge of each output with its gradient.
fn root_edges(outputs: &[&Tensor], grads: Vec<Option<Tensor>>) -> Vec<(node::Edge, Tensor)> {
    outputs
        .iter()
        .zip(grads)
        .enumerate()
        .map(|(i, (out, grad))| {
            let Some(edge) = record::gradient_edge(out) else {
                panic!("element {i} of tensors does not require grad and does not have a grad_fn");
            };
            (edge, grad.expect("outputs with an edge require grad"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_nest() {
        assert!(is_grad_enabled());
        {
            let _a = no_grad();
            assert!(!is_grad_enabled());
            {
                let _b = enable_grad();
                assert!(is_grad_enabled());
            }
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    }
}
