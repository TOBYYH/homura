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
//! // `grad` returns the gradients instead of accumulating them into `.grad`.
//! let y = (&x * &x).sum();
//! let g = autograd::grad(&[&y], &[&x], autograd::GradOptions::default());
//! assert_eq!(g[0].as_ref().unwrap().to_vec::<f32>(), [2.0, 4.0, 6.0]);
//! ```
//!
//! With `create_graph`, the backward pass is recorded too, so gradients can be differentiated
//! again (higher-order derivatives).
//!
//! # Current state
//!
//! Every differentiable operation records its node, but only these have backward formulas so
//! far: `add`, `sub`, `mul`, `neg`, `exp`, `erf`, `sign`, `sum`, `sum_to_size`, `expand`, `view`,
//! `reshape`, `select`, `narrow`, slicing, `as_strided`, `index_select`, indexing with an integer
//! tensor, `index_add`, `gather`, `scatter_add`, `select_scatter`, `slice_scatter`, `deep_clone`
//! and `to` (and what is built from them, such as `flatten`, `contiguous`, `repeat`, `split` and
//! `chunk`). The others record a `NotImplemented` node, and a backward pass that reaches one
//! panics with "the derivative for 'op' is not implemented." instead of silently dropping
//! gradients; [`grad`] doesn't run the parts of the graph it doesn't need.
//!
//! In-place operations are allowed in no-grad mode, which is how optimizers update parameters.
//! In grad mode, an in-place operation whose target or inputs require grad panics before
//! writing anything: on a leaf that requires grad with PyTorch's error, and otherwise because
//! recording in-place operations is not implemented yet.
//!
//! # Views
//!
//! A view of a tensor that requires grad records its own node, which sends gradients to the
//! tensor it was created from. A view created in no-grad mode, or before its base required
//! grad, has no node: it requires grad as long as its base does, but gradients reaching it are
//! dropped, and it can't be the root of a backward pass. Calling `requires_grad_(true)` on such
//! a view makes it a leaf of its own.
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
//! cleared with `set_grad(None)`, as in PyTorch. [`grad`] never writes `.grad`.
//!
//! Backward passes run on the calling thread. Passes on different threads may share leaves:
//! accumulating into the same `.grad` is serialized and loses nothing.
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
#[cfg(test)]
mod graph_tests;
mod meta;
mod node;
mod record;
mod saved;

pub(crate) use meta::{AutogradMeta, ViewInfo, view_of};
pub use node::GradFn;
pub(crate) use record::{check_inplace, needs_grad, record_unimplemented};

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
/// have its shape and is converted to its dtype and device. `None` entries are created as ones,
/// which needs a tensor with a single element. See [`Tensor::backward`] for the common case.
///
/// # Panics
///
/// Panics with PyTorch's messages if a tensor doesn't require grad, if a gradient can't be
/// created or has the wrong shape, if `grad_tensors` has the wrong length, or if the backward
/// pass fails (a saved tensor was freed or modified in place, a derivative isn't implemented).
pub fn backward(tensors: &[&Tensor], grad_tensors: &[Option<&Tensor>], options: BackwardOptions) {
    let roots = root_edges(tensors, make_grads(tensors, grad_tensors));
    let keep_graph = options.retain_graph.unwrap_or(options.create_graph);
    engine::run(roots, None, keep_graph, options.create_graph);
}

/// Computes and returns the gradients of `outputs` with respect to `inputs`
/// (`torch.autograd.grad`), without writing any `.grad`. Only the parts of the graph that lead
/// to `inputs` run.
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
