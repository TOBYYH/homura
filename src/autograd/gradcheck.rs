//! Numerical checks of backward formulas, like `torch.autograd.gradcheck` and
//! `torch.autograd.gradgradcheck`: the Jacobians that backward passes compute are compared with
//! central differences of the function.

use std::fmt;
use std::sync::Arc;

use super::{GradOptions, grad, no_grad};
use crate::{DType, Device, Tensor};

/// Options of [`gradcheck`] and [`gradgradcheck`], with PyTorch's defaults, which suit `F64`
/// inputs.
#[derive(Clone, Debug)]
pub struct GradcheckOptions {
    /// The step of the central differences.
    pub eps: f64,
    /// The absolute tolerance.
    pub atol: f64,
    /// The relative tolerance, relative to the numerical derivative.
    pub rtol: f64,
}

impl Default for GradcheckOptions {
    fn default() -> Self {
        GradcheckOptions {
            eps: 1e-6,
            atol: 1e-5,
            rtol: 1e-3,
        }
    }
}

/// The failure of [`gradcheck`] or [`gradgradcheck`]: the first entry of a Jacobian whose
/// analytical and numerical values differ, or an output that changed under perturbation.
pub struct GradcheckError {
    message: String,
}

impl fmt::Display for GradcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Debug for GradcheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GradcheckError {}

fn error(message: String) -> Result<(), GradcheckError> {
    Err(GradcheckError { message })
}

/// Checks the backward formulas `f` uses against central differences
/// (`torch.autograd.gradcheck`).
///
/// `f` gets handles to `inputs` and returns its outputs. The inputs that require grad are the
/// variables: each must be an `F64` tensor on the CPU whose elements don't overlap in memory and
/// that shares no storage with another input. The other inputs, such as index tensors, are
/// constants. For every floating point output and every variable, the Jacobian computed by
/// backward passes must match the numerical one: `|analytical - numerical| <= atol + rtol *
/// |numerical|` for each entry. Outputs that don't require grad have no graph, so their
/// analytical Jacobians are zero, and they are compared too (PyTorch skips them unless no
/// output requires grad).
///
/// Each numerical derivative perturbs one element in place, in no-grad mode, and runs `f` again
/// from scratch. The element is restored afterwards, also when `f` panics. Check only points
/// where `f` is smooth: ties, kinks, non-finite values and the like need exact reference values
/// instead. Tensors without elements have empty Jacobians, so check their gradients' shapes and
/// values separately.
///
/// ```
/// use homura::autograd::{self, GradcheckOptions};
///
/// let x = homura::Tensor::from_slice(&[0.5f64, -1.0, 2.0], [3]);
/// x.requires_grad_(true);
/// let check = autograd::gradcheck(|x| vec![x[0].exp() * &x[0]], &[&x], GradcheckOptions::default());
/// assert!(check.is_ok());
/// // A formula that drops part of the gradient is caught.
/// let wrong = autograd::gradcheck(|x| vec![&x[0] * &x[0].detach()], &[&x], Default::default());
/// assert!(wrong.is_err());
/// ```
///
/// # Panics
///
/// Panics if no input requires grad, if a variable breaks the rules above, or if `f` panics. An
/// output on the meta device can't be read and panics too.
pub fn gradcheck(
    f: impl Fn(&[Tensor]) -> Vec<Tensor>,
    inputs: &[&Tensor],
    options: GradcheckOptions,
) -> Result<(), GradcheckError> {
    let inputs: Vec<Tensor> = inputs.iter().map(|&t| t.clone()).collect();
    let variables = check_inputs(&inputs);
    let base = f(&inputs);
    let signature: Vec<_> = base.iter().map(|o| (o.shape(), o.dtype())).collect();
    let checked: Vec<usize> = (0..base.len())
        .filter(|&o| base[o].is_floating_point())
        .collect();
    drop(base);

    // numerical[v][j][c]: derivatives of checked output `c` with respect to element `j` of
    // variable `v`.
    let mut numerical = Vec::with_capacity(variables.len());
    for &v in &variables {
        let x = &inputs[v];
        let mut columns = Vec::with_capacity(x.num_elements());
        for j in 0..x.num_elements() {
            let element = element_view(x, j);
            let restore = Restore::new(&element);
            let run = |value: f64| {
                restore.set(value);
                evaluate(&f, &inputs, &signature, &checked)
            };
            let below = run(restore.value - options.eps)?;
            let above = run(restore.value + options.eps)?;
            drop(restore);
            let column: Vec<Vec<f64>> = below
                .iter()
                .zip(&above)
                .map(|(b, a)| {
                    b.iter()
                        .zip(a)
                        .map(|(b, a)| (a - b) / (2.0 * options.eps))
                        .collect()
                })
                .collect();
            columns.push(column);
        }
        numerical.push(columns);
    }

    let outputs = f(&inputs);
    let variable_refs: Vec<&Tensor> = variables.iter().map(|&v| &inputs[v]).collect();
    for (c, &o) in checked.iter().enumerate() {
        let out = &outputs[o];
        let has_graph = out.requires_grad();
        let rows = (0..out.num_elements()).map(|k| {
            if has_graph {
                analytical_row(out, k, &variable_refs)
            } else {
                variable_refs
                    .iter()
                    .map(|x| vec![0.0; x.num_elements()])
                    .collect()
            }
        });
        for (k, row) in rows.enumerate() {
            for (v, analytical) in row.iter().enumerate() {
                for (j, &a) in analytical.iter().enumerate() {
                    let n = numerical[v][j][c][k];
                    if close(a, n, &options) {
                        continue;
                    }
                    let mut message = format!(
                        "Jacobian mismatch for output {o} with respect to input {}: the \
                         derivative of output element {:?} with respect to input element {:?} \
                         is {a} analytically but {n} numerically (atol {}, rtol {})",
                        variables[v],
                        position(&out.shape(), k),
                        position(&variable_refs[v].shape(), j),
                        options.atol,
                        options.rtol,
                    );
                    if !has_graph {
                        message.push_str(
                            "; the output doesn't require grad, so its analytical derivatives \
                             are zero",
                        );
                    }
                    return error(message);
                }
            }
        }
    }
    Ok(())
}

/// Checks the gradients of the backward formulas `f` uses (`torch.autograd.gradgradcheck`).
///
/// For each output of `f` that requires grad, an upstream gradient `g` is made of deterministic
/// values in `[-1, 1)`, in a layout that isn't contiguous (for outputs with more than one
/// element), and requires grad itself. Then [`gradcheck`] checks the function that computes the
/// gradients of `f`'s outputs with respect to its variables with `create_graph`, with the `g`s
/// as further variables. Differentiating with respect to `g` too catches a formula that cuts
/// `g` off its graph. A gradient without a graph has zero analytical derivatives and is still
/// compared with the numerical ones (see [`gradcheck`]). The inputs follow the rules of
/// [`gradcheck`].
///
/// # Panics
///
/// Like [`gradcheck`]; also if the outputs that require grad differ between calls of `f`.
pub fn gradgradcheck(
    f: impl Fn(&[Tensor]) -> Vec<Tensor>,
    inputs: &[&Tensor],
    options: GradcheckOptions,
) -> Result<(), GradcheckError> {
    let xs: Vec<Tensor> = inputs.iter().map(|&t| t.clone()).collect();
    let variables = check_inputs(&xs);
    let outputs = f(&xs);
    let differentiable: Vec<usize> = (0..outputs.len())
        .filter(|&o| outputs[o].requires_grad())
        .collect();
    if differentiable.is_empty() {
        return Ok(());
    }
    let upstream: Vec<Tensor> = differentiable
        .iter()
        .map(|&o| upstream_gradient(&outputs[o], o as u64))
        .collect();
    drop(outputs);
    let n = xs.len();
    let gradients = |all: &[Tensor]| -> Vec<Tensor> {
        let (xs, gs) = all.split_at(n);
        let outputs = f(xs);
        if (0..outputs.len()).any(|o| outputs[o].requires_grad() != differentiable.contains(&o)) {
            panic!("gradgradcheck: the outputs of the function that require grad changed");
        }
        let outputs: Vec<&Tensor> = differentiable.iter().map(|&o| &outputs[o]).collect();
        let variables: Vec<&Tensor> = variables.iter().map(|&v| &xs[v]).collect();
        let options = GradOptions {
            grad_outputs: gs.iter().cloned().map(Some).collect(),
            create_graph: true,
            allow_unused: true,
            ..Default::default()
        };
        grad(&outputs, &variables, options)
            .into_iter()
            .flatten()
            .collect()
    };
    let all: Vec<&Tensor> = inputs.iter().copied().chain(&upstream).collect();
    gradcheck(gradients, &all, options)
}

/// The indices of the inputs that require grad, after checking them.
fn check_inputs(inputs: &[Tensor]) -> Vec<usize> {
    let variables: Vec<usize> = (0..inputs.len())
        .filter(|&i| inputs[i].requires_grad())
        .collect();
    if variables.is_empty() {
        panic!(
            "gradcheck expects at least one input tensor to require gradient, but none of them \
             have requires_grad=True."
        );
    }
    for &v in &variables {
        let x = &inputs[v];
        if x.dtype() != DType::F64 || x.device() != Device::Cpu {
            panic!(
                "gradcheck: input {v} requires grad and is a {} tensor on {:?}; numerical \
                 derivatives need F64 tensors on the CPU",
                x.dtype(),
                x.device()
            );
        }
        let data = x.data();
        if data.layout.may_overlap_internally() {
            panic!(
                "gradcheck: the elements of input {v} may overlap in memory, so perturbing one \
                 would change others; pass a deep_clone() of it"
            );
        }
        if let Some(other) = (0..inputs.len())
            .find(|&i| i != v && Arc::ptr_eq(&inputs[i].data().storage, &data.storage))
        {
            panic!(
                "gradcheck: inputs {v} and {other} share storage, so perturbing one would change \
                 the other; pass a deep_clone() of one of them"
            );
        }
    }
    variables
}

/// The checked outputs of `f` as `f64` values, after checking that the outputs have kept their
/// shapes and dtypes.
fn evaluate(
    f: &impl Fn(&[Tensor]) -> Vec<Tensor>,
    inputs: &[Tensor],
    signature: &[(crate::Shape, DType)],
    checked: &[usize],
) -> Result<Vec<Vec<f64>>, GradcheckError> {
    let outputs = f(inputs);
    let found: Vec<_> = outputs.iter().map(|o| (o.shape(), o.dtype())).collect();
    if found != signature {
        let describe = |outputs: &[(crate::Shape, DType)]| {
            let outputs: Vec<String> = outputs
                .iter()
                .map(|(shape, dtype)| format!("{dtype} {shape:?}"))
                .collect();
            outputs.join(", ")
        };
        return Err(GradcheckError {
            message: format!(
                "the outputs changed when an input was perturbed: [{}] became [{}]",
                describe(signature),
                describe(&found)
            ),
        });
    }
    Ok(checked
        .iter()
        .map(|&o| outputs[o].to(DType::F64).to_vec())
        .collect())
}

/// The gradients of `variables` for an upstream gradient that is 1 at element `k` of `out` and
/// 0 elsewhere, as `f64` values; zeros for variables `out` doesn't depend on.
fn analytical_row(out: &Tensor, k: usize, variables: &[&Tensor]) -> Vec<Vec<f64>> {
    let one_hot = crate::zeros_with(out.shape(), (out.dtype(), out.device()));
    {
        let _no_grad = no_grad();
        element_view(&one_hot, k).fill_(1.0);
    }
    let options = GradOptions {
        grad_outputs: vec![Some(one_hot)],
        retain_graph: Some(true),
        allow_unused: true,
        ..Default::default()
    };
    grad(&[out], variables, options)
        .into_iter()
        .zip(variables)
        .map(|(g, x)| match g {
            Some(g) => g.to(DType::F64).to_vec(),
            None => vec![0.0; x.num_elements()],
        })
        .collect()
}

/// Whether the analytical derivative `a` is close enough to the numerical one `n`, as in
/// `torch.allclose(a, n)`: equal infinities are close, NaN is never close.
fn close(a: f64, n: f64, options: &GradcheckOptions) -> bool {
    a == n || (a - n).abs() <= options.atol + options.rtol * n.abs()
}

/// The multi-index of element `j` of `shape` in row-major order.
fn position(shape: &[usize], mut j: usize) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        index[d] = j % shape[d];
        j /= shape[d];
    }
    index
}

/// A zero-dimensional view of element `j` of `t`, in row-major order. Made in no-grad mode, so
/// it has no node of its own.
fn element_view(t: &Tensor, j: usize) -> Tensor {
    let _no_grad = no_grad();
    position(&t.shape(), j)
        .into_iter()
        .fold(t.clone(), |e, i| e.select(0, i as i64))
}

/// Restores an element to the value it had when created, on drop, even while unwinding.
struct Restore<'a> {
    element: &'a Tensor,
    value: f64,
}

impl<'a> Restore<'a> {
    fn new(element: &'a Tensor) -> Self {
        let value = element.item::<f64>();
        Restore { element, value }
    }

    fn set(&self, value: f64) {
        let _no_grad = no_grad();
        self.element.fill_(value);
    }
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        self.set(self.value);
    }
}

/// A gradient for `out` that requires grad, with deterministic values in `[-1, 1)` from the
/// SplitMix64 sequence of `seed`, and strides twice the contiguous ones when `out` has more
/// than one element. Floating point outputs keep their dtype; others get `F64`.
fn upstream_gradient(out: &Tensor, seed: u64) -> Tensor {
    let dtype = if out.is_floating_point() {
        out.dtype()
    } else {
        DType::F64
    };
    let mut state = seed.wrapping_add(0x5EED);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE5_E9B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let shape = out.shape();
    let g = if out.num_elements() > 1 {
        let mut wide = shape.to_vec();
        wide.push(2);
        let values: Vec<f64> = (0..2 * out.num_elements()).map(|_| next()).collect();
        let _no_grad = no_grad();
        Tensor::from_vec(values, wide).to(dtype).select(-1, 0)
    } else {
        let values: Vec<f64> = (0..out.num_elements()).map(|_| next()).collect();
        Tensor::from_vec(values, shape).to(dtype)
    };
    g.requires_grad_(true);
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::node::{ApplyContext, Node};
    use crate::autograd::record::record;

    /// `x * forward`, whose backward pass multiplies by `backward` instead, and detaches the
    /// upstream gradient first when `detach` is set.
    fn scale(x: &Tensor, forward: f64, backward: f64, detach: bool) -> Tensor {
        struct ScaleBackward {
            factor: f64,
            detach: bool,
        }
        impl Node for ScaleBackward {
            fn name(&self) -> &'static str {
                "ScaleBackward"
            }
            fn apply(
                &self,
                _ctx: &ApplyContext<'_>,
                grads: Vec<Option<Tensor>>,
            ) -> Vec<Option<Tensor>> {
                let g = grads.into_iter().next().flatten().unwrap();
                let g = if self.detach { g.detach() } else { g };
                vec![Some(g.mul(self.factor))]
            }
        }
        let out = {
            let _no_grad = no_grad();
            x.mul(forward)
        };
        record(&[Some(x)], &[&out], |_| ScaleBackward {
            factor: backward,
            detach,
        });
        out
    }

    fn variable() -> Tensor {
        let x = Tensor::from_slice(&[0.5f64, -1.5, 2.0], [3]);
        x.requires_grad_(true);
        x
    }

    #[test]
    fn a_correct_formula_passes_both_checks() {
        let x = variable();
        let f = |x: &[Tensor]| vec![scale(&x[0], 2.0, 2.0, false)];
        gradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
        gradgradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
    }

    /// The analytical and numerical values a mismatch message reports.
    fn reported_values(message: &str) -> (f64, f64) {
        let (_, rest) = message.split_once(" is ").unwrap();
        let (a, rest) = rest.split_once(" analytically but ").unwrap();
        let (n, _) = rest.split_once(" numerically").unwrap();
        (a.parse().unwrap(), n.parse().unwrap())
    }

    #[test]
    fn a_wrong_first_derivative_is_reported() {
        let x = variable();
        let f = |x: &[Tensor]| vec![scale(&x[0], 2.0, 3.0, false)];
        let message = gradcheck(f, &[&x], GradcheckOptions::default())
            .unwrap_err()
            .to_string();
        assert!(
            message.starts_with(
                "Jacobian mismatch for output 0 with respect to input 0: the derivative of \
                 output element [0] with respect to input element [0] is "
            ),
            "{message}"
        );
        assert!(message.ends_with("(atol 0.00001, rtol 0.001)"), "{message}");
        let (a, n) = reported_values(&message);
        assert_eq!(a, 3.0);
        assert!((n - 2.0).abs() < 1e-6, "{message}");
        // A NaN derivative is never close to anything.
        let f = |x: &[Tensor]| vec![scale(&x[0], 2.0, f64::NAN, false)];
        let message = gradcheck(f, &[&x], GradcheckOptions::default())
            .unwrap_err()
            .to_string();
        assert!(reported_values(&message).0.is_nan(), "{message}");
    }

    #[test]
    fn a_formula_that_cuts_the_upstream_gradient_off_fails_gradgradcheck() {
        // Its first derivative is right, and its gradient has no graph at all, so only the
        // derivative with respect to the upstream gradient (2) reveals it.
        let x = variable();
        let f = |x: &[Tensor]| vec![scale(&x[0], 2.0, 2.0, true)];
        gradcheck(f, &[&x], GradcheckOptions::default()).unwrap();
        let message = gradgradcheck(f, &[&x], GradcheckOptions::default())
            .unwrap_err()
            .to_string();
        assert!(
            message.starts_with(
                "Jacobian mismatch for output 0 with respect to input 1: the derivative of \
                 output element [0] with respect to input element [0] is "
            ),
            "{message}"
        );
        assert!(
            message.ends_with(
                "; the output doesn't require grad, so its analytical derivatives are zero"
            ),
            "{message}"
        );
        let (a, n) = reported_values(&message);
        assert_eq!(a, 0.0);
        assert!((n - 2.0).abs() < 1e-6, "{message}");
    }

    #[test]
    fn upstream_gradients_are_strided_leaves_in_range() {
        let out = crate::zeros_with([2, 3], DType::F64);
        let g = upstream_gradient(&out, 0);
        assert!(g.is_leaf() && g.requires_grad());
        assert_eq!(g.shape(), [2, 3]);
        assert_eq!(g.data().layout.strides.to_vec(), [6, 2]);
        let values = g.to_vec::<f64>();
        assert!(values.iter().all(|v| (-1.0..1.0).contains(v)));
        assert_ne!(values, upstream_gradient(&out, 1).to_vec::<f64>());
        // A single element can't be strided.
        assert_eq!(upstream_gradient(&crate::zeros([]), 0).dtype(), DType::F32);
    }
}
