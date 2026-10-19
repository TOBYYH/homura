//! Custom differentiable functions (M2 step 7): `autograd::Function` with its forward and
//! backward contexts. `tools/golden/check_m2_function.py` checks the same cases against the
//! reference PyTorch build, in the same order; where homura differs on purpose, it asserts the
//! reference behavior and names the difference. Run in debug and release profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use homura::autograd::{
    self, BackwardContext, BackwardOptions, ForwardContext, Function, GradOptions, GradcheckOptions,
};
use homura::{DType, Device, Tensor};

fn f64s(values: &[f64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

/// An `F64` leaf that requires grad.
fn leaf(values: &[f64]) -> Tensor {
    let x = f64s(values);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-12 * e.abs().max(1.0),
            "{actual:?} vs {expected:?}"
        );
    }
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

fn assert_panics(f: impl FnOnce(), fragment: &str) {
    let message = panic_message(f);
    assert!(message.contains(fragment), "{message:?} lacks {fragment:?}");
}

/// The gradient of `y` with respect to each of `xs`.
fn grads(y: &Tensor, xs: &[&Tensor], create_graph: bool) -> Vec<Option<Tensor>> {
    let options = GradOptions {
        create_graph,
        allow_unused: true,
        ..Default::default()
    };
    autograd::grad(&[y], xs, options)
}

fn grad1(y: &Tensor, x: &Tensor, create_graph: bool) -> Tensor {
    grads(y, &[x], create_graph).remove(0).expect("a gradient")
}

fn first(function: impl Function, inputs: &[&Tensor]) -> Tensor {
    function.apply(inputs).remove(0)
}

fn no_grad_options() -> GradcheckOptions {
    GradcheckOptions::default()
}

// --- Functions shared by several tests ---

/// `x * x`, saving the input.
struct Square;

impl Function for Square {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0]]);
        vec![inputs[0] * inputs[0]]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let x = &ctx.saved_tensors()[0];
        vec![grads[0].as_ref().map(|g| g * x * 2.0)]
    }
}

/// `a * b`, saving both inputs.
struct Product;

impl Function for Product {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0], inputs[1]]);
        vec![inputs[0] * inputs[1]]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let saved = ctx.saved_tensors();
        let g = grads[0].as_ref().unwrap();
        vec![Some(g * &saved[1]), Some(g * &saved[0])]
    }
}

/// `exp(x)`, saving the output.
struct SaveOut;

impl Function for SaveOut {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let y = inputs[0].exp();
        ctx.save_for_backward(&[&y]);
        vec![y]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let y = &ctx.saved_tensors()[0];
        vec![grads[0].as_ref().map(|g| g * y)]
    }
}

/// `2 * x`, whose backward returns what the function pointer makes of the incoming gradient.
struct Ret(fn(&Tensor) -> Vec<Option<Tensor>>);

impl Function for Ret {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0] * 2.0]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        (self.0)(grads[0].as_ref().unwrap())
    }
}

/// A function generic over a parameter type, for the node name.
struct Scale<T>(T);

impl<T: Into<f64> + Copy + Send + Sync + 'static> Function for Scale<T> {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0] * self.0.into()]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g * self.0.into())]
    }
}

// --- Names and the recorded node ---

#[test]
fn nodes_are_named_after_the_type() {
    let x = leaf(&[-1.0, 2.0]);
    let y = first(Square, &[&x]);
    let grad_fn = y.grad_fn().unwrap();
    assert_eq!(grad_fn.name(), "SquareBackward");
    let next = grad_fn.next_functions();
    assert_eq!(next.len(), 1);
    assert_eq!(next[0].as_ref().unwrap().0.name(), "AccumulateGrad");
    assert!(format!("{y:?}").contains("grad_fn=<SquareBackward>"));
    // Generic arguments are left out.
    let y = first(Scale(3.0f32), &[&x]);
    assert_eq!(y.grad_fn().unwrap().name(), "ScaleBackward");
    y.sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [3.0, 3.0]);
}

// --- Higher-order derivatives ---

#[test]
fn saved_inputs_give_second_derivatives() {
    let x = leaf(&[0.5, -1.5, 2.0]);
    let y = leaf(&[1.25, 0.75, -0.5]);
    let square = |xs: &[Tensor]| Square.apply(&[&xs[0]]);
    let product = |xs: &[Tensor]| Product.apply(&[&xs[0], &xs[1]]);
    autograd::gradcheck(square, &[&x], no_grad_options()).unwrap();
    autograd::gradgradcheck(square, &[&x], no_grad_options()).unwrap();
    autograd::gradcheck(product, &[&x, &y], no_grad_options()).unwrap();
    autograd::gradgradcheck(product, &[&x, &y], no_grad_options()).unwrap();
    let dx = grad1(&first(Square, &[&x]).sum(), &x, true);
    assert_close(&vals(&dx), &[1.0, -3.0, 4.0]);
    assert_close(&vals(&grad1(&dx.sum(), &x, false)), &[2.0, 2.0, 2.0]);
}

#[test]
fn saved_outputs_give_second_derivatives() {
    let x = leaf(&[0.0, 1.0, -0.5]);
    let f = |xs: &[Tensor]| SaveOut.apply(&[&xs[0]]);
    autograd::gradcheck(f, &[&x], no_grad_options()).unwrap();
    autograd::gradgradcheck(f, &[&x], no_grad_options()).unwrap();
    let dx = grad1(&first(SaveOut, &[&x]).sum(), &x, true);
    let d2x = grad1(&dx.sum(), &x, false);
    assert_close(&vals(&d2x), &vals(&x.detach().exp()));
}

/// `exp(x)` saving the exponential as a hidden intermediate (the output is a separate product),
/// or recomputing it from the saved input.
struct HiddenExp {
    recompute: bool,
}

impl Function for HiddenExp {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let e = inputs[0].exp();
        if self.recompute {
            ctx.save_for_backward(&[inputs[0]]);
        } else {
            ctx.save_for_backward(&[&e]);
        }
        vec![&e * 1.0]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let saved = &ctx.saved_tensors()[0];
        let e = if self.recompute {
            saved.exp()
        } else {
            saved.clone()
        };
        vec![grads[0].as_ref().map(|g| g * &e)]
    }
}

#[test]
fn hidden_intermediates_are_constants() {
    let x = leaf(&[0.0, 1.0]);
    let expected = vals(&x.detach().exp());
    // First order is right either way.
    let dx = grad1(
        &first(HiddenExp { recompute: false }, &[&x]).sum(),
        &x,
        true,
    );
    assert_close(&vals(&dx), &expected);
    // The hidden exponential has no history, so the first derivative isn't connected to x.
    assert!(!dx.requires_grad());
    assert!(dx.grad_fn().is_none());
    let dx = grad1(&first(HiddenExp { recompute: true }, &[&x]).sum(), &x, true);
    assert_close(&vals(&grad1(&dx.sum(), &x, false)), &expected);
    let f = |xs: &[Tensor]| HiddenExp { recompute: true }.apply(&[&xs[0]]);
    autograd::gradgradcheck(f, &[&x], no_grad_options()).unwrap();
}

/// `2x`, saving a product with a leaf weight that it computes under `enable_grad`.
struct HiddenWithHistory {
    w: Tensor,
    seen: Arc<Mutex<Vec<bool>>>,
}

impl Function for HiddenWithHistory {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let h = {
            let _guard = homura::enable_grad();
            inputs[0].detach() * &self.w
        };
        ctx.save_for_backward(&[&h]);
        vec![inputs[0] * 2.0]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let h = &ctx.saved_tensors()[0];
        self.seen.lock().unwrap().push(h.requires_grad());
        vec![grads[0].as_ref().map(|g| g * h)]
    }
}

#[test]
fn saved_tensors_with_a_history_keep_it() {
    let seen = Arc::new(Mutex::new(vec![]));
    let w = leaf(&[3.0]);
    let x = leaf(&[1.0]);
    let f = HiddenWithHistory {
        w: w.clone(),
        seen: seen.clone(),
    };
    let dx = grad1(&first(f, &[&x]).sum(), &x, true);
    assert_eq!(vals(&dx), [3.0]);
    assert_eq!(*seen.lock().unwrap(), [true]);
    assert_eq!(dx.grad_fn().unwrap().name(), "MulBackward0");
    assert_eq!(vals(&grad1(&dx.sum(), &w, false)), [1.0]);
}

// --- Outputs ---

/// Four outputs: `2x`, `3x` marked non-differentiable, `x` as `I64`, `exp(x)`. Records which
/// gradients `backward` received.
struct Mixed {
    seen: Arc<Mutex<Vec<Vec<bool>>>>,
}

impl Function for Mixed {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let x = inputs[0];
        let marked = x * 3.0;
        ctx.mark_non_differentiable(&[&marked]);
        vec![x * 2.0, marked, x.to(DType::I64), x.exp()]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        self.seen
            .lock()
            .unwrap()
            .push(grads.iter().map(Option::is_some).collect());
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

#[test]
fn outputs_mixed_float_integer_and_marked() {
    let seen = Arc::new(Mutex::new(vec![]));
    let x = leaf(&[1.0, 2.0]);
    let outputs = Mixed { seen: seen.clone() }.apply(&[&x]);
    let flags: Vec<_> = outputs.iter().map(Tensor::requires_grad).collect();
    assert_eq!(flags, [true, false, false, true]);
    assert!(outputs[1].grad_fn().is_none() && outputs[2].grad_fn().is_none());
    assert_eq!(outputs[2].dtype(), DType::I64);
    assert!(
        outputs[0]
            .grad_fn()
            .unwrap()
            .is_same(&outputs[3].grad_fn().unwrap())
    );
    outputs[0].sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [2.0, 2.0]);
    // Outputs that received no gradient get `None`.
    assert_eq!(*seen.lock().unwrap(), [vec![true, false, false, false]]);
    assert_panics(
        || outputs[1].sum().backward(),
        "element 0 of tensors does not require grad and does not have a grad_fn",
    );
}

/// Saves a marked floating point output and an integer output; `backward` records whether the
/// unpacked tensors require grad, under `create_graph`.
struct SaveNonDifferentiable {
    seen: Arc<Mutex<Vec<bool>>>,
}

impl Function for SaveNonDifferentiable {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let x = inputs[0];
        let (y, marked, index) = (x * 2.0, x * 3.0, x.argmax());
        ctx.mark_non_differentiable(&[&marked]);
        ctx.save_for_backward(&[&marked, &index, &y]);
        vec![y, marked, index]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let saved = ctx.saved_tensors();
        let mut seen = self.seen.lock().unwrap();
        seen.extend(saved.iter().map(|t| t.requires_grad()));
        seen.extend(saved.iter().map(|t| t.grad_fn().is_some()));
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

#[test]
fn saved_non_differentiable_outputs_have_no_history() {
    let seen = Arc::new(Mutex::new(vec![]));
    let x = leaf(&[1.0, 2.0]);
    let outputs = SaveNonDifferentiable { seen: seen.clone() }.apply(&[&x]);
    let dx = grad1(&outputs[0].sum(), &x, true);
    assert_eq!(vals(&dx), [2.0, 2.0]);
    // The marked and integer outputs come back without history; the differentiable one with
    // its node.
    assert_eq!(
        *seen.lock().unwrap(),
        [false, false, true, false, false, true]
    );
}

/// Returns nothing, or a tensor computed from nothing.
struct Nothing;

impl Function for Nothing {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        if inputs.is_empty() {
            vec![homura::ones_with([2], DType::F64)]
        } else {
            vec![]
        }
    }

    fn backward(&self, _: &BackwardContext<'_>, _: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        unreachable!("no node is recorded")
    }
}

#[test]
fn no_outputs_and_no_inputs() {
    let x = leaf(&[1.0]);
    assert!(Nothing.apply(&[&x]).is_empty());
    let y = first(Nothing, &[]);
    assert!(!y.requires_grad() && y.grad_fn().is_none());
}

/// Builds its output with a recorded operation under `enable_grad`, and stashes the handle it
/// returns.
struct EnableGrad {
    stash: Arc<Mutex<Option<Tensor>>>,
}

impl Function for EnableGrad {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let _guard = homura::enable_grad();
        let w = homura::ones_with([3], DType::F64);
        w.requires_grad_(true);
        let y = (inputs[0].deep_clone() * &w).view([3, 1]).transpose(0, 1);
        *self.stash.lock().unwrap() = Some(y.clone());
        vec![y]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g.reshape([3]) * 4.0)]
    }
}

#[test]
fn history_built_in_forward_is_replaced() {
    let stash = Arc::new(Mutex::new(None));
    let x = leaf(&[1.0, 2.0, 3.0]);
    let y = first(
        EnableGrad {
            stash: stash.clone(),
        },
        &[&x],
    );
    let grad_fn = y.grad_fn().unwrap();
    assert_eq!(grad_fn.name(), "EnableGradBackward");
    assert_eq!(grad_fn.next_functions().len(), 1);
    // The returned handle keeps its own history, and the output keeps its layout.
    let original = stash.lock().unwrap().take().unwrap();
    assert_eq!(original.grad_fn().unwrap().name(), "TransposeBackward0");
    assert!(!original.is_same(&y));
    assert_eq!(
        (y.shape(), y.stride()),
        (original.shape(), original.stride())
    );
    y.sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [4.0, 4.0, 4.0]);
}

/// Returns a view of a hidden intermediate.
struct ViewHidden;

impl Function for ViewHidden {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].exp().view([2, 1])]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g.view([2]))]
    }
}

#[test]
fn in_place_operations_on_outputs() {
    // A view of a hidden intermediate is an ordinary output (PyTorch rejects writing into it).
    let x = leaf(&[1.0, 2.0]);
    let y = first(ViewHidden, &[&x]);
    y.mul_(2.0);
    assert_eq!(y.grad_fn().unwrap().name(), "MulBackward0");
    y.sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [2.0, 2.0]);
    // A write into an output that isn't saved is recorded after the node.
    let y = first(Square, &[&x]);
    y.mul_(2.0);
    let (next, _) = y.grad_fn().unwrap().next_functions()[0].clone().unwrap();
    assert_eq!(next.name(), "SquareBackward");
    // A saved output modified since fails with the operation's name.
    let y = first(SaveOut, &[&x]);
    {
        let _guard = homura::no_grad();
        y.add_(1.0);
    }
    assert_panics(
        || y.sum().backward(),
        "[f64 [2]], which is output 0 of SaveOut, is at version 1; expected version 0 instead.",
    );
}

// --- Whether the call is recorded ---

/// `2x`; records `needs_input_grad` in forward and backward.
struct Flags {
    seen: Arc<Mutex<Vec<Vec<bool>>>>,
}

impl Function for Flags {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        assert!(!homura::is_grad_enabled());
        let flags = (0..inputs.len()).map(|i| ctx.needs_input_grad(i)).collect();
        self.seen.lock().unwrap().push(flags);
        vec![
            inputs
                .iter()
                .fold(homura::zeros_with([2], DType::F64), |acc, t| acc + *t),
        ]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let n = self.seen.lock().unwrap()[0].len();
        let flags = (0..n).map(|i| ctx.needs_input_grad(i)).collect();
        self.seen.lock().unwrap().push(flags);
        vec![grads[0].clone(); n]
    }
}

#[test]
fn recording_depends_on_grad_mode_and_inputs() {
    let seen = Arc::new(Mutex::new(vec![]));
    let x = leaf(&[1.0, 2.0]);
    let c = f64s(&[1.0, 1.0]);
    // In no-grad mode, forward sees the inputs' flags and nothing is recorded.
    let y = {
        let _guard = homura::no_grad();
        first(Flags { seen: seen.clone() }, &[&x, &c])
    };
    assert!(!y.requires_grad() && y.grad_fn().is_none());
    assert_eq!(*seen.lock().unwrap(), [vec![true, false]]);
    // Without an input that requires grad, nothing is recorded either.
    let y = first(Square, &[&c]);
    assert!(!y.requires_grad() && y.grad_fn().is_none());
    // An input that requires grad without an edge (a view created in no-grad mode).
    let v = {
        let _guard = homura::no_grad();
        x.view([2])
    };
    let y = first(Square, &[&v]);
    let next = y.grad_fn().unwrap().next_functions();
    assert!(next[0].is_none());
    y.sum().backward();
    assert!(x.grad().is_none());
}

/// `2x`, counting its forward calls.
struct CountForward {
    calls: Arc<AtomicUsize>,
}

impl Function for CountForward {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        vec![inputs[0] * 2.0]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

#[test]
fn stale_input_views_fail_before_forward_runs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let x = leaf(&[1.0, 2.0]);
    let base = &x * 1.0;
    let v = base.view([2]);
    base.mul_(2.0);
    let f = CountForward {
        calls: calls.clone(),
    };
    assert_panics(
        || drop(f.apply(&[&v])),
        "a view whose base was modified by an in-place operation in grad mode",
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    // In no-grad mode nothing is recorded, so the view's history isn't used.
    let _guard = homura::no_grad();
    let f = CountForward {
        calls: calls.clone(),
    };
    assert_eq!(vals(&first(f, &[&v])), [4.0, 8.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// --- Inputs ---

#[test]
fn repeated_and_partially_requested_inputs() {
    // A repeated input has two edges to the same accumulator; its gradients are summed.
    let x = leaf(&[3.0]);
    let y = first(Product, &[&x, &x]);
    let next = y.grad_fn().unwrap().next_functions();
    assert!(
        next[0]
            .as_ref()
            .unwrap()
            .0
            .is_same(&next[1].as_ref().unwrap().0)
    );
    y.sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [6.0]);
    // A gradient for an input that doesn't require grad is dropped.
    let (a, b) = (leaf(&[2.0]), f64s(&[5.0]));
    first(Product, &[&a, &b]).sum().backward();
    assert_eq!(vals(&a.grad().unwrap()), [5.0]);
    assert!(b.grad().is_none());
    // Asking for one input's gradient doesn't change `needs_input_grad`.
    let seen = Arc::new(Mutex::new(vec![]));
    let (a, b) = (leaf(&[1.0, 2.0]), leaf(&[3.0, 4.0]));
    let y = first(Flags { seen: seen.clone() }, &[&a, &b]);
    assert_eq!(vals(&grad1(&y.sum(), &a, false)), [1.0, 1.0]);
    assert_eq!(*seen.lock().unwrap(), [vec![true, true], vec![true, true]]);
}

/// Converts to `F16`; records the dtype of the gradient it receives.
struct ToHalf {
    seen: Arc<Mutex<Option<DType>>>,
}

impl Function for ToHalf {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].to(DType::F16)]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        *self.seen.lock().unwrap() = Some(grads[0].as_ref().unwrap().dtype());
        vec![grads[0].clone()]
    }
}

#[test]
fn meta_inputs_and_mixed_dtypes() {
    let x = homura::ones_with([2], Device::Meta);
    x.requires_grad_(true);
    let y = first(SaveOut, &[&x]);
    assert_eq!(y.device(), Device::Meta);
    let g = grad1(&y.sum(), &x, false);
    assert_eq!((g.device(), g.shape().to_vec()), (Device::Meta, vec![2]));
    let seen = Arc::new(Mutex::new(None));
    let x = homura::ones([2]);
    x.requires_grad_(true);
    first(ToHalf { seen: seen.clone() }, &[&x]).sum().backward();
    assert_eq!(*seen.lock().unwrap(), Some(DType::F16));
    assert_eq!(x.grad().unwrap().dtype(), DType::F32);
}

// --- The gradients backward returns ---

#[test]
fn returned_gradient_counts() {
    let x = leaf(&[1.0, 2.0]);
    // Extra entries are accepted while they are all `None`.
    for f in [
        (|g: &Tensor| vec![Some(g.clone())]) as fn(&Tensor) -> Vec<Option<Tensor>>,
        |g| vec![Some(g.clone()), None],
        |g| vec![Some(g.clone()), None, None],
    ] {
        assert_eq!(
            vals(&grad1(&first(Ret(f), &[&x]).sum(), &x, false)),
            [1.0, 1.0]
        );
    }
    let message = |f: fn(&Tensor) -> Vec<Option<Tensor>>| {
        let y = first(Ret(f), &[&x]).sum();
        panic_message(|| drop(grad1(&y, &x, false)))
    };
    assert_eq!(
        message(|g| vec![Some(g.clone()), None, Some(g.clone())]),
        "function RetBackward returned an incorrect number of gradients (expected 1, got 3)"
    );
    assert_eq!(
        message(|_| vec![]),
        "function RetBackward returned an incorrect number of gradients (expected 1, got 0)"
    );
    // An input without an edge still has its entry.
    let (a, b) = (leaf(&[1.0]), f64s(&[2.0]));
    struct Short;
    impl Function for Short {
        fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
            vec![inputs[0] * inputs[1]]
        }
        fn backward(&self, _: &BackwardContext<'_>, g: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
            vec![g[0].clone()]
        }
    }
    let y = first(Short, &[&a, &b]).sum();
    assert_eq!(
        panic_message(|| y.backward()),
        "function ShortBackward returned an incorrect number of gradients (expected 2, got 1)"
    );
}

#[test]
fn returned_gradients_are_reduced_and_converted() {
    let x = homura::ones_with([2, 3], DType::F64);
    x.requires_grad_(true);
    // A larger shape that the input broadcasts to is summed.
    let g = grad1(
        &first(Ret(|_| vec![Some(homura::ones([4, 2, 3]))]), &[&x]).sum(),
        &x,
        false,
    );
    assert_eq!((g.shape().to_vec(), vals(&g)), (vec![2, 3], vec![4.0; 6]));
    for f in [
        (|_: &Tensor| vec![Some(homura::ones([3]))]) as fn(&Tensor) -> Vec<Option<Tensor>>,
        |_| vec![Some(homura::ones([1, 3]))],
        |_| vec![Some(homura::ones([5]))],
        |_| vec![Some(homura::ones([]))],
    ] {
        let y = first(Ret(f), &[&x]).sum();
        assert_panics(
            || drop(grad1(&y, &x, false)),
            "Function RetBackward returned an invalid gradient at index 0 - got [",
        );
    }
    let y = first(Ret(|_| vec![Some(homura::ones([3]))]), &[&x]).sum();
    assert_eq!(
        panic_message(|| drop(grad1(&y, &x, false))),
        "Function RetBackward returned an invalid gradient at index 0 - got [3] but expected \
         shape compatible with [2, 3]"
    );
    // Empty and size-1 dimensions.
    let e = homura::ones_with([0], DType::F64);
    e.requires_grad_(true);
    let g = grad1(
        &first(Ret(|_| vec![Some(homura::ones([3, 0]))]), &[&e]).sum(),
        &e,
        false,
    );
    assert_eq!(g.shape().to_vec(), [0]);
    let r = homura::ones_with([1, 3], DType::F64);
    r.requires_grad_(true);
    let g = grad1(
        &first(
            Ret(|_| vec![Some(homura::arange(6.0).reshape([2, 3]))]),
            &[&r],
        )
        .sum(),
        &r,
        false,
    );
    assert_eq!(
        (g.shape().to_vec(), vals(&g)),
        (vec![1, 3], vec![3.0, 5.0, 7.0])
    );
    // Other dtypes are converted.
    let x = homura::ones([2]);
    x.requires_grad_(true);
    for f in [
        (|g: &Tensor| vec![Some(g.to(DType::F64))]) as fn(&Tensor) -> Vec<Option<Tensor>>,
        |g| vec![Some(g.to(DType::I64))],
        |g| vec![Some(g.to(DType::Bool))],
    ] {
        let g = grad1(&first(Ret(f), &[&x]).sum(), &x, false);
        assert_eq!((g.dtype(), vals(&g)), (DType::F32, vec![1.0, 1.0]));
    }
}

/// `x * x` whose backward returns its gradient expanded to `[4, 2, 3]`.
struct Larger;

impl Function for Larger {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0]]);
        vec![inputs[0] * inputs[0]]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let x = &ctx.saved_tensors()[0];
        vec![grads[0].as_ref().map(|g| (g * x * 2.0).expand([4, 2, 3]))]
    }
}

#[test]
fn reduced_gradients_stay_differentiable() {
    let x = homura::ones_with([2, 3], DType::F64);
    x.requires_grad_(true);
    let dx = grad1(&first(Larger, &[&x]).sum(), &x, true);
    assert_eq!(vals(&dx), [8.0; 6]);
    assert_eq!(vals(&grad1(&dx.sum(), &x, false)), [8.0; 6]);
}

#[test]
fn returned_gradients_on_another_device() {
    // A zero-dimensional input takes a gradient from another device once it is summed to a
    // scalar.
    for f in [
        (|_: &Tensor| vec![Some(homura::ones([]))]) as fn(&Tensor) -> Vec<Option<Tensor>>,
        |_| vec![Some(homura::ones([4]))],
        |_| vec![Some(homura::ones([2, 3]))],
    ] {
        let x = homura::ones_with([], Device::Meta);
        x.requires_grad_(true);
        let g = grad1(&first(Ret(f), &[&x]), &x, false);
        assert_eq!((g.device(), g.shape().to_vec()), (Device::Meta, vec![]));
    }
    // Moving a meta gradient to the CPU fails like any copy out of a meta tensor.
    for f in [
        (|_: &Tensor| vec![Some(homura::ones_with([], Device::Meta))])
            as fn(&Tensor) -> Vec<Option<Tensor>>,
        |_| vec![Some(homura::ones_with([4], Device::Meta))],
    ] {
        let x = homura::ones([]);
        x.requires_grad_(true);
        let y = first(Ret(f), &[&x]);
        assert_panics(
            || drop(grad1(&y, &x, false)),
            "Cannot copy out of meta tensor",
        );
    }
    // A gradient that isn't a scalar after the sum must be on the input's device.
    let x = homura::ones_with([2, 3], Device::Meta);
    x.requires_grad_(true);
    let y = first(Ret(|_| vec![Some(homura::ones([4, 2, 3]))]), &[&x]).sum();
    assert_eq!(
        panic_message(|| drop(grad1(&y, &x, false))),
        "Function RetBackward returned an invalid gradient at index 0 - expected device meta but \
         got cpu"
    );
    let x = homura::ones([2, 3]);
    x.requires_grad_(true);
    let y = first(
        Ret(|_| vec![Some(homura::ones_with([2, 3], Device::Meta))]),
        &[&x],
    )
    .sum();
    assert_eq!(
        panic_message(|| drop(grad1(&y, &x, false))),
        "Function RetBackward returned an invalid gradient at index 0 - expected device cpu but \
         got meta"
    );
    // The shape is checked first.
    let y = first(
        Ret(|_| vec![Some(homura::ones_with([5], Device::Meta))]),
        &[&x],
    )
    .sum();
    assert_panics(
        || drop(grad1(&y, &x, false)),
        "returned an invalid gradient at index 0 - got [5]",
    );
}

#[test]
fn root_gradients_on_another_device() {
    let x = homura::ones([2]);
    x.requires_grad_(true);
    let y = &x * 2.0;
    assert_eq!(
        panic_message(|| y.backward_with(&homura::ones_with([2], Device::Meta))),
        "invalid gradient at index 0 - expected device cpu but got meta"
    );
    let x = homura::ones_with([2], Device::Meta);
    x.requires_grad_(true);
    let y = &x * 2.0;
    let options = GradOptions {
        grad_outputs: vec![Some(homura::ones([2]))],
        ..Default::default()
    };
    assert_eq!(
        panic_message(|| drop(autograd::grad(&[&y], &[&x], options))),
        "invalid gradient at index 0 - expected device meta but got cpu"
    );
    // A zero-dimensional gradient is moved.
    let x = homura::ones_with([], Device::Meta);
    x.requires_grad_(true);
    let y = &x * 2.0;
    let options = GradOptions {
        grad_outputs: vec![Some(homura::ones([]))],
        ..Default::default()
    };
    let g = autograd::grad(&[&y], &[&x], options).remove(0).unwrap();
    assert_eq!(g.device(), Device::Meta);
}

// --- What forward may not do ---

/// Adds 1 to its input in place.
struct Modify;

impl Function for Modify {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        inputs[0].add_(1.0);
        vec![inputs[0] * 2.0]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

#[test]
fn forward_must_not_modify_inputs() {
    let message = "Modify::forward modified input 0 in place, which custom functions don't \
                   support (PyTorch's mark_dirty isn't available): modify a deep_clone() of it \
                   instead";
    let x = leaf(&[1.0, 2.0]);
    let h = &x * 1.0;
    let c = f64s(&[1.0, 2.0]);
    for t in [&x, &h, &c] {
        assert_eq!(panic_message(|| drop(Modify.apply(&[t]))), message);
        // The write is not undone.
        assert_eq!(vals(t), [2.0, 3.0]);
    }
    let _guard = homura::no_grad();
    assert_eq!(panic_message(|| drop(Modify.apply(&[&x]))), message);
}

/// Returns what the function pointer makes of its input.
struct Alias(fn(&Tensor) -> Vec<Tensor>);

impl Function for Alias {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        (self.0)(inputs[0])
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].clone()]
    }
}

#[test]
fn outputs_must_not_share_memory() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0]);
    for f in [
        (|x: &Tensor| vec![x.clone()]) as fn(&Tensor) -> Vec<Tensor>,
        |x| vec![x.view([2, 2])],
        |x| vec![x * 2.0, x.narrow(0, 1, 2)],
    ] {
        assert_panics(
            || drop(Alias(f).apply(&[&x])),
            "of Alias::forward shares memory with input 0, which custom functions don't \
             support: return a deep_clone() of it instead",
        );
    }
    for f in [
        (|x: &Tensor| {
            let y = x * 2.0;
            vec![y.clone(), y]
        }) as fn(&Tensor) -> Vec<Tensor>,
        |x| (x * 2.0).chunk(2, 0),
    ] {
        assert_eq!(
            panic_message(|| drop(Alias(f).apply(&[&x]))),
            "output 1 of Alias::forward shares memory with output 0, which custom functions \
             don't support: return a deep_clone() of one of them instead"
        );
    }
}

/// Returns `2x` and `3x`, marking what the function pointer chooses.
struct Mark(fn(&mut ForwardContext, &Tensor, &[Tensor]));

impl Function for Mark {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let outputs = vec![inputs[0] * 2.0, inputs[0] * 3.0];
        (self.0)(ctx, inputs[0], &outputs);
        outputs
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let g = grads
            .iter()
            .flatten()
            .fold(homura::zeros_with([2], DType::F64), |a, g| a + g);
        vec![Some(g)]
    }
}

#[test]
fn marking_non_differentiable_outputs() {
    let x = leaf(&[1.0, 2.0]);
    let flags = |f: fn(&mut ForwardContext, &Tensor, &[Tensor])| -> Vec<bool> {
        Mark(f)
            .apply(&[&x])
            .iter()
            .map(Tensor::requires_grad)
            .collect()
    };
    // A later call replaces an earlier one.
    assert_eq!(
        flags(|ctx, _, o| {
            ctx.mark_non_differentiable(&[&o[0]]);
            ctx.mark_non_differentiable(&[&o[1]]);
        }),
        [true, false]
    );
    assert_eq!(
        flags(|ctx, _, o| ctx.mark_non_differentiable(&[&o[0], &o[0]])),
        [false, true]
    );
    assert_eq!(
        flags(|ctx, _, o| {
            ctx.mark_non_differentiable(&[&o[0]]);
            ctx.mark_non_differentiable(&[]);
        }),
        [true, true]
    );
    // Only the last call is checked.
    assert_eq!(
        flags(|ctx, x, o| {
            ctx.mark_non_differentiable(&[x]);
            ctx.mark_non_differentiable(&[&o[1]]);
        }),
        [true, false]
    );
    let message = "Mark::forward marked a tensor as non-differentiable that it doesn't return; \
                   mark only the tensors it returns";
    for f in [
        (|ctx: &mut ForwardContext, x: &Tensor, _: &[Tensor]| ctx.mark_non_differentiable(&[x]))
            as fn(&mut ForwardContext, &Tensor, &[Tensor]),
        |ctx, x, _| ctx.mark_non_differentiable(&[&(x * 1.0)]),
    ] {
        assert_eq!(panic_message(|| drop(Mark(f).apply(&[&x]))), message);
        let _guard = homura::no_grad();
        assert_eq!(panic_message(|| drop(Mark(f).apply(&[&x]))), message);
    }
}

/// Saves `x` twice, then `10 x`.
struct SaveTwice;

impl Function for SaveTwice {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let x = inputs[0];
        ctx.save_for_backward(&[x, x]);
        ctx.save_for_backward(&[&(x * 10.0)]);
        vec![x * 2.0]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let saved = ctx.saved_tensors();
        assert_eq!(saved.len(), 1);
        vec![grads[0].as_ref().map(|g| g * &saved[0])]
    }
}

#[test]
fn saving_again_replaces() {
    let x = leaf(&[1.0]);
    first(SaveTwice, &[&x]).sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [10.0]);
}

// --- Missing gradients and calls that record nothing ---

/// Returns no gradient.
struct NoneGrad;

impl Function for NoneGrad {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0] * 1.0]
    }

    fn backward(&self, _: &BackwardContext<'_>, _: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![None]
    }
}

/// `x * x`, counting its backward calls.
struct Counted {
    calls: Arc<AtomicUsize>,
}

impl Function for Counted {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0]]);
        vec![inputs[0] * inputs[0]]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let x = &ctx.saved_tensors()[0];
        vec![grads[0].as_ref().map(|g| g * x * 2.0)]
    }
}

#[test]
fn backward_is_skipped_when_no_gradient_arrives() {
    let calls = Arc::new(AtomicUsize::new(0));
    let x = leaf(&[1.0, 2.0]);
    let y = first(
        Counted {
            calls: calls.clone(),
        },
        &[&x],
    );
    let z = first(NoneGrad, &[&y]);
    // PyTorch calls backward with zeros (or `None` without materialization) and returns zeros.
    assert!(grads(&z.sum(), &[&x], false)[0].is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    // The pass freed the saved tensors all the same.
    let w = &y * 1.0;
    assert_panics(
        || w.sum().backward(),
        "Trying to backward through the graph a second time",
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Returns only outputs that can't be differentiated; holds a token to show when it is dropped.
struct NoDifferentiable {
    _token: Arc<()>,
}

impl Function for NoDifferentiable {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let x = inputs[0];
        let marked = x * 2.0;
        ctx.mark_non_differentiable(&[&marked]);
        ctx.save_for_backward(&[x, &marked]);
        vec![x.to(DType::I64), x.greater(0.0), marked]
    }

    fn backward(&self, _: &BackwardContext<'_>, _: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        unreachable!("no node is recorded")
    }
}

#[test]
fn calls_without_differentiable_outputs_record_nothing() {
    let token = Arc::new(());
    let x = leaf(&[1.0, -2.0]);
    let outputs = NoDifferentiable {
        _token: token.clone(),
    }
    .apply(&[&x]);
    assert!(
        outputs
            .iter()
            .all(|t| !t.requires_grad() && t.grad_fn().is_none())
    );
    // The function value was dropped with the context.
    assert_eq!(Arc::strong_count(&token), 1);
}

/// Returns a transposed copy of its input, stashing the handle it returns.
struct Stash {
    enable_grad: bool,
    stash: Arc<Mutex<Option<Tensor>>>,
}

impl Function for Stash {
    fn forward(&mut self, _: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        let _guard = homura::set_grad_enabled(self.enable_grad);
        let w = homura::ones_with([2, 3], DType::F64);
        w.requires_grad_(self.enable_grad);
        let y = (inputs[0].deep_clone() * &w).transpose(0, 1);
        *self.stash.lock().unwrap() = Some(y.clone());
        vec![y]
    }

    fn backward(&self, _: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        vec![grads[0].as_ref().map(|g| g.transpose(0, 1))]
    }
}

#[test]
fn unrecorded_calls_return_new_tensors() {
    let values: Vec<f64> = (0..6).map(f64::from).collect();
    let x = Tensor::from_slice(&values, [2, 3]);
    x.requires_grad_(true);
    let c = Tensor::from_slice(&values, [2, 3]);
    for (input, outer_no_grad, enable_grad) in
        [(&x, true, false), (&c, false, false), (&x, true, true)]
    {
        let stash = Arc::new(Mutex::new(None));
        let y = {
            let _guard = homura::set_grad_enabled(!outer_no_grad);
            first(
                Stash {
                    enable_grad,
                    stash: stash.clone(),
                },
                &[input],
            )
        };
        let original = stash.lock().unwrap().take().unwrap();
        assert!(!y.is_same(&original));
        assert!(!y.requires_grad() && y.grad_fn().is_none() && y.is_leaf());
        assert_eq!(
            (y.shape().to_vec(), y.stride().to_vec()),
            (vec![3, 2], vec![1, 3])
        );
        assert_eq!(vals(&y), vals(&original));
        // The returned handle is left as it was.
        assert_eq!(original.requires_grad(), enable_grad);
        assert_eq!(original.grad_fn().is_some(), enable_grad);
    }
}

// --- Saved tensors, lifetimes and panics ---

#[test]
fn saved_tensors_are_checked_and_freed() {
    let x = leaf(&[1.0, 2.0]);
    let y = first(Square, &[&x]);
    {
        let _guard = homura::no_grad();
        x.add_(1.0);
    }
    assert_panics(
        || y.sum().backward(),
        "[f64 [2]] is at version 1; expected version 0 instead.",
    );
    let x = leaf(&[1.0, 2.0]);
    let s = first(Square, &[&x]).sum();
    let retain = BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    autograd::backward(&[&s], &[], retain.clone());
    autograd::backward(&[&s], &[], retain);
    assert_eq!(vals(&x.grad().unwrap()), [4.0, 8.0]);
    s.backward();
    assert_panics(
        || s.backward(),
        "Trying to backward through the graph a second time",
    );
}

/// `sin(x)`, whose backward computes `cos(x)` with a nested backward pass of its own.
struct Nested;

impl Function for Nested {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0]]);
        vec![inputs[0].sin()]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        let x = &ctx.saved_tensors()[0];
        let inner = {
            let _guard = homura::enable_grad();
            let t = x.detach();
            t.requires_grad_(true);
            grad1(&t.sin().sum(), &t, false)
        };
        vec![grads[0].as_ref().map(|g| g * &inner)]
    }
}

#[test]
fn backward_can_run_nested_passes() {
    let x = leaf(&[0.5, -1.0]);
    let g = grad1(&first(Nested, &[&x]).sum(), &x, false);
    assert_close(&vals(&g), &vals(&x.detach().cos()));
}

/// `2x`; panics in forward, or in its first backward call.
struct Panicky {
    in_forward: bool,
    armed: Arc<AtomicBool>,
    _token: Arc<()>,
}

impl Function for Panicky {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0]]);
        if self.in_forward {
            panic!("forward failed");
        }
        vec![inputs[0] * 2.0]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        if self.armed.swap(false, Ordering::SeqCst) {
            panic!("backward failed");
        }
        assert_eq!(ctx.saved_tensors().len(), 1);
        vec![grads[0].as_ref().map(|g| g * 2.0)]
    }
}

#[test]
fn panics_leave_grad_mode_and_the_graph_usable() {
    let token = Arc::new(());
    let armed = Arc::new(AtomicBool::new(true));
    let x = leaf(&[1.0]);
    let make = |in_forward| Panicky {
        in_forward,
        armed: armed.clone(),
        _token: token.clone(),
    };
    assert_eq!(
        panic_message(|| drop(make(true).apply(&[&x]))),
        "forward failed"
    );
    assert!(homura::is_grad_enabled());
    assert_eq!(Arc::strong_count(&token), 1);
    // The same type can be applied again; a panic in backward leaves the node usable.
    let s = first(make(false), &[&x]).sum();
    let retain = BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    assert_eq!(
        panic_message(|| autograd::backward(&[&s], &[], retain.clone())),
        "backward failed"
    );
    assert!(homura::is_grad_enabled());
    autograd::backward(&[&s], &[], retain);
    assert_eq!(vals(&x.grad().unwrap()), [2.0]);
    // The node and the function value go with the last output.
    assert_eq!(Arc::strong_count(&token), 2);
    drop(s);
    assert_eq!(Arc::strong_count(&token), 1);
}

// --- Threads ---

#[test]
fn independent_calls_run_concurrently() {
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let x = leaf(&[f64::from(i), 1.0]);
                first(Square, &[&x]).sum().backward();
                vals(&x.grad().unwrap())
            })
        })
        .collect();
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(handle.join().unwrap(), [2.0 * i as f64, 2.0]);
    }
}

/// `x * x`; in the thread named `slow`, backward waits at `entered` and then `released` before
/// it unpacks the saved input.
struct Shared {
    entered: Arc<Barrier>,
    released: Arc<Barrier>,
}

impl Function for Shared {
    fn forward(&mut self, ctx: &mut ForwardContext, inputs: &[&Tensor]) -> Vec<Tensor> {
        ctx.save_for_backward(&[inputs[0]]);
        vec![inputs[0] * inputs[0]]
    }

    fn backward(&self, ctx: &BackwardContext<'_>, grads: &[Option<Tensor>]) -> Vec<Option<Tensor>> {
        if thread::current().name() == Some("slow") {
            self.entered.wait();
            self.released.wait();
        }
        let x = &ctx.saved_tensors()[0];
        vec![grads[0].as_ref().map(|g| g * x * 2.0)]
    }
}

/// Two passes through the same node, the one on thread `slow` paused inside its backward call
/// while the other one runs to completion; returns the results of `slow` and of the other.
fn concurrent_passes(retain_graph: bool) -> (thread::Result<Vec<f64>>, Vec<f64>, Tensor, Tensor) {
    let (entered, released) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    let x = leaf(&[1.0, 2.0]);
    let y = first(
        Shared {
            entered: entered.clone(),
            released: released.clone(),
        },
        &[&x],
    );
    let pass = move |x: &Tensor, y: &Tensor| {
        let options = GradOptions {
            retain_graph: Some(retain_graph),
            ..Default::default()
        };
        let g = autograd::grad(&[&y.sum()], &[x], options)
            .remove(0)
            .unwrap();
        vals(&g)
    };
    let slow = {
        let (x, y) = (x.clone(), y.clone());
        thread::Builder::new()
            .name("slow".into())
            .spawn(move || pass(&x, &y))
            .unwrap()
    };
    entered.wait();
    let fast = catch_unwind(AssertUnwindSafe(|| pass(&x, &y)));
    released.wait();
    let slow = slow.join();
    (slow, fast.unwrap(), x, y)
}

#[test]
fn shared_graphs_run_concurrently_with_retain_graph() {
    let (slow, fast, x, y) = concurrent_passes(true);
    assert_eq!(slow.unwrap(), [2.0, 4.0]);
    assert_eq!(fast, [2.0, 4.0]);
    // Without concurrent passes, the saved tensors are freed as usual.
    y.sum().backward();
    assert_eq!(vals(&x.grad().unwrap()), [2.0, 4.0]);
    assert_panics(
        || y.sum().backward(),
        "Trying to backward through the graph a second time",
    );
}

#[test]
fn shared_graphs_without_retain_graph_free_the_saved_tensors() {
    let (slow, fast, _, _) = concurrent_passes(false);
    assert_eq!(fast, [2.0, 4.0]);
    let payload = slow.expect_err("the paused pass finds its saved tensors freed");
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .unwrap_or_default();
    assert!(
        message.contains("Trying to backward through the graph a second time"),
        "{message}"
    );
}
