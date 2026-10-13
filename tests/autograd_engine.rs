//! The backward engine and the public autograd API (M1 step 3): first- and second-order
//! gradients through the operations that have formulas, graph release, grad mode, view
//! properties, operations whose derivative isn't implemented yet, the in-place guard and the
//! error messages. `tools/golden/check_m1_engine.py` checks the same cases against the reference
//! PyTorch build. Run in debug and release profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;

use homura::autograd::{self, BackwardOptions, GradFn, GradOptions};
use homura::{DType, Device, Tensor};

fn t(values: &[f32]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

/// A leaf that requires grad.
fn leaf(values: &[f32]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_vec()
}

fn grad_of(x: &Tensor) -> Vec<f32> {
    vals(&x.grad().expect("a gradient"))
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-5 * e.abs().max(1.0),
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

/// `autograd::grad` with every result present.
fn grad(outputs: &[&Tensor], inputs: &[&Tensor], options: GradOptions) -> Vec<Tensor> {
    autograd::grad(outputs, inputs, options)
        .into_iter()
        .map(|g| g.expect("a gradient"))
        .collect()
}

fn create_graph() -> GradOptions {
    GradOptions {
        create_graph: true,
        ..Default::default()
    }
}

/// The names of the nodes reachable from `f`, depth first.
fn graph_names(f: &GradFn) -> Vec<&'static str> {
    let mut names = vec![];
    let mut stack = vec![f.clone()];
    while let Some(f) = stack.pop() {
        names.push(f.name());
        stack.extend(f.next_functions().into_iter().flatten().map(|(f, _)| f));
    }
    names
}

/// The node names of `f`'s edges, `None` where an input has none.
fn next_names(f: &GradFn) -> Vec<Option<&'static str>> {
    f.next_functions()
        .iter()
        .map(|e| e.as_ref().map(|(f, _)| f.name()))
        .collect()
}

// ---------------------------------------------------------------- first order

#[test]
fn first_order_gradients_of_products_and_sums() {
    let x = leaf(&[1., 2., 3.]);
    (&x * &x).sum().backward();
    assert_eq!(grad_of(&x), [2., 4., 6.]);
    // A diamond: `a` feeds two branches that meet again, and `x` is also used directly.
    let x = leaf(&[0., 1., -1.]);
    let a = x.exp();
    ((&a * 2.0) + (&a * 3.0) - &x).sum().backward();
    let e = [1f32, 1f32.exp(), (-1f32).exp()];
    assert_close(&grad_of(&x), &e.map(|v| 5. * v - 1.));
}

#[test]
fn explicit_upstream_gradients_with_broadcasting() {
    let x = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], [2, 3]);
    x.requires_grad_(true);
    let w = leaf(&[0.5, -1., 2.]);
    let y = &x * &w + &w;
    let g = Tensor::from_slice(&[1f32, 0., 2., -1., 3., 0.5], [2, 3]);
    y.backward_with(&g);
    assert_eq!(grad_of(&x), [0.5, 0., 4., -0.5, -3., 1.]);
    assert_eq!(grad_of(&w), [-3., 18., 11.5]);
    assert_eq!(x.grad().unwrap().shape(), [2, 3]);
}

#[test]
fn scalar_operands_and_promoted_dtypes() {
    let x = leaf(&[1., 2.]);
    (2.0 * &x - 1.0).sum().backward();
    assert_eq!(grad_of(&x), [2., 2.]);
    let x = leaf(&[1., 2.]);
    (1.0 - &x).sum().backward();
    assert_eq!(grad_of(&x), [-1., -1.]);
    let x = leaf(&[1., 2.]);
    (1.0 + &x * 3.0).sum().backward();
    assert_eq!(grad_of(&x), [3., 3.]);
    // Gradients are converted back to the dtype of each input.
    let x = leaf(&[1., 2.]);
    let w = Tensor::from_slice(&[3f64, 4.], [2]);
    w.requires_grad_(true);
    let y = (&x * &w).sum();
    assert_eq!(y.dtype(), DType::F64);
    y.backward();
    assert_eq!(x.grad().unwrap().dtype(), DType::F32);
    assert_eq!(grad_of(&x), [3., 4.]);
    assert_eq!(w.grad().unwrap().to_vec::<f64>(), [1., 2.]);
    let h = Tensor::from_slice(&[1f32, 2.], [2]).to(DType::F16);
    h.requires_grad_(true);
    (&h * 3.0).sum().backward();
    assert_eq!(h.grad().unwrap().dtype(), DType::F16);
    assert_eq!(h.grad().unwrap().to_vec::<f32>(), [3., 3.]);
    // An integer operand never requires grad.
    let x = leaf(&[1., 2.]);
    let n = homura::tensor([3i64, 4]);
    (&x * &n).sum().backward();
    assert_eq!(grad_of(&x), [3., 4.]);
}

#[test]
fn explicit_gradients_are_converted_to_the_output_dtype() {
    let x = leaf(&[1., 2.]);
    (&x * 5.0)
        .sum()
        .backward_with(&Tensor::from_slice(&[2f64], []));
    assert_eq!(x.grad().unwrap().dtype(), DType::F32);
    assert_eq!(grad_of(&x), [10., 10.]);
}

#[test]
fn single_element_outputs_get_an_implicit_gradient() {
    let x = leaf(&[1., 2.]);
    (&x * &x).sum().view([1]).backward();
    assert_eq!(grad_of(&x), [2., 4.]);
    let x = leaf(&[3.]);
    (&x * 2.0).view([1, 1]).backward();
    assert_eq!(grad_of(&x), [2.]);
    let message = panic_message(|| (&x * 2.0).expand([2]).backward());
    assert_eq!(
        message,
        "grad can be implicitly created only for scalar outputs"
    );
}

#[test]
fn gradients_accumulate_across_backward_passes_in_place() {
    let x = leaf(&[1., 2.]);
    (&x * 3.0).sum().backward();
    let first = x.grad().unwrap();
    // The gradient of `sum` is an expanded view; the stored one is a writable copy.
    assert!(first.is_contiguous());
    (&x * &x).sum().backward();
    assert!(x.grad().unwrap().is_same(&first), "accumulated in place");
    assert_eq!(grad_of(&x), [5., 7.]);
    {
        let _no_grad = homura::no_grad();
        first.add_(1.0);
    }
    assert_eq!(grad_of(&x), [6., 8.]);
}

#[test]
fn grad_mode_switches_recording() {
    let x = leaf(&[1., 2.]);
    {
        let _no_grad = homura::no_grad();
        let y = &x * &x;
        assert!(!y.requires_grad() && y.is_leaf() && y.grad_fn().is_none());
        let _enabled = homura::enable_grad();
        assert!((&x * &x).requires_grad());
    }
    {
        let _off = homura::set_grad_enabled(false);
        assert!(!x.exp().requires_grad());
    }
    let y = &x * &x;
    assert!(y.requires_grad() && !y.is_leaf());
}

#[test]
fn grad_returns_gradients_without_writing_them() {
    let x = leaf(&[1., 2.]);
    let w = leaf(&[3., 4.]);
    let y = (&x * &w).sum();
    let g = grad(&[&y], &[&x], GradOptions::default());
    assert_eq!(vals(&g[0]), [3., 4.]);
    assert!(x.grad().is_none() && w.grad().is_none());
    // Non-leaf inputs: the gradient flowing into them.
    let a = &x * 2.0;
    let z = (&a * &a).sum();
    let g = grad(&[&z], &[&a, &x], GradOptions::default());
    assert_eq!(vals(&g[0]), [4., 8.]);
    assert_eq!(vals(&g[1]), [8., 16.]);
    assert!(x.grad().is_none());
    // Several outputs; their gradients add up.
    let g = grad(
        &[&(&x * 2.0).sum(), &(&x * 3.0).sum()],
        &[&x],
        GradOptions::default(),
    );
    assert_eq!(vals(&g[0]), [5., 5.]);
}

#[test]
fn grad_runs_only_the_part_of_the_graph_it_needs() {
    let x = leaf(&[1., 2.]);
    let w = leaf(&[3., 4.]);
    // `softmax` has no formula yet, but `w` doesn't need it.
    let y = (x.softmax(0) + &w).sum();
    let g = grad(&[&y], &[&w], GradOptions::default());
    assert_eq!(vals(&g[0]), [1., 1.]);
    let message = panic_message(|| y.backward());
    assert_eq!(message, "the derivative for 'softmax' is not implemented.");
}

#[test]
fn several_roots_and_shared_leaves() {
    let x = leaf(&[1., 2.]);
    let a = &x * 2.0;
    let b = &a * 3.0;
    autograd::backward(
        &[&a.sum(), &b.sum()],
        &[None, Some(&Tensor::from_slice(&[2f32], []))],
        BackwardOptions::default(),
    );
    assert_eq!(grad_of(&x), [14., 14.]);
}

#[test]
fn independent_graphs_on_several_threads() {
    let shared = leaf(&[0., 0.]);
    let workers: Vec<_> = (0..8)
        .map(|k| {
            let shared = shared.clone();
            thread::spawn(move || {
                let x = leaf(&[k as f32, 1.]);
                for _ in 0..50 {
                    (&x * &x).sum().backward();
                    (&shared * (k as f64)).sum().backward();
                }
                grad_of(&x)
            })
        })
        .collect();
    for (k, worker) in workers.into_iter().enumerate() {
        assert_eq!(worker.join().unwrap(), [100. * k as f32, 100.]);
    }
    assert_eq!(grad_of(&shared), [1400., 1400.]);
}

#[test]
fn meta_leaves_get_gradients_of_the_right_shape() {
    let x = homura::zeros_with([2, 3], Device::Meta);
    x.requires_grad_(true);
    let w = homura::zeros_with([3], (DType::F64, Device::Meta));
    w.requires_grad_(true);
    (&x * &w + 1.0).exp().sum().backward();
    let (gx, gw) = (x.grad().unwrap(), w.grad().unwrap());
    assert_eq!(
        (gx.shape(), gx.dtype(), gx.device()),
        ([2, 3].into(), DType::F32, Device::Meta)
    );
    assert_eq!(
        (gw.shape(), gw.dtype(), gw.device()),
        ([3].into(), DType::F64, Device::Meta)
    );
}

// ---------------------------------------------------------------- second order

#[test]
fn second_derivative_of_a_square() {
    let x = leaf(&[1., 2., 3.]);
    let g = grad(&[&(&x * &x).sum()], &[&x], create_graph()).remove(0);
    assert!(g.requires_grad());
    assert_eq!(vals(&g), [2., 4., 6.]);
    let h = grad(&[&g.sum()], &[&x], GradOptions::default()).remove(0);
    assert_eq!(vals(&h), [2., 2., 2.]);
}

#[test]
fn mixed_second_derivatives_of_two_leaves() {
    let x = leaf(&[1., 2.]);
    let w = leaf(&[3., -1.]);
    let gx = grad(&[&(&x * &x * &w).sum()], &[&x], create_graph()).remove(0);
    assert_eq!(vals(&gx), [6., -4.]);
    let h = grad(&[&gx.sum()], &[&x, &w], GradOptions::default());
    assert_eq!(vals(&h[0]), [6., -2.]);
    assert_eq!(vals(&h[1]), [2., 4.]);
}

#[test]
fn second_derivative_through_a_saved_output() {
    let x = leaf(&[0.5, 1.]);
    let g = grad(&[&x.exp().sum()], &[&x], create_graph()).remove(0);
    let e = [0.5f32.exp(), 1f32.exp()];
    assert_close(&vals(&g), &e);
    assert_eq!(g.grad_fn().unwrap().name(), "MulBackward0");
    let h = grad(&[&g.sum()], &[&x], GradOptions::default()).remove(0);
    assert_close(&vals(&h), &e);
}

#[test]
fn second_order_after_dropping_the_forward_handles() {
    let x = leaf(&[1., 2.]);
    let w = leaf(&[3., 4.]);
    let g = {
        let a = &x * &w;
        let y = (&a * &a).sum();
        grad(&[&y], &[&x], create_graph()).remove(0)
    };
    // `a` and `y` are gone; the graph of `g` keeps what it needs: g = 2 x w².
    assert_eq!(vals(&g), [18., 64.]);
    let h = grad(&[&g.sum()], &[&x, &w], GradOptions::default());
    assert_eq!(vals(&h[0]), [18., 32.]);
    assert_eq!(vals(&h[1]), [12., 32.]);
}

#[test]
fn higher_order_graphs_reach_the_original_accumulator() {
    let x = leaf(&[2.]);
    let accumulator = (&x * 1.0).grad_fn().unwrap().next_functions()[0]
        .clone()
        .unwrap()
        .0;
    assert_eq!(accumulator.name(), "AccumulateGrad");
    let g = grad(&[&(&x * &x).sum()], &[&x], create_graph()).remove(0);
    let mut found = 0;
    let mut stack = vec![g.grad_fn().unwrap()];
    while let Some(f) = stack.pop() {
        if f.name() == "AccumulateGrad" {
            assert!(f.is_same(&accumulator));
            found += 1;
        }
        stack.extend(f.next_functions().into_iter().flatten().map(|(f, _)| f));
    }
    assert!(found > 0);
}

/// `f` maps `x` to an output whose gradient with respect to `x` has a graph. Checks that this
/// graph contains only implemented formulas and is linear in the upstream gradient `u`:
/// `d<g, r>/du = J r`, which for the linear operations tested here is `linear(r)`.
fn check_double_backward(
    name: &str,
    x: &Tensor,
    f: impl Fn(&Tensor) -> Tensor,
    linear: impl Fn(&Tensor) -> Tensor,
) {
    let y = f(x);
    let u = Tensor::from_vec(
        (0..y.num_elements()).map(|i| i as f32 - 1.5).collect(),
        y.shape(),
    )
    .to(y.dtype());
    u.requires_grad_(true);
    let options = GradOptions {
        grad_outputs: vec![Some(u.clone())],
        create_graph: true,
        ..Default::default()
    };
    let g = grad(&[&y], &[x], options).remove(0);
    assert!(g.requires_grad(), "{name}");
    // For `add` and the like, the gradient is `u` itself, a leaf.
    if let Some(f) = g.grad_fn() {
        let names = graph_names(&f);
        assert!(!names.contains(&"NotImplemented"), "{name}: {names:?}");
    }
    let r = Tensor::from_vec(
        (0..g.num_elements()).map(|i| 0.5 + i as f32).collect(),
        g.shape(),
    )
    .to(g.dtype());
    let gu = grad(&[&(&g * &r).sum()], &[&u], GradOptions::default()).remove(0);
    let expected = {
        let _no_grad = homura::no_grad();
        linear(&r)
    };
    assert_eq!(gu.shape(), expected.shape(), "{name}");
    assert_close(&vals(&gu.to(DType::F32)), &vals(&expected.to(DType::F32)));
}

#[test]
fn formulas_record_their_own_backward_pass() {
    let x = Tensor::from_slice(&[1f32, -2., 3., 0.5, 4., -1.], [2, 3]);
    x.requires_grad_(true);
    let w = Tensor::from_slice(&[2f32, -1., 0.5], [3]);
    let row = Tensor::from_slice(&[1f32, 2., 3.], [1, 3]);
    row.requires_grad_(true);
    let id = |r: &Tensor| r.clone();
    check_double_backward("add", &x, |x| x + &w, id);
    check_double_backward("add scalar", &x, |x| 2.0 + x, id);
    check_double_backward("sub", &x, |x| x - &w, id);
    check_double_backward("rsub", &x, |x| 1.0 - x, |r| -r);
    let wg = w.deep_clone();
    wg.requires_grad_(true);
    check_double_backward("sub, second input", &wg, |w| &x - w, |r| -r.expand([2, 3]));
    check_double_backward("mul", &x, |x| x * &w, |r| r * &w);
    check_double_backward("mul scalar", &x, |x| x * 3.0, |r| r * 3.0);
    check_double_backward("neg", &x, |x| -x, |r| -r);
    check_double_backward("sum", &x, |x| x.sum(), |r| r.sum());
    check_double_backward("expand", &row, |r| r.expand([4, 3]), |r| r.expand([4, 3]));
    check_double_backward(
        "sum_to_size",
        &x,
        |x| x.sum_to_size([3]),
        |r| r.sum_to_size([3]),
    );
    check_double_backward("deep_clone", &x, |x| x.deep_clone(), id);
    check_double_backward("view", &x, |x| x.view([3, 2]), |r| r.view([3, 2]));
    check_double_backward("reshape", &x, |x| x.reshape([-1]), |r| r.reshape([-1]));
    check_double_backward(
        "reshape that copies",
        &row,
        |r| r.expand([2, 3]).reshape([6]),
        |r| r.expand([2, 3]).reshape([6]),
    );
    check_double_backward("repeat", &row, |r| r.repeat([2, 2]), |r| r.repeat([2, 2]));
    check_double_backward("to", &x, |x| x.to(DType::F64), |r| r.to(DType::F64));
    check_double_backward("flatten", &x, |x| x.flatten(0, 1), |r| r.flatten(0, 1));
    // Nonlinear: the gradient is also differentiable with respect to `x`.
    let x = leaf(&[0.5, -1.]);
    let ex = {
        let _no_grad = homura::no_grad();
        x.exp()
    };
    check_double_backward("exp", &x, |x| x.exp(), |r| r * &ex);
    check_double_backward("mul of self", &x, |x| x * x, |r| r * &x * 2.0);
}

// ---------------------------------------------------------------- release

#[test]
fn saved_tensors_are_freed_by_a_backward_pass() {
    let x = leaf(&[1., 2.]);
    let y = (&x * &x).sum();
    y.backward();
    let message = panic_message(|| y.backward());
    assert_eq!(
        message,
        "Trying to backward through the graph a second time (or directly access saved tensors \
         after they have already been freed). Saved intermediate values of the graph are freed \
         when you call backward() or autograd::grad(). Set retain_graph if you need to backward \
         through the graph a second time or if you need to access saved tensors after calling \
         backward."
    );
    // Retained, the graph runs again.
    let x = leaf(&[1., 2.]);
    let y = (&x * &x).sum();
    let retain = BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    autograd::backward(&[&y], &[], retain);
    y.backward();
    assert_eq!(grad_of(&x), [4., 8.]);
    // `create_graph` retains by default.
    let x = leaf(&[1., 2.]);
    let y = (&x * &x).sum();
    grad(&[&y], &[&x], create_graph());
    assert_eq!(
        vals(&grad(&[&y], &[&x], GradOptions::default())[0]),
        [2., 4.]
    );
}

#[test]
fn nodes_that_saved_nothing_can_run_again() {
    let x = leaf(&[1., 2.]);
    let y = (&x + 1.0).sum();
    y.backward();
    y.backward();
    assert_eq!(grad_of(&x), [2., 2.]);
    let x = leaf(&[1., 2.]);
    let y = (1.0 - &x).sum();
    y.backward();
    y.backward();
    assert_eq!(grad_of(&x), [-2., -2.]);
}

#[test]
fn scalar_factors_are_saved_and_released_like_tensors() {
    // As in PyTorch, which saves the scalar of `x * 2` as a tensor: whatever its value or type,
    // and on either side.
    type Op = Box<dyn Fn(&Tensor) -> Tensor>;
    let products: [Op; 6] = [
        Box::new(|x| x * 2.5),
        Box::new(|x| 2.5 * x),
        Box::new(|x| x * 0.0),
        Box::new(|x| x.mul(1i64)),
        Box::new(|x| x.mul(-3i64)),
        Box::new(|x| x.mul(true)),
    ];
    let devices = [DType::F16, DType::BF16, DType::F32, DType::F64]
        .map(|dtype| (dtype, Device::Cpu))
        .into_iter()
        .chain([(DType::F32, Device::Meta)]);
    for (dtype, device) in devices {
        for (i, product) in products.iter().enumerate() {
            let input = || {
                let x = homura::zeros_with([2], (dtype, device));
                x.requires_grad_(true);
                x
            };
            let x = input();
            let y = product(&x).sum();
            y.backward();
            let message = panic_message(|| y.backward());
            assert!(
                message.starts_with("Trying to backward through the graph a second time"),
                "{dtype} {device:?} {i}: {message}"
            );
            // Retained, explicitly or by `create_graph`, it runs again.
            let x = input();
            let y = product(&x).sum();
            let retain = BackwardOptions {
                retain_graph: Some(true),
                ..Default::default()
            };
            autograd::backward(&[&y], &[], retain);
            y.backward();
            let options = BackwardOptions {
                create_graph: true,
                ..Default::default()
            };
            let x = input();
            let y = product(&x).sum();
            autograd::backward(&[&y], &[], options.clone());
            autograd::backward(&[&y], &[], options);
            assert_eq!(x.grad().unwrap().dtype(), dtype);
        }
    }
    let x = leaf(&[1., 2.]);
    (&x * 2.5).sum().backward();
    assert_eq!(grad_of(&x), [2.5, 2.5]);
}

// ---------------------------------------------------------------- tensor API

#[test]
fn requires_grad_follows_pytorch_rules() {
    let n = homura::tensor([1i64, 2]);
    assert_eq!(
        panic_message(|| {
            n.requires_grad_(true);
        }),
        "only Tensors of floating point dtype can require gradients"
    );
    assert!(!n.requires_grad_(false).requires_grad());
    assert!(
        panic_message(|| {
            homura::tensor([true]).requires_grad_(true);
        })
        .starts_with("only Tensors of floating point dtype")
    );
    let x = leaf(&[1.]);
    let y = &x * 2.0;
    let before = y.grad_fn().unwrap();
    assert!(y.requires_grad_(true).grad_fn().unwrap().is_same(&before));
    assert_eq!(
        panic_message(|| {
            y.requires_grad_(false);
        }),
        "you can only change requires_grad flags of leaf variables. If you want to use a \
         computed variable in a subgraph that doesn't require differentiation use var_no_grad = \
         var.detach()."
    );
    assert!(x.is_leaf() && !y.is_leaf());
    x.requires_grad_(false);
    assert!(!x.requires_grad());
}

#[test]
fn set_grad_checks_dtype_then_device_then_size() {
    let x = leaf(&[1., 2., 3.]);
    let message = |g: Tensor| panic_message(|| x.set_grad(Some(g)));
    assert_eq!(
        message(homura::zeros_with([2], (DType::F64, Device::Meta))),
        "attempting to assign a gradient with dtype 'f64' to a tensor with dtype 'f32'. Please \
         ensure that the gradient and the tensor have the same dtype"
    );
    assert_eq!(
        message(homura::zeros_with([2], Device::Meta)),
        "attempting to assign a gradient with device type 'meta' to a tensor with device type \
         'cpu'. Please ensure that the gradient and the tensor are on the same device"
    );
    assert_eq!(
        message(homura::zeros([2])),
        "attempting to assign a gradient of size '[2]' to a tensor of size '[3]'. Please ensure \
         that the gradient and the tensor are the same size"
    );
    // Stored as given, even if not contiguous; on any tensor.
    let g = homura::zeros([3, 2]).select(1, 0);
    x.set_grad(Some(g.clone()));
    assert!(x.grad().unwrap().is_same(&g));
    let y = &x * 2.0;
    y.set_grad(Some(homura::ones([3])));
    let n = homura::tensor([1i64]);
    n.set_grad(Some(homura::tensor([2i64])));
    x.set_grad(None);
    assert!(x.grad().is_none());
}

#[test]
fn detach_shares_data_without_history() {
    let x = leaf(&[1., 2.]);
    let y = &x * 2.0;
    let d = y.detach();
    assert!(!d.requires_grad() && d.is_leaf() && d.grad_fn().is_none());
    // Not a view: it can be modified in place in grad mode, and the change is visible in `y`.
    d.add_(1.0);
    assert_eq!(vals(&y), [3., 5.]);
}

#[test]
fn detach_in_place_cuts_the_history() {
    let x = leaf(&[1., 2.]);
    let y = &x * 2.0;
    assert!(y.detach_().is_leaf());
    assert!(!y.requires_grad() && y.grad_fn().is_none());
    // A leaf keeps its gradient.
    (&x * 3.0).sum().backward();
    x.detach_();
    assert!(!x.requires_grad());
    assert_eq!(grad_of(&x), [3., 3.]);
    for v in [t(&[1.]).view([1]), homura::tensor([1i64]).view([1])] {
        assert_eq!(
            panic_message(|| {
                v.detach_();
            }),
            "Can't detach views in-place. Use detach() instead."
        );
    }
}

#[test]
fn node_names_and_edges_follow_pytorch() {
    let x = leaf(&[1., 2.]);
    let y = leaf(&[3., 4.]);
    let name = |t: &Tensor| t.grad_fn().unwrap().name();
    let acc = Some("AccumulateGrad");
    for (case, out, edges) in [
        ("x + 1", &x + 1.0, vec![acc, None]),
        ("1 + x", 1.0 + &x, vec![acc, None]),
        ("x * 2", &x * 2.0, vec![acc, None]),
        ("2 * x", 2.0 * &x, vec![acc, None]),
        ("x - y", &x - &y, vec![acc, acc]),
        ("1 - x", 1.0 - &x, vec![acc]),
    ] {
        assert_eq!(next_names(&out.grad_fn().unwrap()), edges, "{case}");
    }
    assert_eq!(name(&(&x + 1.0)), "AddBackward0");
    assert_eq!(name(&(&x - 1.0)), "SubBackward0");
    assert_eq!(name(&(1.0 - &x)), "RsubBackward1");
    assert_eq!(name(&(&x * &y)), "MulBackward0");
    assert_eq!(name(&-&x), "NegBackward0");
    assert_eq!(name(&x.exp()), "ExpBackward0");
    assert_eq!(name(&x.sum()), "SumBackward0");
    assert_eq!(name(&x.expand([3, 2])), "ExpandBackward0");
    assert_eq!(name(&x.view([2, 1])), "ViewBackward0");
    assert_eq!(name(&x.reshape([1, 2])), "ViewBackward0");
    assert_eq!(name(&x.deep_clone()), "CloneBackward0");
    assert_eq!(name(&x.to(DType::F64)), "ToCopyBackward0");
    assert_eq!(name(&x.to(Device::Meta)), "ToCopyBackward0");
    let copied = x.expand([2, 2]).reshape([4]);
    assert_eq!(
        graph_names(&copied.grad_fn().unwrap()),
        [
            "UnsafeViewBackward0",
            "CloneBackward0",
            "ExpandBackward0",
            "AccumulateGrad"
        ]
    );
    let m = x.expand([3, 2]);
    assert_eq!(name(&m.sum_to_size([1, 2])), "SumToSizeBackward0");
    assert!(m.sum_to_size([3, 2]).is_same(&m));
    assert_eq!(
        format!("{:?}", (&x * &y).grad_fn().unwrap()),
        "<MulBackward0>"
    );
}

#[test]
fn reshapes_that_copy_and_repeat_pass_gradients_back() {
    let x = Tensor::from_slice(&[1f32, 2., 3.], [1, 3]);
    x.requires_grad_(true);
    let y = x.expand([2, 3]).reshape([6]);
    assert!(y.grad_fn().is_some());
    y.backward_with(&t(&[1., 2., 3., 4., 5., 6.]));
    assert_eq!(grad_of(&x), [5., 7., 9.]);
    let x = leaf(&[1., 2.]);
    let r = x.repeat([3]);
    r.backward_with(&t(&[1., 2., 3., 4., 5., 6.]));
    assert_eq!(grad_of(&x), [9., 12.]);
    let x = Tensor::from_slice(&[1f32, 2.], [1, 2]);
    x.requires_grad_(true);
    x.repeat([2, 2]).sum().backward();
    assert_eq!(grad_of(&x), [4., 4.]);
}

// ---------------------------------------------------------------- views

#[test]
fn a_view_with_history_keeps_requiring_grad() {
    // R12, first example.
    let b = leaf(&[2.]);
    let v = b.view([1]);
    b.requires_grad_(false);
    assert!(v.requires_grad() && !v.is_leaf() && v.grad_fn().is_some());
}

#[test]
fn a_view_made_a_leaf_accumulates_into_itself() {
    // R12, second example.
    let b = t(&[2.]);
    let v = b.view([1]);
    v.requires_grad_(true);
    let w = v.view([1]);
    assert_eq!(next_names(&w.grad_fn().unwrap()), [Some("AccumulateGrad")]);
    (&w * 2.0).sum().backward();
    assert!(b.grad().is_none());
    assert_eq!(grad_of(&v), [2.]);
}

#[test]
fn views_without_history_follow_their_base_but_drop_gradients() {
    let z = leaf(&[1., 2., 3.]);
    let v = {
        let _no_grad = homura::no_grad();
        z.view([3])
    };
    assert!(v.requires_grad() && v.is_leaf() && v.grad_fn().is_none());
    let y = &v * 2.0;
    assert!(y.requires_grad());
    assert_eq!(next_names(&y.grad_fn().unwrap()), [None, None]);
    y.sum().backward();
    assert!(z.grad().is_none() && v.grad().is_none());
    assert_eq!(
        panic_message(|| v.backward_with(&homura::ones([3]))),
        "element 0 of tensors does not require grad and does not have a grad_fn"
    );
    assert_eq!(
        panic_message(|| {
            autograd::grad(&[&(&v * 2.0).sum()], &[&v], GradOptions::default());
        }),
        "The differentiated Tensor at index 0 appears to not have been used in the graph. Set \
         allow_unused if this is the desired behavior."
    );
    // Created before its base required grad.
    let b = homura::ones([2]);
    let early = b.view([2]);
    b.requires_grad_(true);
    assert!(early.requires_grad() && early.is_leaf() && early.grad_fn().is_none());
}

#[test]
fn view_properties_after_the_base_changes() {
    for detach in [false, true] {
        let b = leaf(&[1., 2.]);
        let quiet = {
            let _no_grad = homura::no_grad();
            b.view([2])
        };
        let recorded = b.view([2]);
        if detach {
            b.detach_();
        } else {
            b.requires_grad_(false);
        }
        assert!(!quiet.requires_grad(), "detach {detach}");
        assert!(
            recorded.requires_grad() && !recorded.is_leaf(),
            "detach {detach}"
        );
    }
}

#[test]
fn chunks_have_independent_autograd_state() {
    // R23: also the empty chunks of an empty dimension.
    for x in [homura::zeros([4, 3]), homura::zeros([0, 3])] {
        let pieces = {
            let _no_grad = homura::no_grad();
            x.chunk(3, 0)
        };
        let n = pieces.len();
        assert!(n >= 2);
        pieces[0].requires_grad_(true);
        assert!(pieces[1..].iter().all(|p| !p.requires_grad()));
        let g = homura::ones(pieces[1].shape());
        pieces[1].set_grad(Some(g));
        assert!(pieces[0].grad().is_none());
        assert!(pieces[2..].iter().all(|p| p.grad().is_none()));
    }
}

// ---------------------------------------------------------------- not implemented yet

#[test]
fn operations_without_formulas_fail_when_the_backward_pass_reaches_them() {
    let x = Tensor::from_slice(&[0.5f32, 1., 2., 3.], [2, 2]);
    x.requires_grad_(true);
    let w = Tensor::from_slice(&[1f32, 2., 3., 4.], [2, 2]);
    let index = homura::tensor([1i64, 0]);
    let cases: Vec<(&str, Tensor)> = vec![
        ("matmul", x.matmul(&w)),
        ("softmax", x.softmax(-1)),
        ("addmm", x.addmm(&w, &w, 0.0, 1.0)),
    ];
    for (op, out) in cases {
        assert!(out.requires_grad(), "{op}");
        let message = panic_message(|| out.sum().backward());
        assert_eq!(
            message,
            format!("the derivative for '{op}' is not implemented."),
            "{op}"
        );
    }
    assert_eq!(x.softmax(-1).grad_fn().unwrap().name(), "NotImplemented");
    // The integer-indexed result is reshaped by a recorded `unsafe_view` first.
    assert_eq!(
        x.index((&index,)).grad_fn().unwrap().name(),
        "UnsafeViewBackward0"
    );
}

#[test]
fn non_differentiable_operations_record_nothing() {
    let x = leaf(&[1., -2.]);
    for (case, out) in [
        ("greater", x.greater(0.0)),
        ("equal", x.equal(&x)),
        ("logical_and", x.logical_and(&x)),
        ("logical_not", x.logical_not()),
        ("argmax", x.argmax()),
        ("all", x.all()),
        ("to integer", x.to(DType::I64)),
        ("detach", x.detach()),
        ("zeros_like", homura::zeros_like(&x)),
        ("rand_like", homura::rand_like(&x)),
    ] {
        assert!(!out.requires_grad() && out.grad_fn().is_none(), "{case}");
    }
}

// ---------------------------------------------------------------- in-place operations

/// Checks that `op` panics with `message` and changes neither the data, the version (seen
/// through a saved copy) nor the history of `target`.
fn assert_rejected(target: &Tensor, message: &str, op: impl FnOnce()) {
    let data = vals(target);
    let grad_fn = target.grad_fn();
    // A graph that saved the data of `target`: a version change would make it fail.
    let probe = target.detach();
    let saved = leaf(&[1.]);
    let check = (&saved * &probe).sum();
    assert_eq!(panic_message(op), message);
    assert_eq!(vals(target), data);
    match (grad_fn, target.grad_fn()) {
        (None, None) => {}
        (Some(a), Some(b)) => assert!(a.is_same(&b)),
        _ => panic!("the history changed"),
    }
    check.backward();
}

const LEAF: &str = "a leaf Variable that requires grad is being used in an in-place operation.";
const RECORDED: &str = "in-place operations that need to be recorded for autograd are not \
                        supported yet: the tensor written to or one of the inputs requires grad; \
                        use the out-of-place operation";
const VIEW: &str = "in-place operations on a view are not supported yet when the view or one of \
                    the inputs requires grad; use the out-of-place operation, or deep_clone() the \
                    view first";

#[test]
fn in_place_operations_on_leaves_that_require_grad_are_rejected() {
    let x = leaf(&[0.5, 0.25]);
    let o = t(&[1., 1.]);
    type InPlace = Box<dyn Fn(&Tensor)>;
    let ones = || t(&[1., 1.]);
    let ops: Vec<(&str, InPlace)> = vec![
        ("add_", Box::new(|x| _ = x.add_(1.0))),
        ("sub_", Box::new(move |x| _ = x.sub_(ones()))),
        ("mul_", Box::new(|x| _ = x.mul_(2.0))),
        ("div_", Box::new(|x| _ = x.div_(2.0))),
        ("pow_", Box::new(|x| _ = x.pow_(2.0))),
        ("clamp_", Box::new(|x| _ = x.clamp_(0.0, 1.0))),
        (
            "masked_fill_",
            Box::new(|x| _ = x.masked_fill_(&homura::tensor([true, false]), 0.0)),
        ),
        ("copy_", Box::new(|x| _ = x.copy_(&t(&[3., 4.])))),
        ("fill_", Box::new(|x| _ = x.fill_(1.0))),
        ("zero_", Box::new(|x| _ = x.zero_())),
        ("uniform_", Box::new(|x| _ = x.uniform_(0.0, 1.0))),
        ("normal_", Box::new(|x| _ = x.normal_(0.0, 1.0))),
        ("lerp_", Box::new(move |x| _ = x.lerp_(&ones(), 0.5))),
        (
            "addcmul_",
            Box::new(move |x| _ = x.addcmul_(&ones(), &ones(), 1.0)),
        ),
        (
            "addcdiv_",
            Box::new(move |x| _ = x.addcdiv_(&ones(), &ones(), 1.0)),
        ),
        (
            "+=",
            Box::new(|x| {
                let mut y = x.clone();
                y += 1.0;
            }),
        ),
    ];
    for (_, op) in &ops {
        assert_rejected(&x, LEAF, || op(&x));
    }
    // Writing into a tensor that doesn't require grad from one that does is recorded in
    // PyTorch; here it is rejected until in-place operations record.
    for (_, op) in &ops[..1] {
        let y = &x * 2.0;
        assert_rejected(&y, RECORDED, || op(&y));
    }
    assert_rejected(&o, RECORDED, || {
        o.add_(&x);
    });
    assert_rejected(&o, RECORDED, || {
        o.copy_(&x);
    });
    assert_rejected(&o, RECORDED, || {
        o.masked_fill_(&homura::tensor([true, false]), x.sum());
    });
    assert_rejected(&o, RECORDED, || {
        o.clamp_(&x, 2.0);
    });
}

#[test]
fn in_place_operations_on_views_are_rejected_when_they_need_recording() {
    let x = leaf(&[1., 2.]);
    let v = x.view([2]);
    assert_rejected(&v, VIEW, || {
        v.add_(1.0);
    });
    let quiet = {
        let _no_grad = homura::no_grad();
        x.view([2])
    };
    assert_rejected(&quiet, VIEW, || {
        quiet.mul_(2.0);
    });
    let plain = homura::zeros([2]);
    let pv = plain.view([2]);
    assert_rejected(&pv, VIEW, || {
        pv.copy_(&x);
    });
    // Nothing requires grad: allowed.
    pv.add_(1.0);
    assert_eq!(vals(&plain), [1., 1.]);
}

#[test]
fn in_place_operations_run_in_no_grad_mode_and_are_detected_later() {
    let x = leaf(&[1., 2.]);
    let y = (&x * &x).sum();
    {
        let _no_grad = homura::no_grad();
        x.sub_(0.5);
        x.view([2]).mul_(2.0);
    }
    assert_eq!(vals(&x), [1., 3.]);
    assert!(x.requires_grad() && x.is_leaf());
    assert_eq!(
        panic_message(|| y.backward()),
        "one of the variables needed for gradient computation has been modified by an inplace \
         operation: [f32 [2]] is at version 2; expected version 0 instead."
    );
    // Also through `detach`, in grad mode.
    let x = leaf(&[1., 2.]);
    let y = (&x * &x).sum();
    x.detach().add_(1.0);
    assert_eq!(
        panic_message(|| y.backward()),
        "one of the variables needed for gradient computation has been modified by an inplace \
         operation: [f32 [2]] is at version 1; expected version 0 instead."
    );
}

#[test]
fn in_place_operations_into_integer_tensors_are_not_recorded() {
    let x = leaf(&[1.5, 2.5]);
    let n = homura::zeros_with([2], DType::I64);
    n.copy_(&x);
    assert_eq!(n.to_vec::<i64>(), [1, 2]);
    assert!(!n.requires_grad());
    assert_eq!(
        panic_message(|| _ = n.add_(&x)),
        "result type f32 can't be cast to the desired output type i64"
    );
}

// ---------------------------------------------------------------- saved tensors (R16–R18, R20)

#[test]
fn a_leaf_that_stops_requiring_grad_gets_no_gradient() {
    // R16.
    let x = leaf(&[2.]);
    let y = (&x * &x).sum();
    x.requires_grad_(false);
    y.backward();
    assert!(x.grad().is_none());
}

#[test]
fn a_saved_leaf_that_starts_requiring_grad_is_differentiated() {
    // R17.
    let x = t(&[2.]);
    let z = leaf(&[3.]);
    let y = (&x * &z).sum();
    x.requires_grad_(true);
    let g = grad(&[&y], &[&z], create_graph()).remove(0);
    assert!(g.requires_grad());
    assert_eq!(
        vals(&grad(&[&g.sum()], &[&x], GradOptions::default())[0]),
        [1.]
    );
}

#[test]
fn saved_factors_are_checked_even_when_their_gradient_is_not_needed() {
    // R25: `grad` with respect to `x` doesn't need the saved `x`, but a pass through the node
    // checks every tensor it saved, as in PyTorch.
    let x = leaf(&[2., 3.]);
    let w = leaf(&[4., 5.]);
    let y = (&x * &w).sum();
    {
        let _no_grad = homura::no_grad();
        x.add_(100.0);
    }
    assert_eq!(
        panic_message(|| {
            autograd::grad(&[&y], &[&x], GradOptions::default());
        }),
        "one of the variables needed for gradient computation has been modified by an inplace \
         operation: [f32 [2]] is at version 1; expected version 0 instead."
    );
    let x = leaf(&[2., 3.]);
    let a = &x * 2.0;
    let w = leaf(&[4., 5.]);
    let y = (&a * &w).sum();
    a.detach_();
    assert!(
        panic_message(|| {
            autograd::grad(&[&y], &[&x], GradOptions::default());
        })
        .starts_with("Trying to use a saved tensor that has been detached in-place")
    );
    // Unmodified, or only dropped, the saved factors are fine.
    let x = leaf(&[2., 3.]);
    let a = &x * 2.0;
    let w = leaf(&[4., 5.]);
    let y = (&a * &w).sum();
    drop(a);
    assert_eq!(
        vals(&grad(&[&y], &[&x], GradOptions::default())[0]),
        [8., 10.]
    );
    // A node that only has its input captured doesn't run, so it checks nothing.
    let x = leaf(&[2., 3.]);
    let w = leaf(&[4., 5.]);
    let a = &x * &w;
    let y = (&a * 1.0).sum();
    {
        let _no_grad = homura::no_grad();
        x.add_(1.0);
    }
    assert_eq!(
        vals(&grad(&[&y], &[&a], GradOptions::default())[0]),
        [1., 1.]
    );
}

#[test]
fn a_saved_non_leaf_detached_in_place_can_not_be_used() {
    // R18.
    const DETACHED: &str = "Trying to use a saved tensor that has been detached in-place, i.e. \
                            with .detach_().This is not supported, please use out-of-place \
                            `.detach()` instead";
    let x = leaf(&[2.]);
    let a = &x * 2.0;
    let y = (&a * &a).sum();
    a.detach_();
    assert_eq!(panic_message(|| y.backward()), DETACHED);
    // Only dropping it is fine.
    let x = leaf(&[2.]);
    let a = &x * 2.0;
    let y = (&a * &a).sum();
    drop(a);
    y.backward();
    assert_eq!(grad_of(&x), [16.]);
    // R20: saved again by a higher-order graph, which must still see the `detach_`.
    for detach in [true, false] {
        let x = leaf(&[2.]);
        let a = &x * &x;
        let b = leaf(&[3.]);
        let u = Tensor::from_slice(&[1f32], []);
        u.requires_grad_(true);
        let options = GradOptions {
            grad_outputs: vec![Some(u.clone())],
            create_graph: true,
            ..Default::default()
        };
        let g = grad(&[&(&a * &b).sum()], &[&b], options).remove(0);
        if detach {
            a.detach_();
            assert_eq!(
                panic_message(|| {
                    autograd::grad(&[&g], &[&u], GradOptions::default());
                }),
                DETACHED
            );
        } else {
            drop(a);
            assert_eq!(vals(&grad(&[&g], &[&u], GradOptions::default())[0]), [4.]);
        }
    }
}

// ---------------------------------------------------------------- errors

#[test]
fn backward_and_grad_report_pytorch_errors() {
    let x = leaf(&[1., 2.]);
    let plain = t(&[1.]);
    assert_eq!(
        panic_message(|| plain.sum().backward()),
        "element 0 of tensors does not require grad and does not have a grad_fn"
    );
    assert_eq!(
        panic_message(|| {
            autograd::backward(&[&x.sum(), &plain], &[], BackwardOptions::default())
        }),
        "element 1 of tensors does not require grad and does not have a grad_fn"
    );
    let one = Tensor::from_slice(&[1f32], []);
    assert_eq!(
        panic_message(|| {
            autograd::backward(
                &[&x.sum(), &x.sum()],
                &[Some(&one)],
                BackwardOptions::default(),
            )
        }),
        "got 2 tensors and 1 gradients"
    );
    // The given pairs are checked first, as in PyTorch.
    assert_eq!(
        panic_message(|| {
            autograd::backward(
                &[&x.sum(), &x.sum()],
                &[Some(&t(&[1.]))],
                BackwardOptions::default(),
            )
        }),
        "Mismatch in shape: grad_output[0] has a shape of [1] and output[0] has a shape of []."
    );
    assert_eq!(
        panic_message(|| (&x * 2.0).backward_with(&t(&[1., 2., 3.]))),
        "Mismatch in shape: grad_output[0] has a shape of [3] and output[0] has a shape of [2]."
    );
    assert_eq!(
        panic_message(|| {
            autograd::grad(&[&x.sum()], &[], GradOptions::default());
        }),
        "`inputs` argument to `grad()` cannot be empty."
    );
    assert_eq!(
        panic_message(|| {
            autograd::grad(&[&x.sum()], &[&plain], GradOptions::default());
        }),
        "One of the differentiated Tensors does not require grad"
    );
    let w = leaf(&[3.]);
    assert_eq!(
        panic_message(|| {
            autograd::grad(&[&x.sum()], &[&x, &w], GradOptions::default());
        }),
        "The differentiated Tensor at index 1 appears to not have been used in the graph. Set \
         allow_unused if this is the desired behavior."
    );
    let unused = GradOptions {
        allow_unused: true,
        ..Default::default()
    };
    let g = autograd::grad(&[&x.sum()], &[&x, &w], unused);
    assert!(g[0].is_some() && g[1].is_none());
    // No outputs: nothing is used.
    assert!(
        panic_message(|| {
            autograd::grad(&[], &[&x], GradOptions::default());
        })
        .starts_with("The differentiated Tensor at index 0 appears to not have been used")
    );
}

#[test]
fn backward_through_meta_to_cpu_fails_like_forward_to() {
    let x = leaf(&[1., 2.]);
    let m = x.to(Device::Meta);
    assert!(m.requires_grad());
    assert_eq!(
        panic_message(|| m.sum().backward()),
        "Cannot copy out of meta tensor; no data!"
    );
}
