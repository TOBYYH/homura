//! `gradcheck` and `gradgradcheck`: their contract (variables, restoring perturbed elements,
//! reporting mismatches), and both checks for every backward formula implemented so far, with
//! broadcasting, negative dimensions, non-contiguous and zero-dimensional inputs, plus
//! separate shape and zero-gradient assertions for empty inputs. Run in debug and release
//! profiles.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::{DType, Device, Tensor};

/// A variable of `shape` with deterministic values of both signs, between 0.3 and 1.3 in
/// magnitude, so that no element sits at a kink such as `sign`'s.
fn var(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let magnitude = 0.3 + ((i * 7 + seed * 3) % 11) as f64 / 10.0;
            if (i + seed).is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect();
    let x = Tensor::from_vec(values, shape);
    x.requires_grad_(true);
    x
}

/// A variable of shape `[rows, cols]` whose layout is transposed, so not contiguous.
fn transposed(rows: usize, cols: usize, seed: usize) -> Tensor {
    let base = var(&[cols, rows], seed).detach();
    let x = base.transpose(0, 1);
    x.requires_grad_(true);
    x
}

fn ints(values: &[i64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

/// Runs both checks of `f` with the default options.
fn check(name: &str, inputs: &[&Tensor], f: impl Fn(&[Tensor]) -> Vec<Tensor>) {
    let options = GradcheckOptions::default();
    if let Err(e) = gradcheck(&f, inputs, options.clone()) {
        panic!("{name}: {e}");
    }
    if let Err(e) = gradgradcheck(&f, inputs, options) {
        panic!("{name}, second order: {e}");
    }
}

// ---------------------------------------------------------------- contract

#[test]
fn variables_must_be_distinct_f64_cpu_tensors() {
    let f = |x: &[Tensor]| vec![x[0].exp()];
    let options = GradcheckOptions::default;
    let constant = Tensor::from_slice(&[1f64, 2.], [2]);
    assert_eq!(
        panic_message(|| drop(gradcheck(f, &[&constant], options()))),
        "gradcheck expects at least one input tensor to require gradient, but none of them have \
         requires_grad=True."
    );
    let single = homura::ones([2]);
    single.requires_grad_(true);
    assert_eq!(
        panic_message(|| drop(gradcheck(f, &[&single], options()))),
        "gradcheck: input 0 requires grad and is a f32 tensor on Cpu; numerical derivatives need \
         F64 tensors on the CPU"
    );
    let meta = homura::ones_with([2], (DType::F64, Device::Meta));
    meta.requires_grad_(true);
    assert!(
        panic_message(|| drop(gradgradcheck(f, &[&meta], options())))
            .starts_with("gradcheck: input 0 requires grad and is a f64 tensor on Meta")
    );
    // Elements that share memory would be perturbed together.
    let expanded = {
        let row = Tensor::from_slice(&[1f64, 2.], [2]);
        let e = row.expand([3, 2]);
        e.requires_grad_(true);
        e
    };
    assert_eq!(
        panic_message(|| drop(gradcheck(f, &[&expanded], options()))),
        "gradcheck: the elements of input 0 may overlap in memory, so perturbing one would \
         change others; pass a deep_clone() of it"
    );
    let x = var(&[2], 0);
    let g = |x: &[Tensor]| vec![&x[0] * &x[1]];
    assert_eq!(
        panic_message(|| drop(gradcheck(g, &[&x, &x], options()))),
        "gradcheck: inputs 0 and 1 share storage, so perturbing one would change the other; \
         pass a deep_clone() of one of them"
    );
    let index = ints(&[1, 0]);
    let h = |x: &[Tensor]| vec![x[0].index_select(0, &x[1])];
    assert!(gradcheck(h, &[&x, &index], options()).is_ok());
}

#[test]
fn perturbed_elements_are_restored_even_when_the_function_panics() {
    let x = transposed(2, 3, 1);
    let before = x.to_vec::<f64>();
    gradcheck(|x| vec![x[0].exp()], &[&x], GradcheckOptions::default()).unwrap();
    assert_eq!(x.to_vec::<f64>(), before);
    // The function fails while an element is perturbed.
    let calls = std::cell::Cell::new(0);
    let failing = |x: &[Tensor]| {
        calls.set(calls.get() + 1);
        if calls.get() == 3 {
            panic!("failed");
        }
        vec![x[0].exp()]
    };
    assert_eq!(
        panic_message(|| drop(gradcheck(failing, &[&x], GradcheckOptions::default()))),
        "failed"
    );
    assert_eq!(x.to_vec::<f64>(), before);
}

#[test]
fn mismatches_are_reported_with_their_positions() {
    let x = var(&[2, 2], 0);
    // The second factor is a constant to the backward pass, so the gradient misses `x`.
    let message = gradcheck(
        |x| vec![x[0].exp(), &x[0] * &x[0].detach()],
        &[&x],
        GradcheckOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        message.starts_with(
            "Jacobian mismatch for output 1 with respect to input 0: the derivative of output \
             element [0, 0] with respect to input element [0, 0] is -0.3 analytically but "
        ),
        "{message}"
    );
    let numerical: f64 = message
        .split_once(" analytically but ")
        .and_then(|(_, rest)| rest.split_once(" numerically"))
        .map(|(n, _)| n.parse().unwrap())
        .unwrap();
    assert!((numerical + 0.6).abs() < 1e-6, "{message}");
    // An output without a graph has zero analytical derivatives, and those are compared.
    let message = gradcheck(
        |x| vec![x[0].detach() * 2.0],
        &[&x],
        GradcheckOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        message
            .ends_with("the output doesn't require grad, so its analytical derivatives are zero"),
        "{message}"
    );
    // Integer outputs are not differentiable and are ignored; constant outputs pass.
    let constant = homura::zeros_with([2], DType::F64);
    check("integer output", &[&x], |x| {
        vec![x[0].greater(0.0), x[0].exp(), constant.clone()]
    });
    // Outputs must keep their number, shapes and dtypes under perturbation.
    let at_threshold = Tensor::from_slice(&[0.5f64], [1]);
    at_threshold.requires_grad_(true);
    let message = gradcheck(
        |x| {
            let y = x[0].exp();
            if x[0].item::<f64>() > 0.5 {
                vec![y.clone(), y]
            } else {
                vec![y]
            }
        },
        &[&at_threshold],
        GradcheckOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert_eq!(
        message,
        "the outputs changed when an input was perturbed: [f64 [1]] became [f64 [1], f64 [1]]"
    );
    // Tolerances are options.
    let loose = GradcheckOptions {
        rtol: 1.0,
        ..Default::default()
    };
    assert!(gradcheck(|x| vec![&x[0] * &x[0].detach()], &[&x], loose).is_ok());
}

#[test]
fn second_order_checks_differentiate_with_respect_to_the_upstream_gradient() {
    // The first derivative of `2x` is a constant with respect to `x`, but `2g` with respect to
    // the upstream gradient `g`.
    let x = var(&[3], 0);
    check("2x", &[&x], |x| vec![&x[0] * 2.0]);
    check("x^3", &[&x], |x| vec![&x[0] * &x[0] * &x[0]]);
    // Several outputs, one of which is a constant.
    let constant = Tensor::from_slice(&[1f64], [1]);
    check("outputs", &[&x], |x| vec![constant.clone(), x[0].exp()]);
    // `sign` has a zero gradient that doesn't depend on `g` at all.
    check("sign", &[&x], |x| vec![x[0].sign()]);
}

// ---------------------------------------------------------------- formulas

#[test]
fn elementwise_formulas() {
    let x = var(&[2, 3], 0);
    let t = transposed(2, 3, 1);
    let s = var(&[], 2);
    for (name, x) in [
        ("contiguous", &x),
        ("transposed", &t),
        ("zero-dimensional", &s),
    ] {
        check(&format!("neg {name}"), &[x], |x| vec![x[0].neg()]);
        check(&format!("exp {name}"), &[x], |x| vec![x[0].exp()]);
        check(&format!("erf {name}"), &[x], |x| vec![x[0].erf()]);
        check(&format!("sign {name}"), &[x], |x| vec![x[0].sign()]);
    }
    let b = var(&[3], 3);
    let c = var(&[2, 1], 4);
    check("add", &[&x, &b], |x| vec![&x[0] + &x[1]]);
    check("add broadcast both", &[&b, &c], |x| vec![&x[0] + &x[1]]);
    check("add scalar", &[&t], |x| vec![&x[0] + 2.5, 1.5 + &x[0]]);
    check("sub", &[&x, &c], |x| vec![&x[0] - &x[1], &x[1] - &x[0]]);
    check("rsub", &[&t], |x| vec![2.0 - &x[0]]);
    check("mul", &[&x, &b], |x| vec![&x[0] * &x[1]]);
    check("mul self", &[&t], |x| vec![&x[0] * &x[0]]);
    check("mul scalar", &[&x], |x| vec![&x[0] * 1.5, -0.5 * &x[0]]);
    check("mul zero-dimensional", &[&s, &b], |x| vec![&x[0] * &x[1]]);
    check("mul integer", &[&x], |x| vec![x[0].mul(ints(&[2, -1, 3]))]);
}

#[test]
fn reduction_and_view_formulas() {
    let x = var(&[2, 3], 0);
    let t = transposed(2, 3, 1);
    let v = var(&[3], 2);
    check("sum", &[&t], |x| vec![x[0].sum()]);
    check("sum_to_size", &[&x], |x| {
        vec![
            x[0].sum_to_size([1, 3]),
            x[0].sum_to_size([3]),
            x[0].sum_to_size([2, 1]),
        ]
    });
    check("expand", &[&v], |x| {
        vec![x[0].expand([2, 3]), x[0].expand([2, -1])]
    });
    check("view", &[&x], |x| vec![x[0].view([3, 2]), x[0].view([-1])]);
    check("reshape", &[&x, &t], |x| {
        vec![x[0].reshape([3, 2]), x[1].reshape([6]), x[1].flatten(0, -1)]
    });
    check("contiguous", &[&t], |x| vec![x[0].contiguous()]);
    check("deep_clone", &[&t], |x| vec![x[0].deep_clone()]);
    check("to the same dtype", &[&x], |x| vec![x[0].to(DType::F64)]);
    check("select", &[&t], |x| {
        vec![x[0].select(-1, -1), x[0].select(0, 1)]
    });
    check("narrow", &[&x], |x| {
        vec![x[0].narrow(-1, 1, 2), x[0].narrow(0, 1, 1)]
    });
    check("slices", &[&t], |x| {
        vec![
            x[0].index((.., 1..)),
            x[0].index((.., (0..).step_by(2))),
            x[0].index((1, ..2)),
        ]
    });
    check("split", &[&x], |x| x[0].split(2, -1));
    check("chunk", &[&x], |x| x[0].chunk(2, 0));
    check("repeat", &[&v], |x| vec![x[0].repeat([2, 2])]);
    let w = var(&[6], 3);
    check("as_strided", &[&x, &w], |x| {
        vec![
            x[0].as_strided([2, 2], [1, 2], 1),
            // Output elements that overlap, and a strided input. (Storage elements outside the
            // input view would get no gradient, as in PyTorch, so a numerical check can't
            // cover them.)
            x[1].as_strided([4, 2], [1, 1], 0),
            x[1].index((0..).step_by(2)).as_strided([2], [2], 2),
        ]
    });
}

#[test]
fn index_formulas() {
    let x = var(&[3, 2], 0);
    let t = transposed(3, 2, 1);
    check("index_select", &[&t], |x| {
        vec![
            x[0].index_select(0, &ints(&[2, 0, 2])),
            x[0].index_select(-1, &ints(&[1])),
        ]
    });
    check("integer index", &[&t], |x| {
        vec![
            x[0].index(ints(&[-1, 0, 0])),
            x[0].index(homura::tensor([[0i64, 2], [1, 1]])),
            x[0].index((.., &ints(&[1, 1, 0]))),
        ]
    });
    let source = var(&[4, 2], 2);
    check("index_add", &[&x, &source], |x| {
        vec![x[0].index_add(0, &ints(&[2, 0, 2, 1]), &x[1])]
    });
    let index = homura::tensor([[1i64, 1], [0, 1], [0, 0]]);
    check("gather", &[&t], |x| {
        vec![x[0].gather(1, &index), x[0].gather(0, &index)]
    });
    let src = var(&[3, 2], 3);
    check("scatter_add", &[&x, &src], |x| {
        vec![
            x[0].scatter_add(1, &index, &x[1]),
            x[0].scatter_add(0, &index, &x[1]),
        ]
    });
    let row = var(&[3], 4);
    check("select_scatter", &[&x, &row], |x| {
        vec![x[0].select_scatter(&x[1], -1, -1)]
    });
    let column = var(&[3, 1], 5);
    check("slice_scatter", &[&t, &column], |x| {
        vec![
            x[0].slice_scatter(&x[1], 1, Some(1), None, 1),
            x[0].slice_scatter(&x[1], -1, None, None, 2),
        ]
    });
    // Zero-dimensional sources and indices.
    let s = var(&[], 6);
    let v = var(&[3], 7);
    check("zero-dimensional scatter_add", &[&v, &s], |x| {
        vec![x[0].scatter_add(0, &homura::tensor(1i64), &x[1])]
    });
    check("zero-dimensional gather", &[&s], |x| {
        vec![x[0].gather(0, &ints(&[0, 0]))]
    });
}

// ---------------------------------------------------------------- empty inputs

#[test]
fn empty_inputs_get_zero_gradients_of_their_shape() {
    // Their Jacobians have no entries, so the checks above prove nothing about them. Each
    // operation reaches an empty result from the empty input, `x`, or both.
    let empty = var(&[0, 3], 0);
    let x = var(&[2, 3], 1);
    let none = Tensor::from_slice::<i64>(&[], [0]);
    type Op = (&'static str, fn(&Tensor, &Tensor, &Tensor) -> Tensor);
    let ops: [Op; 10] = [
        ("exp", |e, _, _| e.exp()),
        ("mul", |e, _, _| e * &e.sign()),
        ("sum", |e, _, _| e.sum()),
        ("sum_to_size", |e, _, _| e.sum_to_size([1, 3])),
        ("expand", |e, _, _| e.expand([2, 0, 3])),
        ("reshape", |e, _, _| e.reshape([3, 0])),
        ("narrow", |_, x, _| x.narrow(0, 1, 0)),
        ("index_select", |_, x, i| x.index_select(0, i)),
        ("index_add", |e, x, i| x.index_add(0, i, e)),
        ("slice", |_, x, _| x.index((.., 3..))),
    ];
    for (name, op) in ops {
        let y = op(&empty, &x, &none);
        let inputs = [&empty, &x];
        let options = GradOptions {
            allow_unused: true,
            ..Default::default()
        };
        let grads = autograd::grad(&[&y.sum()], &inputs, options);
        if let Some(g) = &grads[0] {
            assert_eq!(g.shape(), [0, 3], "{name}");
        }
        // `x` gets its gradient, which is zero where it reaches only an empty result.
        if let Some(g) = &grads[1] {
            assert_eq!(g.shape(), [2, 3], "{name}");
            let expected = if name == "index_add" { 1.0 } else { 0.0 };
            assert!(g.to_vec::<f64>().iter().all(|&v| v == expected), "{name}");
        }
    }
}
