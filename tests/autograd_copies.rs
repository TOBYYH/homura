//! Gradients of the operations that copy into a new shape: the nodes of `concat`, `flip`, `triu`
//! and `tril` (pinned PyTorch names, edges and values, including skipped empty inputs, mixed
//! dtypes and non-finite upstream gradients), their second derivatives, and the composite
//! `stack` and `repeat`. Also metadata-only backward passes and empty inputs. Run in debug and
//! release profiles.

use homura::autograd::{self, GradFn, GradOptions};
use homura::{DType, Device, Tensor};

fn leaf(values: &[f64], shape: impl homura::IntoShape) -> Tensor {
    let x = Tensor::from_slice(values, shape);
    x.requires_grad_(true);
    x
}

/// A leaf of `shape` holding `0, 1, 2, ...`.
fn counting(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    leaf(&(0..n).map(|i| i as f64).collect::<Vec<_>>(), shape)
}

/// `0, 1, 2, ...` of `shape`, not requiring grad.
fn arange(shape: &[usize]) -> Tensor {
    counting(shape).detach()
}

fn scalar(v: f64) -> Tensor {
    Tensor::from_slice(&[v], [])
}

fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor, graph: bool) -> Vec<Option<Tensor>> {
    let options = GradOptions {
        grad_outputs: vec![Some(g.clone())],
        create_graph: graph,
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
}

fn grad_with(out: &Tensor, x: &Tensor, g: &Tensor, graph: bool) -> Tensor {
    grads(out, &[x], g, graph).remove(0).unwrap()
}

fn edge_names(f: &GradFn) -> Vec<Option<&'static str>> {
    f.next_functions()
        .iter()
        .map(|e| e.as_ref().map(|(f, _)| f.name()))
        .collect()
}

/// The values of `t` as `f64`, with NaN mapped to a marker so that vectors compare exactly.
fn exact(t: &Tensor) -> Vec<String> {
    t.to(DType::F64)
        .to_vec::<f64>()
        .iter()
        .map(|v| format!("{v:?}"))
        .collect()
}

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

// ---------------------------------------------------------------- concat

#[test]
fn concat_narrows_the_gradient_back_to_each_input() {
    // Measured on the pinned PyTorch: one `CatBackward0` with an edge per input (none for an
    // integer input), each gradient the input's part of `g` in the input's dtype, and a skipped
    // one-dimensional empty input gets an empty gradient.
    let a = Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5.], [2, 3]);
    a.requires_grad_(true);
    let e = Tensor::from_slice::<f32>(&[], [0]);
    e.requires_grad_(true);
    let b = counting(&[2, 2]);
    let i = homura::tensor([[7i64, 8], [9, 10]]);
    let out = homura::concat(&[&a, &e, &b, &i], 1);
    assert_eq!(out.dtype(), DType::F64);
    let f = out.grad_fn().unwrap();
    assert_eq!(f.name(), "CatBackward0");
    assert_eq!(
        edge_names(&f),
        [
            Some("AccumulateGrad"),
            Some("AccumulateGrad"),
            Some("AccumulateGrad"),
            None
        ]
    );
    let g = Tensor::from_vec((1..=14).map(f64::from).collect(), [2, 7]);
    let [ga, ge, gb] = <[_; 3]>::try_from(grads(&out, &[&a, &e, &b], &g, false)).unwrap();
    let (ga, ge, gb) = (ga.unwrap(), ge.unwrap(), gb.unwrap());
    assert_eq!(ga.dtype(), DType::F32);
    assert_eq!(ga.to_vec::<f32>(), [1., 2., 3., 8., 9., 10.]);
    assert_eq!((ge.shape(), ge.dtype()), ([0].into(), DType::F32));
    assert_eq!(gb.dtype(), DType::F64);
    assert_eq!(gb.to_vec::<f64>(), [4., 5., 11., 12.]);
}

#[test]
fn concat_boundaries() {
    let a = counting(&[2, 3]);
    let e = leaf(&[], [0]);
    // A skipped empty input before the others, with a negative dimension.
    let out = homura::concat(&[&e, &a], -1);
    assert_eq!(out.shape(), [2, 3]);
    let g = arange(&[2, 3]);
    let [ge, ga] = <[_; 2]>::try_from(grads(&out, &[&e, &a], &g, false)).unwrap();
    assert_eq!(ge.unwrap().shape(), [0]);
    assert_eq!(ga.unwrap().to_vec::<f64>(), g.to_vec::<f64>());
    // Only empty inputs: an empty result, still recorded.
    let e2 = leaf(&[], [0]);
    let out = homura::concat(&[&e, &e2], 0);
    assert_eq!(out.grad_fn().unwrap().name(), "CatBackward0");
    for grad in grads(&out, &[&e, &e2], &arange(&[0]), false) {
        assert_eq!(grad.unwrap().shape(), [0]);
    }
    // An empty input with two dimensions is not skipped.
    let z = leaf(&[], [2, 0]);
    let out = homura::concat(&[&a, &z], 1);
    let [ga, gz] = <[_; 2]>::try_from(grads(&out, &[&a, &z], &g, false)).unwrap();
    assert_eq!(ga.unwrap().to_vec::<f64>(), g.to_vec::<f64>());
    assert_eq!(gz.unwrap().shape(), [2, 0]);
    // The same input twice gets both parts.
    let out = homura::concat(&[&a, &a], 0);
    let ga = grad_with(&out, &a, &arange(&[4, 3]), false);
    assert_eq!(ga.to_vec::<f64>(), [6., 8., 10., 12., 14., 16.]);
    // A single input still gets its own node.
    let out = homura::concat(&[&a], 0);
    assert_eq!(out.grad_fn().unwrap().name(), "CatBackward0");
    assert_eq!(
        grad_with(&out, &a, &g, false).to_vec::<f64>(),
        g.to_vec::<f64>()
    );
    // Non-finite upstream gradients and the sign of zero pass through unchanged.
    let (x, y) = (leaf(&[1., 2.], [2]), leaf(&[3.], [1]));
    let out = homura::concat(&[&x, &y], 0);
    let g = Tensor::from_slice(&[NAN, INF, -0.0], [3]);
    let [gx, gy] = <[_; 2]>::try_from(grads(&out, &[&x, &y], &g, false)).unwrap();
    assert_eq!(exact(&gx.unwrap()), ["NaN", "inf"]);
    assert_eq!(exact(&gy.unwrap()), ["-0.0"]);
    // Non-contiguous inputs.
    let t = arange(&[3, 2]).transpose(0, 1);
    t.requires_grad_(true);
    let out = homura::concat(&[&a, &t], 0);
    let gt = grad_with(&out, &t, &arange(&[4, 3]), false);
    assert_eq!(gt.to_vec::<f64>(), [6., 7., 8., 9., 10., 11.]);
}

#[test]
fn concat_gradients_differentiate_again() {
    // As in PyTorch, each gradient is a slice of `g` (converted back to the input's dtype by the
    // engine), and a skipped empty input's gradient has no graph.
    let a = Tensor::from_slice(&[0f32, 1., 2., 3.], [2, 2]);
    a.requires_grad_(true);
    let b = counting(&[2, 1]);
    let e = leaf(&[], [0]);
    let out = homura::concat(&[&a, &e, &b], 1);
    let g = arange(&[2, 3]);
    g.requires_grad_(true);
    let [ga, ge, gb] = <[_; 3]>::try_from(grads(&out, &[&a, &e, &b], &g, true)).unwrap();
    let (ga, ge, gb) = (ga.unwrap(), ge.unwrap(), gb.unwrap());
    let fa = ga.grad_fn().unwrap();
    assert_eq!(fa.name(), "ToCopyBackward0");
    assert_eq!(edge_names(&fa), [Some("SliceBackward0")]);
    let fb = gb.grad_fn().unwrap();
    assert_eq!(fb.name(), "SliceBackward0");
    assert_eq!(edge_names(&fb), [Some("AccumulateGrad")]);
    assert!(!ge.requires_grad());
    // d(sum(gb * w))/dg puts `w` where `b`'s part of `g` is.
    let w = Tensor::from_slice(&[5.0, 7.0], [2, 1]);
    let gg = grad_with(&(&gb * &w).sum(), &g, &scalar(1.0), false);
    assert_eq!(gg.to_vec::<f64>(), [0., 0., 5., 0., 0., 7.]);
}

// ---------------------------------------------------------------- stack

#[test]
fn stack_is_built_from_unsqueeze_and_concat() {
    // PyTorch records one `StackBackward0`; homura's `stack` concatenates unsqueezed inputs, so
    // its node is `CatBackward0` over `UnsqueezeBackward0`s, with the same gradients.
    let a = counting(&[2, 3]);
    let s = homura::stack(&[a.clone(), &a * 2.0], 1);
    assert_eq!(s.shape(), [2, 2, 3]);
    let f = s.grad_fn().unwrap();
    assert_eq!(f.name(), "CatBackward0");
    assert_eq!(
        edge_names(&f),
        [Some("UnsqueezeBackward0"), Some("UnsqueezeBackward0")]
    );
    let ga = grad_with(&s, &a, &arange(&[2, 2, 3]), false);
    assert_eq!(ga.to_vec::<f64>(), [6., 9., 12., 24., 27., 30.]);
    // A new last dimension and mixed dtypes.
    let h = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], [2, 3]);
    h.requires_grad_(true);
    let s = homura::stack(&[&a, &h], -1);
    assert_eq!((s.shape(), s.dtype()), ([2, 3, 2].into(), DType::F64));
    let g = arange(&[2, 3, 2]);
    let [ga, gh] = <[_; 2]>::try_from(grads(&s, &[&a, &h], &g, false)).unwrap();
    assert_eq!(ga.unwrap().to_vec::<f64>(), [0., 2., 4., 6., 8., 10.]);
    let gh = gh.unwrap();
    assert_eq!(gh.dtype(), DType::F32);
    assert_eq!(gh.to_vec::<f32>(), [1., 3., 5., 7., 9., 11.]);
}

// ---------------------------------------------------------------- flip

#[test]
fn flip_flips_the_gradient_back() {
    let a = counting(&[2, 3]);
    let g = arange(&[2, 3]);
    let cases: [(Tensor, [f64; 6]); 4] = [
        (a.flip([0, 1]), [5., 4., 3., 2., 1., 0.]),
        (a.flip(-1), [2., 1., 0., 5., 4., 3.]),
        (a.flip([-2]), [3., 4., 5., 0., 1., 2.]),
        // No dimensions: a copy, still recorded.
        (a.flip(Vec::<i64>::new()), [0., 1., 2., 3., 4., 5.]),
    ];
    for (out, expected) in cases {
        let f = out.grad_fn().unwrap();
        assert_eq!(f.name(), "FlipBackward0");
        assert_eq!(edge_names(&f), [Some("AccumulateGrad")]);
        assert_eq!(grad_with(&out, &a, &g, false).to_vec::<f64>(), expected);
    }
    // A zero-dimensional input takes 0 and -1 as its dimension.
    let s = leaf(&[3.0], []);
    for out in [
        s.flip(Vec::<i64>::new()),
        s.flip(0),
        s.flip(-1),
        s.flip([0]),
    ] {
        assert_eq!(out.grad_fn().unwrap().name(), "FlipBackward0");
        let gs = grad_with(&out, &s, &scalar(5.0), false);
        assert_eq!((gs.shape(), gs.item::<f64>()), ([].into(), 5.0));
    }
    // A non-contiguous input, and non-finite upstream gradients pass through.
    let t = arange(&[3, 2]).transpose(0, 1);
    t.requires_grad_(true);
    let g = Tensor::from_slice(&[NAN, INF, -0.0, 1.0, -INF, 2.0], [2, 3]);
    let gt = grad_with(&t.flip(1), &t, &g, false);
    assert_eq!(exact(&gt), ["-0.0", "inf", "NaN", "2.0", "-inf", "1.0"]);
    // The second derivative is a flip of `g`.
    let g = arange(&[2, 3]);
    g.requires_grad_(true);
    let ga = grad_with(&a.flip(1), &a, &g, true);
    let f = ga.grad_fn().unwrap();
    assert_eq!(f.name(), "FlipBackward0");
    assert_eq!(edge_names(&f), [Some("AccumulateGrad")]);
}

// ---------------------------------------------------------------- triu and tril

#[test]
fn triangles_select_the_gradient() {
    // Measured on the pinned PyTorch: the gradient is the same triangle of `g`, selected, so a
    // NaN or infinite upstream gradient outside it gives 0, not NaN.
    let m = counting(&[3, 4]);
    let mut gv = vec![NAN; 12];
    gv[0] = INF;
    let g = Tensor::from_vec(gv, [3, 4]);
    let (n, i, z) = ("NaN", "inf", "0.0");
    type Case = (&'static str, Tensor, [&'static str; 12]);
    let cases: [Case; 10] = [
        (
            "TriuBackward0",
            m.triu(0),
            [i, n, n, n, z, n, n, n, z, z, n, n],
        ),
        (
            "TriuBackward0",
            m.triu(1),
            [z, n, n, n, z, z, n, n, z, z, z, n],
        ),
        (
            "TriuBackward0",
            m.triu(-1),
            [i, n, n, n, n, n, n, n, z, n, n, n],
        ),
        ("TriuBackward0", m.triu(5), [z; 12]),
        (
            "TriuBackward0",
            m.triu(-5),
            [i, n, n, n, n, n, n, n, n, n, n, n],
        ),
        (
            "TrilBackward0",
            m.tril(0),
            [i, z, z, z, n, n, z, z, n, n, n, z],
        ),
        (
            "TrilBackward0",
            m.tril(1),
            [i, n, z, z, n, n, n, z, n, n, n, n],
        ),
        (
            "TrilBackward0",
            m.tril(-1),
            [z, z, z, z, n, z, z, z, n, n, z, z],
        ),
        (
            "TrilBackward0",
            m.tril(5),
            [i, n, n, n, n, n, n, n, n, n, n, n],
        ),
        ("TrilBackward0", m.tril(-5), [z; 12]),
    ];
    for (k, (name, out, expected)) in cases.into_iter().enumerate() {
        let f = out.grad_fn().unwrap();
        assert_eq!(f.name(), name, "case {k}");
        assert_eq!(edge_names(&f), [Some("AccumulateGrad")], "case {k}");
        assert_eq!(exact(&grad_with(&out, &m, &g, false)), expected, "case {k}");
    }
    // Batches of non-contiguous matrices, with a non-contiguous upstream gradient.
    let base = arange(&[2, 3, 3]);
    let x = base.transpose(1, 2);
    x.requires_grad_(true);
    let g = (arange(&[2, 3, 3]) + 1.0).transpose(1, 2);
    let gx = grad_with(&x.triu(1), &x, &g, false);
    assert_eq!(gx.to_vec::<f64>(), g.triu(1).to_vec::<f64>());
    let gx = grad_with(&x.tril(-1), &x, &g, false);
    assert_eq!(gx.to_vec::<f64>(), g.tril(-1).to_vec::<f64>());
    // The second derivative is the same triangle of `g`.
    let g = arange(&[3, 4]);
    g.requires_grad_(true);
    for (name, out) in [("TriuBackward0", m.triu(1)), ("TrilBackward0", m.tril(-1))] {
        let gm = grad_with(&out, &m, &g, true);
        let f = gm.grad_fn().unwrap();
        assert_eq!(f.name(), name);
        assert_eq!(edge_names(&f), [Some("AccumulateGrad")]);
    }
}

// ---------------------------------------------------------------- repeat

#[test]
fn repeat_sums_the_copies() {
    // PyTorch records one `RepeatBackward0`; homura's `repeat` is a composite of recorded
    // reshapes, `expand` and a copy, with the same gradients.
    let a = counting(&[2, 3]);
    let r = a.repeat([2, 1, 2]);
    assert_eq!(r.shape(), [2, 2, 6]);
    let ga = grad_with(&r, &a, &arange(&[2, 2, 6]), false);
    assert_eq!(ga.to_vec::<f64>(), [30., 34., 38., 54., 58., 62.]);
    let r = a.repeat([0, 2]);
    assert_eq!(r.shape(), [0, 6]);
    let ga = grad_with(&r, &a, &arange(&[0, 6]), false);
    assert_eq!(ga.to_vec::<f64>(), [0.0; 6]);
}

// ---------------------------------------------------------------- shared behavior

#[test]
fn copies_save_nothing_so_backward_passes_can_repeat() {
    let x = counting(&[2, 2]);
    let outputs = [
        homura::concat(&[&x, &x], 1),
        homura::stack(&[&x], 0),
        x.flip(0),
        x.triu(0),
        x.tril(0),
    ];
    for out in outputs {
        let y = out.sum();
        y.backward();
        y.backward();
    }
    // Twice each: concat 2 per element, stack 1, flip 1, and triu and tril 1 between them plus
    // 1 more on the diagonal.
    assert_eq!(x.grad().unwrap().to_vec::<f64>(), [12., 10., 10., 12.]);
}

#[test]
fn gradients_of_copies_on_meta_and_empty_inputs() {
    for device in [Device::Cpu, Device::Meta] {
        for rows in [0, 1 << 40] {
            if device == Device::Cpu && rows != 0 {
                continue;
            }
            let x = homura::zeros_with([2, rows, 3], device);
            x.requires_grad_(true);
            let y = homura::zeros_with([2, 1, 3], device).to(DType::F64);
            y.requires_grad_(true);
            let e = homura::zeros_with([0], device);
            e.requires_grad_(true);
            let outputs = [
                ("concat", homura::concat(&[&x, &e, &y], 1)),
                ("stack", homura::stack(&[&x, &x], 1)),
                ("flip", x.flip([0, 2])),
                ("triu", x.triu(1)),
                ("tril", x.tril(-1)),
            ];
            for (name, out) in outputs {
                let g = homura::ones_with(out.shape(), (out.dtype(), device));
                g.requires_grad_(true);
                let inputs = [&x, &e, &y];
                let options = GradOptions {
                    grad_outputs: vec![Some(g)],
                    create_graph: true,
                    allow_unused: true,
                    ..Default::default()
                };
                let found = autograd::grad(&[&out], &inputs, options);
                for (input, grad) in inputs.iter().zip(found) {
                    let Some(grad) = grad else {
                        assert_ne!(name, "concat", "{name} on {device}");
                        continue;
                    };
                    let what = format!("{name} on {device}");
                    assert_eq!(grad.shape(), input.shape(), "{what}");
                    assert_eq!(grad.dtype(), input.dtype(), "{what}");
                    assert_eq!(grad.device(), device, "{what}");
                }
            }
        }
    }
}
