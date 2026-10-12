//! The backward formulas written after the minimal engine. M1 step 4: `erf`, `sign`, and the
//! indexing operations whose formulas use each other in pairs (`index_select` and `index_add`,
//! `gather` and `scatter_add`, `select` and `select_scatter`, slicing and `narrow` with
//! `slice_scatter`), plus integer indexing and `as_strided`. Step 6a: the elementwise unary
//! functions, with `div` and `where_`, which their formulas use. Gradients are checked by value,
//! including non-finite inputs and upstream gradients, node names and edges against PyTorch's,
//! and each formula's own backward pass for higher-order derivatives.
//! `tools/golden/check_m1_formulas.py` checks the same cases against the reference PyTorch build.
//! Run in debug and release profiles.

use std::f32::consts::FRAC_2_SQRT_PI;
use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, BackwardOptions, GradFn, GradOptions};
use homura::{DType, Device, GeluApproximate, Tensor, where_};

/// A leaf of shape `shape` that requires grad.
fn leaf(values: &[f32], shape: &[usize]) -> Tensor {
    let x = Tensor::from_slice(values, shape);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to(DType::F32).to_vec()
}

/// Asserts equal values, with NaN equal to NaN and infinities equal only to themselves.
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: {actual:?} vs {expected:?}"
    );
    for (a, e) in actual.iter().zip(expected) {
        let ok = if e.is_nan() {
            a.is_nan()
        } else if e.is_infinite() {
            a == e
        } else {
            (a - e).abs() <= 1e-6 * e.abs().max(1.0)
        };
        assert!(ok, "{what}: {actual:?} vs {expected:?}");
    }
}

/// The message of the panic of `f`, or `None` if it returns.
fn try_panic_message(f: impl FnOnce()) -> Option<String> {
    let payload = catch_unwind(AssertUnwindSafe(f)).err()?;
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message");
    Some(message)
}

fn panic_message(f: impl FnOnce()) -> String {
    try_panic_message(f).expect("expected a panic")
}

/// The gradients of `out` with respect to `inputs` for the upstream gradient `g`.
fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor) -> Vec<Tensor> {
    let options = GradOptions {
        grad_outputs: vec![Some(g.clone())],
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
        .into_iter()
        .map(|g| g.expect("a gradient"))
        .collect()
}

/// The gradient of `out` with respect to `input` for the upstream gradient `g`.
fn grad(out: &Tensor, input: &Tensor, g: &[f32]) -> Vec<f32> {
    let g = Tensor::from_slice(g, out.shape());
    vals(&grads(out, &[input], &g)[0])
}

fn ints(values: &[i64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
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

/// The node name of `t`'s `grad_fn` and the names of its edges, `None` where an input has none.
fn node(t: &Tensor) -> (&'static str, Vec<Option<&'static str>>) {
    let f = t.grad_fn().expect("a grad_fn");
    let edges = f
        .next_functions()
        .iter()
        .map(|e| e.as_ref().map(|(f, _)| f.name()))
        .collect();
    (f.name(), edges)
}

/// The unary functions of step 6a, by the names that [`unary`] takes.
const UNARY: [&str; 13] = [
    "abs",
    "relu",
    "log",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "silu",
    "gelu",
    "gelu with tanh",
];

/// Applies the unary function `name` of [`UNARY`] to `x`.
fn unary(name: &str, x: &Tensor) -> Tensor {
    match name {
        "abs" => x.abs(),
        "relu" => x.relu(),
        "log" => x.log(),
        "sqrt" => x.sqrt(),
        "rsqrt" => x.rsqrt(),
        "reciprocal" => x.reciprocal(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tanh" => x.tanh(),
        "sigmoid" => x.sigmoid(),
        "silu" => x.silu(),
        "gelu" => x.gelu(GeluApproximate::None),
        "gelu with tanh" => x.gelu(GeluApproximate::Tanh),
        _ => unreachable!("{name}"),
    }
}

// ---------------------------------------------------------------- nodes

#[test]
fn node_names_and_edges_follow_pytorch() {
    let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let acc = Some("AccumulateGrad");
    let index = ints(&[1, 0]);
    let zeros = homura::zeros([3, 3]);
    let other = homura::zeros([3, 3]);
    other.requires_grad_(true);
    for (case, out, name, edges) in [
        ("erf", x.erf(), "ErfBackward0", vec![acc]),
        ("sign", x.sign(), "SignBackward0", vec![acc]),
        ("select", x.select(0, 1), "SelectBackward0", vec![acc]),
        ("narrow", x.narrow(1, 1, 2), "SliceBackward0", vec![acc]),
        ("slice", x.index(1..), "SliceBackward0", vec![acc]),
        (
            "as_strided",
            x.as_strided([2], [1], 1),
            "AsStridedBackward0",
            vec![acc],
        ),
        (
            "index_select",
            x.index_select(1, &index),
            "IndexSelectBackward0",
            vec![acc],
        ),
        (
            "index_add of a constant",
            zeros.narrow(0, 0, 2).index_add(1, &ints(&[1, 0, 2]), &x),
            "IndexAddBackward0",
            vec![None, acc],
        ),
        (
            "index_add",
            other.narrow(0, 0, 2).index_add(1, &ints(&[1, 0, 2]), &x),
            "IndexAddBackward0",
            vec![Some("SliceBackward0"), acc],
        ),
        (
            "gather",
            x.gather(1, &homura::tensor([[0i64], [2]])),
            "GatherBackward0",
            vec![acc],
        ),
        (
            "scatter_add",
            other.scatter_add(1, &homura::tensor([[0i64, 1, 2], [2, 1, 0]]), &x),
            "ScatterAddBackward0",
            vec![acc, acc],
        ),
        (
            "select_scatter",
            other.select_scatter(&x.select(0, 1), 1, 0),
            "SelectScatterBackward0",
            vec![acc, Some("SelectBackward0")],
        ),
        (
            "slice_scatter",
            other.slice_scatter(&x, 0, Some(1), None, 1),
            "SliceScatterBackward0",
            vec![acc, acc],
        ),
    ] {
        assert_eq!(node(&out), (name, edges), "{case}");
    }
    // Integer indexing reshapes the selected slices with a recorded `unsafe_view`.
    assert_eq!(
        graph_names(&x.index(&index).grad_fn().unwrap()),
        ["UnsafeViewBackward0", "IndexBackward0", "AccumulateGrad"]
    );
    // `split` and `chunk` are made of `narrow`, where PyTorch records one `SplitBackward0`.
    for piece in x.split(2, 1).iter().chain(&x.chunk(2, 0)) {
        assert_eq!(node(piece), ("SliceBackward0", vec![acc]));
    }
    // Integer outputs record nothing.
    let i = Tensor::from_slice(&[1i64, -2], [2]);
    assert!(i.sign().grad_fn().is_none());
}

#[test]
fn elementwise_node_names_and_edges_follow_pytorch() {
    let x = leaf(&[1., 2., 3.], &[3]);
    let y = leaf(&[4., 5., 6.], &[3]);
    let acc = Some("AccumulateGrad");
    let constant = Tensor::from_slice(&[1f32, 2., 3.], [3]);
    let integers = ints(&[1, 2, 3]);
    let cond = homura::tensor([true, false, true]);
    for (case, out, name, edges) in [
        ("abs", x.abs(), "AbsBackward0", vec![acc]),
        ("relu", x.relu(), "ReluBackward0", vec![acc]),
        ("log", x.log(), "LogBackward0", vec![acc]),
        ("sqrt", x.sqrt(), "SqrtBackward0", vec![acc]),
        ("rsqrt", x.rsqrt(), "RsqrtBackward0", vec![acc]),
        (
            "reciprocal",
            x.reciprocal(),
            "ReciprocalBackward0",
            vec![acc],
        ),
        ("sin", x.sin(), "SinBackward0", vec![acc]),
        ("cos", x.cos(), "CosBackward0", vec![acc]),
        ("tanh", x.tanh(), "TanhBackward0", vec![acc]),
        ("sigmoid", x.sigmoid(), "SigmoidBackward0", vec![acc]),
        ("silu", x.silu(), "SiluBackward0", vec![acc]),
        (
            "gelu",
            x.gelu(GeluApproximate::None),
            "GeluBackward0",
            vec![acc],
        ),
        (
            "gelu with tanh",
            x.gelu(GeluApproximate::Tanh),
            "GeluBackward0",
            vec![acc],
        ),
        ("div", &x / &y, "DivBackward0", vec![acc, acc]),
        ("div by a scalar", &x / 2.0, "DivBackward0", vec![acc, None]),
        ("div by an integer", &x / 2, "DivBackward0", vec![acc, None]),
        (
            "div by a constant",
            &x / &constant,
            "DivBackward0",
            vec![acc, None],
        ),
        (
            "div by integers",
            x.div(&integers),
            "DivBackward0",
            vec![acc, None],
        ),
        (
            "integers divided",
            integers.div(&x),
            "DivBackward0",
            vec![None, acc],
        ),
        // PyTorch's Python operator computes `2.0 / x` as `x.reciprocal() * 2.0`, which records a
        // `MulBackward0` after a `ReciprocalBackward0`; homura divides, like
        // `torch.div(torch.tensor(2.0), x)`.
        ("scalar divided", 2.0 / &x, "DivBackward0", vec![None, acc]),
        (
            "where",
            where_(&cond, &x, &y),
            "WhereBackward0",
            vec![acc, acc],
        ),
        (
            "where with a scalar",
            where_(&cond, &x, 0.0),
            "WhereBackward0",
            vec![acc, None],
        ),
        (
            "where of a scalar",
            where_(&cond, 0.0, &y),
            "WhereBackward0",
            vec![None, acc],
        ),
        (
            "where with a constant",
            where_(&cond, &x, &constant),
            "WhereBackward0",
            vec![acc, None],
        ),
    ] {
        assert_eq!(node(&out), (name, edges), "{case}");
    }
    // Integer and bool results record nothing.
    assert!(integers.abs().grad_fn().is_none());
    assert!(integers.relu().grad_fn().is_none());
}

// ---------------------------------------------------------------- values

#[test]
fn gradients_of_erf_and_sign() {
    let x = leaf(&[0., 0.5, -1., f32::INFINITY, f32::NAN, 30.], &[6]);
    assert_close(
        &grad(&x.erf(), &x, &[1., 2., 3., 4., 5., 6.]),
        &[FRAC_2_SQRT_PI, 1.7575653, 1.2453226, 0., f32::NAN, 0.],
        "erf",
    );
    // Zero, even for an infinite or NaN upstream gradient.
    let x = leaf(&[-2., 0., 3., f32::NAN], &[4]);
    assert_eq!(
        grad(&x.sign(), &x, &[1., f32::INFINITY, f32::NAN, 2.]),
        [0.; 4]
    );
}

#[test]
fn gradients_of_index_select_and_integer_indexing() {
    let x = leaf(&[1., 2., 3., 4., 5., 6.], &[3, 2]);
    let g = [1., 2., 3., 4., 5., 6.];
    // Repeated positions add up.
    let dup = x.index_select(0, &ints(&[2, 0, 2]));
    assert_eq!(grad(&dup, &x, &g), [3., 4., 0., 0., 6., 8.]);
    let i32_index = Tensor::from_slice(&[1i32, 1], [2]);
    let y = x.index_select(1, &i32_index);
    assert_eq!(grad(&y, &x, &g), [0., 3., 0., 7., 0., 11.]);
    let y = x.index_select(0, &homura::tensor(1i64));
    assert_eq!(grad(&y, &x, &[7., 8.]), [0., 0., 7., 8., 0., 0.]);
    // Integer indexing wraps negative positions.
    let y = x.index(ints(&[-1, 0, -1]));
    assert_eq!(grad(&y, &x, &g), [3., 4., 0., 0., 6., 8.]);
    let y = x.index((.., &homura::tensor([[1i64, 0], [1, 1]])));
    let g: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(grad(&y, &x, &g), [1., 5., 5., 17., 9., 29.]);
}

#[test]
fn gradients_of_select_slices_and_narrow() {
    let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_eq!(
        grad(&x.select(1, -1), &x, &[7., 8.]),
        [0., 0., 7., 0., 0., 8.]
    );
    assert_eq!(
        grad(&x.index((.., (0..).step_by(2))), &x, &[1., 2., 3., 4.]),
        [1., 0., 2., 3., 0., 4.]
    );
    assert_eq!(
        grad(&x.index((-5..1, 1..100)), &x, &[1., 2.]),
        [0., 1., 2., 0., 0., 0.]
    );
    assert_eq!(
        grad(&x.narrow(1, 1, 2), &x, &[1., 2., 3., 4.]),
        [0., 1., 2., 0., 3., 4.]
    );
    assert_eq!(
        grad(&x.split(2, 1)[1], &x, &[1., 2.]),
        [0., 0., 1., 0., 0., 2.]
    );
    assert_eq!(
        grad(&x.index((1.., (0..).step_by(2))), &x, &[1., 2.]),
        [0., 0., 0., 1., 0., 2.]
    );
}

#[test]
fn gradients_of_as_strided() {
    let values: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let base = leaf(&values, &[10]);
    // The output starts at storage offset 1, outside the view `base[2:8]`: that element's
    // gradient reaches the view's storage but not the view, so it is dropped.
    let view = base.narrow(0, 2, 6);
    let y = view.as_strided([3], [2], 1);
    assert_eq!(
        grad(&y, &base, &[1., 2., 3.]),
        [0., 0., 0., 2., 0., 3., 0., 0., 0., 0.]
    );
    let y = view.as_strided([3], [2], 1);
    assert_eq!(grad(&y, &view, &[1., 2., 3.]), [0., 2., 0., 3., 0., 0.]);
    // Output elements that share storage add up.
    let y = base.as_strided([2, 2], [1, 1], 0);
    assert_eq!(
        grad(&y, &base, &[1., 2., 3., 4.]),
        [1., 5., 4., 0., 0., 0., 0., 0., 0., 0.]
    );
    let y = base.as_strided([3], [0], 4);
    assert_eq!(
        grad(&y, &base, &[1., 2., 3.]),
        [0., 0., 0., 0., 6., 0., 0., 0., 0., 0.]
    );
    // Through a strided view, whose layout reads the gradient back.
    let y = base.index((1..).step_by(3)).as_strided([2, 2], [3, 1], 1);
    assert_eq!(
        grad(&y, &base, &[1., 2., 3., 4.]),
        [0., 1., 0., 0., 3., 0., 0., 0., 0., 0.]
    );
    // An empty output sends zeros.
    let y = base.as_strided([0], [1], 3);
    assert_eq!(grad(&y.sum(), &base, &[1.]), [0.; 10]);
    // The gradient of an input whose elements overlap is not supported yet.
    let row = leaf(&[1., 2.], &[2]);
    let y = row.expand([3, 2]).as_strided([2], [1], 0);
    let message = panic_message(|| {
        y.sum().backward();
    });
    assert_eq!(
        message,
        "computing the gradient of as_strided for an input whose elements overlap in memory is \
         not supported yet; deep_clone() the input first"
    );
}

#[test]
fn gradients_of_index_add_gather_and_scatter_add() {
    let source = leaf(&[1., 2., 3., 4., 5., 6.], &[3, 2]);
    let x = leaf(&[0.; 4], &[2, 2]);
    let y = x.index_add(0, &ints(&[1, 0, 1]), &source);
    let g = Tensor::from_slice(&[1f32, 2., 3., 4.], [2, 2]);
    let [gx, gs] = <[Tensor; 2]>::try_from(grads(&y, &[&x, &source], &g)).unwrap();
    assert_eq!(vals(&gx), [1., 2., 3., 4.]);
    assert_eq!(vals(&gs), [3., 4., 1., 2., 3., 4.]);

    let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let y = x.gather(1, &homura::tensor([[2i64, 2], [0, 1]]));
    assert_eq!(grad(&y, &x, &[1., 2., 3., 4.]), [0., 0., 3., 3., 4., 0.]);

    let x = leaf(&[0.; 6], &[2, 3]);
    let src = leaf(&[1., 2., 3., 4.], &[2, 2]);
    let y = x.scatter_add(1, &homura::tensor([[2i64, 2], [0, 1]]), &src);
    let g = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], [2, 3]);
    let [gx, gs] = <[Tensor; 2]>::try_from(grads(&y, &[&x, &src], &g)).unwrap();
    assert_eq!(vals(&gx), [1., 2., 3., 4., 5., 6.]);
    assert_eq!(vals(&gs), [3., 3., 4., 5.]);
    // As in PyTorch, the gradient of `src` has the shape of the index, so a `src` larger than
    // the index can't get one.
    let src = leaf(&[0.; 12], &[3, 4]);
    let y = homura::zeros([3, 4]).scatter_add(1, &homura::tensor([[0i64], [1], [2]]), &src);
    assert_eq!(
        panic_message(|| y.sum().backward()),
        "Function ScatterAddBackward0 returned an invalid gradient at index 1 - got [3, 1] but \
         expected shape compatible with [3, 4]"
    );
}

#[test]
fn gradients_of_select_scatter_and_slice_scatter() {
    let x = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let g = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], [2, 3]);
    // `src` gets its gradient in its own dtype.
    let src = Tensor::from_slice(&[7f64, 8.], [2]);
    src.requires_grad_(true);
    let y = x.select_scatter(&src, 1, 1);
    let [gx, gs] = <[Tensor; 2]>::try_from(grads(&y, &[&x, &src], &g)).unwrap();
    assert_eq!(vals(&gx), [1., 0., 3., 4., 0., 6.]);
    assert_eq!((gs.dtype(), vals(&gs)), (DType::F64, vec![2., 5.]));

    let src = leaf(&[7., 8.], &[2, 1]);
    let y = x.slice_scatter(&src, 1, Some(2), None, 1);
    let [gx, gs] = <[Tensor; 2]>::try_from(grads(&y, &[&x, &src], &g)).unwrap();
    assert_eq!(vals(&gx), [1., 2., 0., 4., 5., 0.]);
    assert_eq!(vals(&gs), [3., 6.]);
    let src = leaf(&[7., 8., 9., 10.], &[2, 2]);
    let y = x.slice_scatter(&src, 1, None, None, 2);
    let [gx, gs] = <[Tensor; 2]>::try_from(grads(&y, &[&x, &src], &g)).unwrap();
    assert_eq!(vals(&gx), [0., 2., 0., 0., 5., 0.]);
    assert_eq!(vals(&gs), [1., 3., 4., 6.]);
}

#[test]
fn gradients_of_unary_functions_at_special_values() {
    // Zeros of both signs, infinities (`I`) and NaN (`N`), with infinite and NaN upstream
    // gradients. The formulas follow PyTorch's: `relu` selects, so it gives zeros even where `g`
    // is infinite or NaN, while `abs` multiplies by the sign, which gives NaN for an infinite `g`
    // at zero.
    const I: f32 = f32::INFINITY;
    const N: f32 = f32::NAN;
    let x = [0., -0., -1., 2., I, -I, N, 0., -2.];
    let g = [1., 1., I, N, 1., 1., 1., I, N];
    let expected: [(&str, [f32; 9]); 13] = [
        ("abs", [0., 0., -I, N, 1., -1., 0., N, N]),
        ("relu", [0., 0., 0., N, 1., 0., 1., 0., 0.]),
        ("log", [I, -I, -I, N, 0., -0., N, I, N]),
        ("sqrt", [I, -I, N, N, 0., N, N, I, N]),
        ("rsqrt", [-I, I, N, N, -0., N, N, -I, N]),
        ("reciprocal", [-I, -I, -I, N, -0., -0., N, -I, N]),
        ("sin", [1., 1., I, N, N, N, N, I, N]),
        ("cos", [-0., 0., I, N, N, N, N, N, N]),
        ("tanh", [1., 1., I, N, 0., 0., N, I, N]),
        ("sigmoid", [0.25, 0.25, I, N, 0., 0., N, I, N]),
        ("silu", [0.5, 0.5, I, N, N, N, N, I, N]),
        ("gelu", [0.5, 0.5, -I, N, N, N, N, I, N]),
        ("gelu with tanh", [0.5, 0.5, -I, N, N, N, N, I, N]),
    ];
    for (name, expected) in expected {
        let x = leaf(&x, &[9]);
        assert_close(&grad(&unary(name, &x), &x, &g), &expected, name);
    }
    // `relu` passes `g` wherever the result is positive or NaN, even for tiny inputs.
    let x = leaf(&[N, 0., -0., 1e-30, -1e-30], &[5]);
    let gx = grad(&x.relu(), &x, &[I, I, N, I, I]);
    assert_close(&gx, &[I, 0., 0., I, 0.], "relu");
    // Half precision results get gradients in their own dtype.
    for dtype in [DType::F16, DType::BF16] {
        let x = Tensor::from_slice(&[0.25f32, 1., 4.], [3]).to(dtype);
        x.requires_grad_(true);
        let g = homura::ones_with([3], dtype);
        let gx = grads(&x.sqrt(), &[&x], &g).remove(0);
        assert_eq!((gx.dtype(), vals(&gx)), (dtype, vec![1., 0.5, 0.25]));
    }
}

#[test]
fn gradients_of_div() {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    // `a` gets `g / b` and `b` gets `-g * ((a / b) / b)`, PyTorch's formulas.
    let a = leaf(&[1., 0., inf, 1., -1., 2.], &[6]);
    let b = leaf(&[0., 0., inf, inf, -0., 4.], &[6]);
    let g = Tensor::from_slice(&[1f32, 1., 1., 1., 1., inf], [6]);
    let [ga, gb] = <[Tensor; 2]>::try_from(grads(&(&a / &b), &[&a, &b], &g)).unwrap();
    assert_close(&vals(&ga), &[inf, inf, 0., 0., -inf, inf], "a");
    assert_close(&vals(&gb), &[-inf, nan, nan, -0., inf, -inf], "b");
    // Broadcast inputs get their gradients summed to their shapes.
    let a = leaf(&[1., 2.], &[2, 1]);
    let b = leaf(&[1., 2., 4.], &[3]);
    let g = homura::ones([2, 3]);
    let [ga, gb] = <[Tensor; 2]>::try_from(grads(&(&a / &b), &[&a, &b], &g)).unwrap();
    assert_eq!(vals(&ga), [1.75, 1.75]);
    assert_eq!(vals(&gb), [-3., -0.75, -0.1875]);
    // Scalars on either side, and integer tensors, have no edge.
    let x = leaf(&[0., 1., inf, -2.], &[4]);
    let expected = [-inf, -2., -0., -0.5];
    assert_close(&grad(&(2.0 / &x), &x, &[1.; 4]), &expected, "2.0 / x");
    assert_close(&grad(&(2 / &x), &x, &[1.; 4]), &expected, "2 / x");
    assert_close(&grad(&(&x / 2.0), &x, &[1.; 4]), &[0.5; 4], "x / 2.0");
    assert_close(&grad(&(&x / 0.0), &x, &[1.; 4]), &[inf; 4], "x / 0.0");
    let x = leaf(&[0., 1., inf], &[3]);
    assert_close(
        &grad(&(0.0 / &x), &x, &[1.; 3]),
        &[nan, -0., -0.],
        "0.0 / x",
    );
    let x = leaf(&[1., 2., -4.], &[3]);
    let i = ints(&[2, -1, 4]);
    assert_eq!(grad(&x.div(&i), &x, &[1.; 3]), [0.5, -1., 0.25]);
    assert_eq!(grad(&i.div(&x), &x, &[1.; 3]), [-2., 0.25, -0.25]);
    // Each input gets its gradient in its own dtype.
    let a = leaf(&[1., 2.], &[2]);
    let b = Tensor::from_slice(&[3f64, 4.], [2]);
    b.requires_grad_(true);
    let g = homura::ones_with([2], DType::F64);
    let [ga, gb] = <[Tensor; 2]>::try_from(grads(&(&a / &b), &[&a, &b], &g)).unwrap();
    assert_eq!((ga.dtype(), gb.dtype()), (DType::F32, DType::F64));
    assert_close(&vals(&ga), &[1. / 3., 0.25], "a");
    assert_close(&vals(&gb), &[-1. / 9., -0.125], "b");
}

#[test]
fn gradients_of_where() {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    // Each input gets `g` where it was selected and exact zeros elsewhere, even where `g` is
    // infinite or NaN: the formulas select instead of multiplying by a mask.
    let cond = homura::tensor([true, false, true, false]);
    let x = leaf(&[1., 2., 3., 4.], &[4]);
    let y = leaf(&[5., 6., 7., 8.], &[4]);
    let g = Tensor::from_slice(&[nan, inf, -inf, nan], [4]);
    let [gx, gy] = <[Tensor; 2]>::try_from(grads(&where_(&cond, &x, &y), &[&x, &y], &g)).unwrap();
    assert_close(&vals(&gx), &[nan, 0., -inf, 0.], "x");
    assert_close(&vals(&gy), &[0., inf, 0., nan], "y");
    // Broadcast inputs get their gradients summed to their shapes.
    let cond = homura::tensor([[true, false], [false, false]]);
    let x = leaf(&[1., 2.], &[2]);
    let y = leaf(&[3.], &[]);
    let g = Tensor::from_slice(&[1f32, 2., 3., 4.], [2, 2]);
    let [gx, gy] = <[Tensor; 2]>::try_from(grads(&where_(&cond, &x, &y), &[&x, &y], &g)).unwrap();
    assert_eq!((vals(&gx), vals(&gy)), (vec![1., 0.], vec![9.]));
    // A condition broadcast over the input, and scalars on either side.
    let cond = homura::tensor([[true], [false]]);
    let x = leaf(&[1., 2.], &[2]);
    assert_eq!(
        grad(&where_(&cond, &x, -1.0), &x, &[1., 2., 3., 4.]),
        [1., 2.]
    );
    assert_eq!(
        grad(&where_(&cond, 0.5, &x), &x, &[1., 2., 3., 4.]),
        [3., 4.]
    );
    // Each input gets its gradient in its own dtype.
    let cond = homura::tensor([true, false]);
    let x = leaf(&[1., 2.], &[2]);
    let y = Tensor::from_slice(&[3f64, 4.], [2]);
    y.requires_grad_(true);
    let g = homura::ones_with([2], DType::F64);
    let [gx, gy] = <[Tensor; 2]>::try_from(grads(&where_(&cond, &x, &y), &[&x, &y], &g)).unwrap();
    assert_eq!((gx.dtype(), vals(&gx)), (DType::F32, vec![1., 0.]));
    assert_eq!((gy.dtype(), vals(&gy)), (DType::F64, vec![0., 1.]));
}

#[test]
fn meta_leaves_get_gradients_of_the_right_shape() {
    let x = homura::zeros_with([3, 4], Device::Meta);
    x.requires_grad_(true);
    let index = homura::zeros_with([2], (DType::I64, Device::Meta));
    let index_2d = homura::zeros_with([3, 2], (DType::I64, Device::Meta));
    let outputs = [
        x.erf(),
        x.sign(),
        x.select(1, 2),
        x.narrow(0, 1, 2),
        x.index((.., (0..).step_by(3))),
        x.as_strided([2, 2], [1, 4], 1),
        x.index_select(1, &index),
        x.index(&index),
        x.gather(1, &index_2d),
        x.index_add(1, &index, &homura::zeros_with([3, 2], Device::Meta)),
        x.scatter_add(1, &index_2d, &homura::zeros_with([3, 2], Device::Meta)),
        x.select_scatter(&homura::zeros_with([4], Device::Meta), 0, 1),
        x.slice_scatter(
            &homura::zeros_with([3, 2], Device::Meta),
            1,
            Some(1),
            Some(3),
            1,
        ),
        x.abs(),
        x.relu(),
        x.log(),
        x.sqrt(),
        x.rsqrt(),
        x.reciprocal(),
        x.sin(),
        x.cos(),
        x.tanh(),
        x.sigmoid(),
        x.silu(),
        x.gelu(GeluApproximate::None),
        x.gelu(GeluApproximate::Tanh),
        &x / &x.index(0),
        &x.index((.., 0..1)) / &x,
        2.0 / &x,
        where_(&x.greater(0.0), &x, 0.0),
        where_(
            &homura::ones_with([4], (DType::Bool, Device::Meta)),
            1.0,
            &x,
        ),
    ];
    for (i, y) in outputs.iter().enumerate() {
        let g = autograd::grad(&[&y.sum()], &[&x], GradOptions::default()).remove(0);
        let g = g.unwrap_or_else(|| panic!("output {i}"));
        assert_eq!(
            (g.shape().to_vec(), g.dtype(), g.device()),
            (vec![3, 4], DType::F32, Device::Meta),
            "output {i}"
        );
    }
}

// ---------------------------------------------------------------- higher order

/// Checks that the gradient of `f` with respect to `x` has a graph of implemented formulas only
/// and is linear in the upstream gradient `u`: `d<g, r>/du = J r`, which for the linear
/// functions tested here (constants are zeros) is `f(r)` with `r` in place of `x`.
fn check_double_backward(name: &str, x: &Tensor, f: impl Fn(&Tensor) -> Tensor) {
    let y = f(x);
    let u = Tensor::from_vec(
        (0..y.num_elements()).map(|i| i as f32 - 1.5).collect(),
        y.shape(),
    );
    u.requires_grad_(true);
    let options = GradOptions {
        grad_outputs: vec![Some(u.clone())],
        create_graph: true,
        ..Default::default()
    };
    let g = autograd::grad(&[&y], &[x], options).remove(0).unwrap();
    assert!(g.requires_grad(), "{name}");
    // Where the gradient is `u` itself, a leaf, there is no graph to look at.
    if let Some(f) = g.grad_fn() {
        let names = graph_names(&f);
        assert!(!names.contains(&"NotImplemented"), "{name}: {names:?}");
    }
    let r = Tensor::from_vec(
        (0..g.num_elements()).map(|i| 0.5 + i as f32).collect(),
        g.shape(),
    );
    let gu = autograd::grad(&[&(&g * &r).sum()], &[&u], GradOptions::default()).remove(0);
    let expected = {
        let _no_grad = homura::no_grad();
        f(&r)
    };
    let gu = gu.unwrap();
    assert_eq!(gu.shape(), expected.shape(), "{name}");
    assert_close(&vals(&gu), &vals(&expected), name);
}

#[test]
fn formulas_record_their_own_backward_pass() {
    let x = leaf(&[1., -2., 3., 0.5, 4., -1.], &[2, 3]);
    let index = ints(&[2, 0, 2]);
    let index_2d = homura::tensor([[2i64, 2], [0, 1]]);
    check_double_backward("select", &x, |x| x.select(1, -1));
    check_double_backward("slice", &x, |x| x.index((.., (0..).step_by(2))));
    check_double_backward("narrow", &x, |x| x.narrow(1, 1, 2));
    check_double_backward("as_strided", &x, |x| x.as_strided([2, 2], [1, 1], 1));
    // Storage positions 2 and 5, both inside the strided view.
    check_double_backward("as_strided of a view", &x, |x| {
        x.index((.., (0..).step_by(2))).as_strided([2], [3], 2)
    });
    check_double_backward("index_select", &x, |x| x.index_select(1, &index));
    check_double_backward("integer index", &x, |x| x.index((.., &ints(&[-1, 0]))));
    check_double_backward("gather", &x, |x| x.gather(1, &index_2d));
    let zeros = homura::zeros([2, 4]);
    check_double_backward("index_add of self", &x, |x| {
        x.index_add(1, &index, &homura::zeros([2, 3]))
    });
    check_double_backward("index_add of source", &x, |x| {
        zeros.index_add(1, &ints(&[3, 0, 3]), x)
    });
    check_double_backward("scatter_add of self", &x, |x| {
        x.scatter_add(1, &index_2d, &homura::zeros([2, 2]))
    });
    // The index has the shape of `src`, which can't be larger to get a gradient.
    let index_like_x = homura::tensor([[3i64, 0, 3], [1, 1, 2]]);
    check_double_backward("scatter_add of src", &x, |x| {
        zeros.scatter_add(1, &index_like_x, x)
    });
    check_double_backward("select_scatter of self", &x, |x| {
        x.select_scatter(&homura::zeros([2]), 1, 0)
    });
    check_double_backward("select_scatter of src", &x, |x| {
        homura::zeros([2, 3, 2]).select_scatter(x, 2, 1)
    });
    check_double_backward("slice_scatter of self", &x, |x| {
        x.slice_scatter(&homura::zeros([2, 2]), 1, Some(1), None, 1)
    });
    check_double_backward("slice_scatter of src", &x, |x| {
        zeros.slice_scatter(x, 1, Some(-3), None, 1)
    });
    let cond = homura::tensor([[true, false, true], [false, false, true]]);
    let divisor = Tensor::from_slice(&[2f32, -4., 0.5], [3]);
    check_double_backward("where", &x, |x| where_(&cond, x, 0.0));
    check_double_backward("where of the other input", &x, |x| where_(&cond, 0.0, x));
    check_double_backward("div by a constant", &x, |x| x / &divisor);
    check_double_backward("div by a scalar", &x, |x| x / 4.0);
}

#[test]
fn second_derivatives_of_erf_and_sign() {
    // d/dx erf(x) = 2/√π e^(−x²), and d²/dx² erf(x) = −2x · 2/√π e^(−x²).
    for dtype in [DType::F32, DType::F64] {
        let x = Tensor::from_slice(&[0.5f32, -1.25], [2]).to(dtype);
        x.requires_grad_(true);
        let options = GradOptions {
            create_graph: true,
            ..Default::default()
        };
        let d1 = autograd::grad(&[&x.erf().sum()], &[&x], options)
            .remove(0)
            .unwrap();
        let d2 = autograd::grad(&[&d1.sum()], &[&x], GradOptions::default())
            .remove(0)
            .unwrap();
        assert_eq!((d1.dtype(), d2.dtype()), (dtype, dtype));
        assert_close(&vals(&d1), &[0.87878263, 0.23652114], "erf'");
        assert_close(&vals(&d2), &[-0.87878263, 0.5913029], "erf''");
    }
    // The gradient of `sign` is a constant zero: it has no graph to differentiate.
    let x = leaf(&[1., -2.], &[2]);
    let options = GradOptions {
        create_graph: true,
        ..Default::default()
    };
    let d1 = autograd::grad(&[&x.sign().sum()], &[&x], options)
        .remove(0)
        .unwrap();
    assert!(!d1.requires_grad());
    assert_eq!(vals(&d1), [0., 0.]);
}

#[test]
fn second_derivatives_of_unary_functions() {
    // The first and second derivatives at [0.5, -1.25, 2], or [0.5, 1.25, 2] where only
    // positive inputs are defined, from PyTorch.
    let cases: [(&str, [f32; 3], [f32; 3]); 13] = [
        ("abs", [1., -1., 1.], [0., 0., 0.]),
        ("relu", [1., 0., 1.], [0., 0., 0.]),
        ("log", [2., 0.8, 0.5], [-4., -0.64, -0.25]),
        (
            "sqrt",
            [0.70710677, 0.4472136, 0.35355338],
            [-0.70710677, -0.17888544, -0.08838835],
        ),
        (
            "rsqrt",
            [-1.4142134, -0.3577709, -0.1767767],
            [4.2426405, 0.42932504, 0.13258252],
        ),
        ("reciprocal", [-4., -0.64, -0.25], [16., -1.024, 0.25]),
        (
            "sin",
            [0.87758255, 0.31532237, -0.41614684],
            [-0.47942555, 0.9489846, -0.9092974],
        ),
        (
            "cos",
            [-0.47942555, 0.9489846, -0.9092974],
            [-0.87758255, -0.31532237, 0.41614684],
        ),
        (
            "tanh",
            [0.7864477, 0.28041488, 0.07065082],
            [-0.72686195, 0.4757427, -0.1362187],
        ),
        (
            "sigmoid",
            [0.23500371, 0.1731048, 0.10499359],
            [-0.05755679, 0.09600387, -0.0799625],
        ),
        (
            "silu",
            [0.7399612, 0.006319155, 1.0907842],
            [0.441229, 0.22620474, 0.05006217],
        ),
        (
            "gelu",
            [0.8674951, -0.12266158, 1.0852318],
            [0.6161143, 0.07990897, -0.10798193],
        ),
        (
            "gelu with tanh",
            [0.8673699, -0.12249263, 1.0860993],
            [0.6155069, 0.08118098, -0.10787508],
        ),
    ];
    let options = || GradOptions {
        create_graph: true,
        ..Default::default()
    };
    for dtype in [DType::F32, DType::F64] {
        for (name, first, second) in &cases {
            let positive = ["log", "sqrt", "rsqrt"].contains(name);
            let values = [0.5f32, if positive { 1.25 } else { -1.25 }, 2.];
            let x = Tensor::from_slice(&values, [3]).to(dtype);
            x.requires_grad_(true);
            let d1 = autograd::grad(&[&unary(name, &x).sum()], &[&x], options())
                .remove(0)
                .unwrap();
            // As in PyTorch, every first derivative depends on `x`, even where its derivative is
            // zero, like those of `abs` and `relu`.
            assert!(d1.requires_grad(), "{name}");
            let names = graph_names(&d1.grad_fn().unwrap());
            assert!(!names.contains(&"NotImplemented"), "{name}: {names:?}");
            let d2 = autograd::grad(&[&d1.sum()], &[&x], GradOptions::default())
                .remove(0)
                .unwrap();
            assert_eq!((d1.dtype(), d2.dtype()), (dtype, dtype), "{name}");
            assert_close(&vals(&d1), first, &format!("{name}' {dtype}"));
            assert_close(&vals(&d2), second, &format!("{name}'' {dtype}"));
        }
    }
}

#[test]
fn second_derivatives_of_div_and_where() {
    let a = leaf(&[1.5, -2.], &[2]);
    let b = leaf(&[0.5, 4.], &[2]);
    let options = || GradOptions {
        create_graph: true,
        allow_unused: true,
        ..Default::default()
    };
    // d/db (a / b) = -a / b², whose derivatives are -1 / b² and 2a / b³.
    let gb = autograd::grad(&[&(&a / &b).sum()], &[&b], options())
        .remove(0)
        .unwrap();
    assert_eq!(vals(&gb), [-6., 0.125]);
    let [gba, gbb] =
        <[Option<Tensor>; 2]>::try_from(autograd::grad(&[&gb.sum()], &[&a, &b], options()))
            .unwrap();
    assert_eq!(vals(&gba.unwrap()), [-4., -0.0625]);
    assert_eq!(vals(&gbb.unwrap()), [24., -0.0625]);
    // d/da (a / b) = 1 / b doesn't depend on `a`.
    let ga = autograd::grad(&[&(&a / &b).sum()], &[&a], options())
        .remove(0)
        .unwrap();
    let [gaa, gab] =
        <[Option<Tensor>; 2]>::try_from(autograd::grad(&[&ga.sum()], &[&a, &b], options()))
            .unwrap();
    assert!(gaa.is_none());
    assert_eq!(vals(&gab.unwrap()), [-4., -0.0625]);
    // A scalar divided by `x`: -2 / x², then 4 / x³.
    let x = leaf(&[0.5, -1.25, 2.], &[3]);
    let d1 = autograd::grad(&[&(2.0 / &x).sum()], &[&x], options())
        .remove(0)
        .unwrap();
    let d2 = autograd::grad(&[&d1.sum()], &[&x], GradOptions::default())
        .remove(0)
        .unwrap();
    assert_close(&vals(&d1), &[-8., -1.28, -0.5], "2 / x");
    assert_close(&vals(&d2), &[32., -2.048, 0.5], "2 / x");
    // The gradient of `where_` for a constant upstream gradient is a constant, as in PyTorch.
    let cond = homura::tensor([true, false]);
    let x = leaf(&[1., 2.], &[2]);
    let d1 = autograd::grad(&[&where_(&cond, &x, 0.0).sum()], &[&x], options())
        .remove(0)
        .unwrap();
    assert!(!d1.requires_grad());
    assert_eq!(vals(&d1), [1., 0.]);
}

// ---------------------------------------------------------------- saved tensors

#[test]
fn nodes_release_what_they_saved_like_pytorch() {
    type Case = (&'static str, Box<dyn Fn() -> Tensor>, bool);
    let index = || ints(&[1, 0]);
    let x = || leaf(&[1., 2., 3.], &[3]);
    let matrix = || leaf(&[1., 2., 3., 4., 5., 6.], &[3, 2]);
    // Whether a second backward pass without `retain_graph` fails: only nodes that saved a
    // tensor (an input or an index) refuse.
    let cases: Vec<Case> = vec![
        ("erf", Box::new(move || x().erf().sum()), true),
        ("sign", Box::new(move || x().sign().sum()), false),
        (
            "index_select",
            Box::new(move || x().index_select(0, &index()).sum()),
            true,
        ),
        (
            "integer index",
            Box::new(move || x().index(index()).sum()),
            true,
        ),
        (
            "select",
            Box::new(move || matrix().select(0, 1).sum()),
            false,
        ),
        ("slice", Box::new(move || matrix().index(1..).sum()), false),
        (
            "narrow",
            Box::new(move || matrix().narrow(0, 1, 2).sum()),
            false,
        ),
        (
            "as_strided",
            Box::new(move || x().as_strided([2], [2], 0).sum()),
            false,
        ),
        (
            "index_add of self",
            Box::new(move || x().index_add(0, &index(), &homura::ones([2])).sum()),
            false,
        ),
        (
            "index_add of source",
            Box::new(move || {
                homura::zeros([3])
                    .index_add(0, &index(), &leaf(&[1., 2.], &[2]))
                    .sum()
            }),
            true,
        ),
        (
            "gather",
            Box::new(move || x().gather(0, &index()).sum()),
            true,
        ),
        (
            "scatter_add of self",
            Box::new(move || x().scatter_add(0, &index(), &homura::ones([2])).sum()),
            false,
        ),
        (
            "scatter_add of src",
            Box::new(move || {
                homura::zeros([3])
                    .scatter_add(0, &index(), &leaf(&[1., 2.], &[2]))
                    .sum()
            }),
            true,
        ),
        (
            "select_scatter",
            Box::new(move || matrix().select_scatter(&leaf(&[1., 2.], &[2]), 0, 1).sum()),
            false,
        ),
        (
            "slice_scatter",
            Box::new(move || {
                matrix()
                    .slice_scatter(&leaf(&[1., 2.], &[1, 2]), 0, Some(1), Some(2), 1)
                    .sum()
            }),
            false,
        ),
    ];
    // Every elementwise formula of step 6a saves its input, its result, its divisor or its
    // condition, so a second pass through any of them fails. A scalar divisor is saved too, as
    // PyTorch saves the wrapped number.
    let mut cases = cases;
    for name in UNARY {
        cases.push((name, Box::new(move || unary(name, &x()).sum()), true));
    }
    let divisor = || Tensor::from_slice(&[2f32, 4., 8.], [3]);
    let cond = || homura::tensor([true, false, true]);
    let elementwise: Vec<Case> = vec![
        ("div", Box::new(move || (&x() / &x()).sum()), true),
        (
            "div by a constant",
            Box::new(move || (&x() / &divisor()).sum()),
            true,
        ),
        (
            "constant divided",
            Box::new(move || (&divisor() / &x()).sum()),
            true,
        ),
        (
            "div by a scalar",
            Box::new(move || (&x() / 2.0).sum()),
            true,
        ),
        ("scalar divided", Box::new(move || (2.0 / &x()).sum()), true),
        (
            "where",
            Box::new(move || where_(&cond(), x(), 0.0).sum()),
            true,
        ),
    ];
    cases.extend(elementwise);
    for (name, make, saves) in cases {
        let y = make();
        y.backward();
        match try_panic_message(|| y.backward()) {
            None => assert!(!saves, "{name}: the second pass should fail"),
            Some(message) => {
                assert!(saves, "{name}: the second pass should succeed");
                assert!(
                    message.starts_with("Trying to backward through the graph a second time"),
                    "{name}: {message}"
                );
            }
        }
        // Retained, every graph runs again.
        let y = make();
        let retain = BackwardOptions {
            retain_graph: Some(true),
            ..Default::default()
        };
        autograd::backward(&[&y], &[], retain);
        y.backward();
    }
}

#[test]
fn saved_indices_are_checked_for_in_place_changes() {
    let modified = |name: &str, saved: bool, make: &dyn Fn(&Tensor) -> Tensor| {
        let index = ints(&[1, 0]);
        let y = make(&index);
        {
            let _no_grad = homura::no_grad();
            index.add_(0);
        }
        match try_panic_message(|| y.sum().backward()) {
            None => assert!(!saved, "{name}: the modified index should be detected"),
            Some(message) => {
                assert!(saved, "{name}: the index isn't needed");
                assert!(
                    message.starts_with(
                        "one of the variables needed for gradient computation has been \
                         modified by an inplace operation: [i64 [2]] is at version 1; expected \
                         version 0 instead."
                    ),
                    "{name}: {message}"
                );
            }
        }
    };
    let x = || leaf(&[1., 2., 3.], &[3]);
    let source = || leaf(&[1., 2.], &[2]);
    modified("index_select", true, &|i| x().index_select(0, i));
    modified("integer index", true, &|i| x().index(i));
    modified("gather", true, &|i| x().gather(0, i));
    modified("index_add of source", true, &|i| {
        homura::zeros([3]).index_add(0, i, &source())
    });
    modified("index_add of self", false, &|i| {
        x().index_add(0, i, &homura::ones([2]))
    });
    modified("scatter_add of src", true, &|i| {
        homura::zeros([3]).scatter_add(0, i, &source())
    });
    modified("scatter_add of self", false, &|i| {
        x().scatter_add(0, i, &homura::ones([2]))
    });
}

#[test]
fn saved_operands_conditions_and_results_are_checked_for_in_place_changes() {
    let version_error = |what: &str| {
        format!(
            "one of the variables needed for gradient computation has been modified by an \
             inplace operation: {what} is at version 1; expected version 0 instead."
        )
    };
    let modify = |t: &Tensor| {
        let _no_grad = homura::no_grad();
        t.mul_(1);
    };
    // `a / b` saves `b`, and `a` when `b` needs a gradient. As in PyTorch, both are unpacked,
    // and so checked, even when only the gradient of `a`, which doesn't need `a`, is computed.
    for requested in 0..2 {
        let a = leaf(&[1., 2.], &[2]);
        let b = leaf(&[3., 4.], &[2]);
        let y = (&a / &b).sum();
        modify(&a);
        let inputs = [&a, &b];
        let message = panic_message(|| {
            autograd::grad(&[&y], &[inputs[requested]], GradOptions::default());
        });
        assert_eq!(message, version_error("[f32 [2]]"), "input {requested}");
    }
    // When `b` needs no gradient, `a` isn't saved and may change; `b` may not.
    let a = leaf(&[1., 2.], &[2]);
    let b = Tensor::from_slice(&[3f32, 4.], [2]);
    let y = (&a / &b).sum();
    modify(&a);
    y.backward();
    let y = (&a / &b).sum();
    modify(&b);
    assert_eq!(panic_message(|| y.backward()), version_error("[f32 [2]]"));
    // The condition of `where_`.
    let cond = homura::tensor([true, false]);
    let y = where_(&cond, leaf(&[1., 2.], &[2]), 0.0).sum();
    {
        let _no_grad = homura::no_grad();
        cond.copy_(&homura::tensor([false, true]));
    }
    assert_eq!(panic_message(|| y.backward()), version_error("[bool [2]]"));
    // Saved results name the operation that computed them, as PyTorch's messages do.
    let x = leaf(&[1., -2.], &[2]);
    let y = x.relu();
    modify(&y);
    assert_eq!(
        panic_message(|| y.sum().backward()),
        version_error("[f32 [2]], which is output 0 of Relu,")
    );
    let a = &x * 2.0;
    let y = (&a * &a).sum();
    modify(&a);
    assert_eq!(
        panic_message(|| y.backward()),
        version_error("[f32 [2]], which is output 0 of Mul,")
    );
}
