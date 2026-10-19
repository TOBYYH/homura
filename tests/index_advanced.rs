//! M2 step 6: advanced indexing with `Tensor::index` (several index tensors, bool masks, `bool`
//! items) and `Tensor::index_put` / `index_put_`. Outputs and gradients are compared with
//! `tests/golden/m2_index.safetensors` from `tools/golden/gen_m2_index.py`; the reference
//! behavior encoded in the other tests is asserted by `tools/golden/check_m2_index.py`, in the
//! same order.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Case, Golden};
use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::{DType, Device, Ellipsis, IndexItem, NewAxis, Tensor, TensorIndex, tensor};

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn assert_panics(f: impl FnOnce(), expected: &str) {
    let message = panic_message(f);
    assert!(
        message.contains(expected),
        "expected a panic with {expected:?}, got {message:?}"
    );
}

/// The node names of `t`'s graph, depth first.
fn chain(t: &Tensor) -> Vec<&'static str> {
    fn walk(f: &autograd::GradFn, out: &mut Vec<&'static str>) {
        out.push(f.name());
        for (next, _) in f.next_functions().into_iter().flatten() {
            walk(&next, out);
        }
    }
    let mut out = vec![];
    if let Some(f) = t.grad_fn() {
        walk(&f, &mut out);
    }
    out
}

/// A tensor of `F64` values (`homura::tensor` gives float literals the default dtype, `F32`).
fn f64t<D: homura::NestedData>(data: D) -> Tensor {
    homura::tensor_with(data, DType::F64)
}

fn ints(values: &[i64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn mask(values: &[bool]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

/// `0, 1, 2, ...` in `F64`, requiring grad.
fn counting(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let x = Tensor::from_vec((0..n).map(|i| i as f64).collect(), shape);
    x.requires_grad_(true);
    x
}

fn values(t: &Tensor) -> Vec<f64> {
    t.detach().to(DType::F64).to_vec::<f64>()
}

fn grads(out: &Tensor, inputs: &[&Tensor], g: &Tensor, create_graph: bool) -> Vec<Option<Tensor>> {
    autograd::grad(
        &[out],
        inputs,
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph,
            retain_graph: Some(true),
            allow_unused: true,
        },
    )
}

fn full() -> TensorIndex {
    (..).into_index()
}

/// The index expression of golden case `name` (without its dtype), from its index tensors `i`:
/// mirrors `CASES` in `tools/golden/gen_m2_index.py`.
fn spec(name: &str, i: &[Tensor]) -> Vec<TensorIndex> {
    let t = |k: usize| TensorIndex::Tensor(i[k].clone());
    match name.trim_end_matches("_acc") {
        "get_one"
        | "get_repeated_negative"
        | "get_mask"
        | "get_mask_2d"
        | "get_mask_all_false"
        | "get_scalar_int"
        | "put_one"
        | "put_scalar_values"
        | "put_mask_scalar"
        | "put_mask_rows"
        | "put_scalar_index"
        | "put_empty"
        | "put_repeated" => vec![t(0)],
        "get_adjacent" | "get_i32" | "put_adjacent" | "get_scalar_int_then_tensor" => {
            vec![full(), t(0), t(1)]
        }
        "get_non_adjacent"
        | "put_non_adjacent"
        | "put_broadcast_values"
        | "get_mask_then_slice" => vec![t(0), full(), t(1)],
        "get_int_between" => vec![full(), t(0), 1.into_index(), t(1)],
        "get_full_slice_between" => vec![full(), t(0), (0..2).into_index(), t(1)],
        "get_new_axis_between" => vec![full(), t(0), NewAxis.into_index(), t(1)],
        "get_zero_width_ellipsis" => vec![full(), t(0), Ellipsis.into_index(), t(1)],
        "get_ellipsis_between" => vec![t(0), Ellipsis.into_index(), t(1)],
        "get_broadcast" | "get_mask_broadcast" => vec![t(0), t(1)],
        "get_after_slice" => vec![(1..4).into_index(), full(), t(0)],
        "get_mask_later" | "get_empty_index" | "get_empty_other_dim" | "put_i32" => {
            vec![full(), t(0)]
        }
        "get_true" | "get_scalar_target_true" => vec![true.into_index()],
        "get_false" => vec![false.into_index()],
        "get_int_and_true" => vec![t(0), true.into_index()],
        "put_ellipsis" => vec![Ellipsis.into_index(), t(0)],
        other => panic!("unknown case {other}"),
    }
}

#[test]
fn golden_cases_match_pytorch() {
    let golden = Golden::load("m2_index.safetensors");
    let cases = golden.cases();
    assert!(cases.len() > 100, "{} cases", cases.len());
    for case in &cases {
        let c = Case::new(&golden, case);
        let name = case.split_once('.').unwrap().0;
        let names = golden.names(case);
        let has = |n: &str| names.iter().any(|m| m == n);
        let i: Vec<Tensor> = (0..)
            .map(|k| format!("i{k}"))
            .take_while(|n| has(n))
            .map(|n| c.input(&n))
            .collect();
        let x = c.input("x");
        x.requires_grad_(true);
        let v = has("v").then(|| c.input("v"));
        let out = match &v {
            None => x.index(spec(name, &i)),
            Some(v) => {
                v.requires_grad_(true);
                x.index_put(spec(name, &i), v, name.ends_with("_acc"))
            }
        };
        c.check("out", &out);
        let second = has("gg_x");
        let g = c.input("g");
        g.requires_grad_(second);
        let mut variables = vec![&x];
        variables.extend(&v);
        let first = grads(&out, &variables, &g, second);
        let first: Vec<Tensor> = first.into_iter().map(|t| t.expect("a gradient")).collect();
        let labels = ["x", "v", "g"];
        for (label, gr) in labels.iter().zip(&first) {
            c.check(&format!("grad_{label}"), gr);
        }
        if !second {
            c.assert_all_used();
            continue;
        }
        let total = first
            .iter()
            .zip(labels)
            .map(|(gr, label)| (gr * &c.input(&format!("gg_{label}"))).sum())
            .reduce(|a, b| &a + &b)
            .unwrap();
        variables.push(&g);
        let seconds = autograd::grad(
            &[&total],
            &variables,
            GradOptions {
                allow_unused: true,
                ..Default::default()
            },
        );
        let labels = [&labels[..first.len()], &["g"]].concat();
        for (label, s) in labels.iter().zip(seconds) {
            let key = format!("grad2_{label}");
            match s {
                Some(s) if has(&key) => c.check(&key, &s),
                None if !has(&key) => {}
                s => panic!("{case}: second derivative for {label}: {s:?}"),
            }
        }
        c.assert_all_used();
    }
}

// ---------------------------------------------------------------- positions and parsing

#[test]
fn result_positions_follow_pytorch() {
    let x = homura::zeros([4, 5, 6, 7]);
    let (i, j) = (ints(&[0, 1]), ints(&[2, 3]));
    let shape = |items: Vec<TensorIndex>| x.index(items).shape().to_vec();
    let t = |t: &Tensor| TensorIndex::Tensor(t.clone());
    assert_eq!(shape(vec![full(), t(&i), t(&j)]), [4, 2, 7]);
    assert_eq!(shape(vec![t(&i), full(), t(&j)]), [2, 5, 7]);
    assert_eq!(shape(vec![full(), t(&i), full(), t(&j)]), [2, 4, 6]);
    // An integer doesn't separate the tensors (NumPy gives [2, 4]); slices, even skipped full
    // ones, and new axes do.
    assert_eq!(shape(vec![full(), t(&i), 0.into_index(), t(&j)]), [4, 2]);
    assert_eq!(
        shape(vec![full(), t(&i), (0..6).into_index(), t(&j)]),
        [2, 4, 6]
    );
    assert_eq!(
        shape(vec![full(), t(&i), NewAxis.into_index(), t(&j)]),
        [2, 4, 1, 7]
    );
    // An ellipsis separates them only when it stands for dimensions.
    let x3 = homura::zeros([3, 4, 5]);
    let (a, b) = (ints(&[0, 1]), ints(&[1, 2]));
    assert_eq!(x3.index((.., &a, Ellipsis, &b)).shape(), [3, 2]);
    assert_eq!(x.index((&i, Ellipsis, &j)).shape(), [2, 5, 6]);
    // Tensors broadcast together.
    let i21 = Tensor::from_slice(&[0i64, 1], [2, 1]);
    assert_eq!(x.index((&i21, &ints(&[0, 1, 2]))).shape(), [2, 3, 6, 7]);
    // A mask covers as many dimensions as it has and counts as one entry: `x[m, :, j]`.
    let m2 = homura::ones_with([4, 5], DType::Bool);
    m2.index_put_((&ints(&[0]), &ints(&[0])), &tensor(false), false);
    assert_eq!(x.index((&m2, .., &ints(&[1]))).shape(), [19, 6]);
    // Basic items after the mask apply to the dimensions after the ones it covers.
    assert_eq!(x.index((&m2, 0)).shape(), [19, 7]);
    assert_eq!(x.index((&m2, NewAxis)).shape(), [19, 1, 6, 7]);
    assert_eq!(x.index((&m2, 1..3)).shape(), [19, 2, 7]);
    let m5 = mask(&[true, false, true, true, false]);
    assert_eq!(x.index((&i21, &m5)).shape(), [2, 3, 6, 7]);
    assert_eq!(x.index((.., &m5)).shape(), [4, 3, 6, 7]);
    assert_eq!(x.index(mask(&[false; 4])).shape(), [0, 5, 6, 7]);
    // `bool` items and zero-dimensional masks insert a dimension, then keep or empty it.
    assert_eq!(x.index(true).shape(), [1, 4, 5, 6, 7]);
    assert_eq!(x.index(false).shape(), [0, 4, 5, 6, 7]);
    assert_eq!(x.index(tensor(true)).shape(), [1, 4, 5, 6, 7]);
    assert_eq!(x.index((&i, true)).shape(), [2, 5, 6, 7]);
    assert_eq!(homura::zeros([]).index(true).shape(), [1]);
    // Empty indices give empty results.
    let empty = Tensor::from_slice::<i64>(&[], [0]);
    assert_eq!(
        homura::zeros([4, 5, 6, 7])
            .index((&ints(&[0]).index(..0), &empty))
            .shape(),
        [0, 6, 7]
    );
}

#[test]
fn index_records_one_node_after_the_views() {
    let x = counting(&[3, 4]);
    let i = ints(&[2, 0]);
    let acc = "AccumulateGrad";
    assert_eq!(chain(&x.index(&i)), ["IndexBackward0", acc]);
    assert_eq!(chain(&x.index((&i, ..))), ["IndexBackward0", acc]);
    assert_eq!(
        chain(&x.index((0, &i))),
        ["IndexBackward0", "SelectBackward0", acc]
    );
    assert_eq!(
        chain(&x.index((&i, &ints(&[1, 1])))),
        ["IndexBackward0", acc]
    );
    assert_eq!(
        chain(&x.index(true)),
        ["IndexBackward0", "UnsqueezeBackward0", acc]
    );
    assert_eq!(
        chain(&x.index((true, ..))),
        ["IndexBackward0", "UnsqueezeBackward0", acc]
    );
    assert_eq!(
        chain(&x.index(tensor(true))),
        ["IndexBackward0", "UnsqueezeBackward0", acc]
    );
    // A zero-dimensional integer tensor selects: a view.
    let picked = x.index(tensor(1i64));
    assert_eq!(chain(&picked), ["SelectBackward0", acc]);
    assert_eq!(picked.storage_offset(), 4);
    // index_put records one node with edges to the target and the values.
    let v = counting(&[2, 4]);
    let y = x.index_put((&i,), &v, false);
    assert_eq!(chain(&y), ["IndexPutBackward0", acc, acc]);
    let y = (&x * 1.0).index_put((&i,), &v.detach(), true);
    assert_eq!(chain(&y), ["IndexPutBackward0", "MulBackward0", acc]);
    let y = x.detach().index_put((&i,), &v, true);
    assert_eq!(chain(&y), ["IndexPutBackward0", acc]);
    let names: Vec<_> = y
        .grad_fn()
        .unwrap()
        .next_functions()
        .into_iter()
        .map(|e| e.map(|(f, _)| f.name()))
        .collect();
    assert_eq!(names, [None, Some(acc)]);
}

// ---------------------------------------------------------------- errors

#[test]
fn index_errors_follow_pytorch() {
    let x = homura::zeros([4, 5, 6, 7]);
    let i = ints(&[0, 1]);
    for dtype in [DType::I8, DType::I16, DType::F32] {
        for shape in [&[1][..], &[][..]] {
            let bad = homura::zeros_with(shape, dtype);
            assert_panics(
                || drop(x.index(&bad)),
                &format!("tensors used as indices must be i64, i32 or bool tensors, got {dtype}"),
            );
        }
    }
    assert_panics(
        || drop(x.index(homura::zeros_with([4], DType::U8))),
        "got u8; use a bool tensor for a mask",
    );
    assert_panics(
        || drop(x.index((.., &ints(&[9])))),
        "index 9 is out of bounds for dimension 0 with size 5",
    );
    assert_panics(
        || drop(x.index((&i, &ints(&[0, -6])))),
        "index -6 is out of bounds for dimension 1 with size 5",
    );
    assert_panics(
        || drop(x.index((&i, .., &ints(&[0, 6])))),
        "index 6 is out of bounds for dimension 1 with size 6",
    );
    assert_panics(
        || drop(homura::zeros([3, 0]).index((.., &ints(&[0])))),
        "index is out of bounds for dimension with size 0",
    );
    assert_panics(
        || drop(x.index((&i, &i, &i, &i, &i))),
        "too many indices for tensor of dimension 4",
    );
    assert_panics(
        || drop(homura::zeros([]).index(&i)),
        "too many indices for tensor of dimension 0",
    );
    assert_panics(
        || drop(homura::zeros([3]).index(homura::ones_with([3, 1], DType::Bool))),
        "too many indices for tensor of dimension 1",
    );
    assert_panics(
        || drop(x.index((&i, &ints(&[0, 1, 2])))),
        "shape mismatch: indexing tensors could not be broadcast together with shapes [2], [3]",
    );
    let m2 = homura::ones_with([4, 5], DType::Bool);
    assert_panics(
        || drop(x.index((&m2, &i))),
        "could not be broadcast together with shapes [20], [20], [2]",
    );
    assert_panics(
        || drop(x.index(homura::ones_with([3, 5], DType::Bool))),
        "The shape of the mask [3, 5] at index 0 does not match the shape of the indexed tensor \
         [4, 5, 6, 7] at index 0",
    );
    assert_panics(
        || drop(x.index((.., &homura::ones_with([5, 4], DType::Bool)))),
        "The shape of the mask [5, 4] at index 1 does not match the shape of the indexed tensor \
         [4, 5, 6, 7] at index 2",
    );
    // PyTorch's select message differs; homura's select reports its own.
    assert_panics(
        || drop(x.index(tensor(4i64))),
        "select(): index 4 out of range",
    );
}

/// Checks that `f`, an in-place operation on `x` that must fail with `expected`, changes neither
/// the data, the version nor the history of `x`, and leaves grad mode as it was.
fn assert_rejected(x: &Tensor, f: impl FnOnce(), expected: &str) {
    let before = values(x);
    let history = x.grad_fn();
    // A graph that saved `x`: its backward pass fails once `x` is modified in place.
    let w = homura::ones_with(x.shape(), x.dtype());
    w.requires_grad_(true);
    let probe = (&w * &x.detach()).sum();
    assert_panics(f, expected);
    assert_eq!(values(x), before);
    match (&history, x.grad_fn()) {
        (None, None) => {}
        (Some(a), Some(b)) => assert!(a.is_same(&b)),
        _ => panic!("history changed"),
    }
    assert!(homura::is_grad_enabled());
    autograd::grad(&[&probe], &[&w], GradOptions::default());
    if let Some(f) = history {
        // The old graph still runs.
        let _ = f.name();
        let retain = autograd::BackwardOptions {
            retain_graph: Some(true),
            ..Default::default()
        };
        autograd::backward(&[&x.sum()], &[None], retain);
    }
}

#[test]
fn index_put_errors_change_nothing() {
    let leaf = counting(&[4, 3]);
    let x = &leaf * 1.0;
    let i = ints(&[0, 2]);
    let v = homura::ones_with([2, 3], DType::F64);
    let put = |indices: Vec<TensorIndex>, v: &Tensor| {
        let (x, v) = (x.clone(), v.clone());
        move || {
            x.index_put_(indices, &v, false);
        }
    };
    let t = |t: &Tensor| TensorIndex::Tensor(t.clone());
    assert_rejected(
        &x,
        put(vec![t(&i)], &v.to(DType::F32)),
        "Index put requires the source and destination dtypes match, got f64 for the \
         destination and f32 for the source.",
    );
    assert_rejected(
        &x,
        put(vec![t(&i)], &homura::ones_with([1, 2, 3], DType::F64)),
        "shape mismatch: value tensor of shape [1, 2, 3] cannot be broadcast to indexing result \
         of shape [2, 3]",
    );
    assert_rejected(
        &x,
        put(vec![t(&i)], &homura::ones_with([3, 3], DType::F64)),
        "value tensor of shape [3, 3] cannot be broadcast to indexing result of shape [2, 3]",
    );
    assert_rejected(
        &x,
        put(vec![t(&ints(&[0, 4]))], &v),
        "index 4 is out of bounds for dimension 0 with size 4",
    );
    assert_rejected(
        &x,
        put(
            vec![t(&i)],
            &homura::ones_with([2, 3], (DType::F64, Device::Meta)),
        ),
        "Expected all tensors to be on the same device",
    );
    assert_rejected(
        &x,
        put(vec![t(&i.to(Device::Meta))], &v),
        "indices should be either on cpu or on the same device as the indexed tensor (cpu)",
    );
    assert_rejected(
        &x,
        put(vec![t(&i), t(&i), t(&i)], &v),
        "too many indices for tensor of dimension 2 (got 3)",
    );
    assert_rejected(
        &x,
        put(vec![0.into_index(), t(&i)], &v),
        "index_put(): indices can only be tensors, full slices (..) and Ellipsis",
    );
    assert_rejected(
        &x,
        put(vec![t(&i), (0..2).into_index()], &v),
        "index_put(): indices can only be tensors",
    );
    assert_rejected(
        &x,
        put(vec![NewAxis.into_index(), t(&i)], &v),
        "index_put(): indices can only be tensors",
    );
    assert_rejected(
        &x,
        put(vec![full(), Ellipsis.into_index()], &v),
        "index_put(): at least one index must be a tensor",
    );
    assert_rejected(&x, put(vec![], &v), "at least one index must be a tensor");
    assert_rejected(
        &x,
        put(vec![t(&tensor(true))], &v),
        "zero-dimensional bool masks are not supported",
    );
    assert_rejected(
        &x,
        put(vec![t(&homura::zeros_with([2], DType::U8))], &v),
        "tensors used as indices must be i64, i32 or bool tensors, got u8",
    );
    // Out-of-place calls check the same, except where noted below.
    assert_panics(
        || drop(x.index_put((&i,), &v.to(DType::F32), false)),
        "Index put requires the source and destination dtypes match",
    );
}

// ---------------------------------------------------------------- aliasing (R69)

#[test]
fn only_in_place_index_put_rejects_aliasing() {
    let overlap = "some elements of the input tensor and the written-to tensor refer to a single \
                   memory location";
    // Values that are the target or part of it.
    let x = homura::arange(4.0);
    let reverse = ints(&[3, 2, 1, 0]);
    assert_eq!(
        values(&x.index_put((&reverse,), &x, false)),
        [3.0, 2.0, 1.0, 0.0]
    );
    assert_eq!(values(&x), [0.0, 1.0, 2.0, 3.0]);
    assert_rejected(
        &x,
        || {
            x.index_put_((&reverse,), &x, false);
        },
        overlap,
    );
    let part = x.narrow(0, 0, 2);
    let i = ints(&[2, 3]);
    assert_eq!(
        values(&x.index_put((&i,), &part, true)),
        [0.0, 1.0, 2.0, 4.0]
    );
    assert_rejected(
        &x,
        || {
            x.index_put_((&i,), &part, true);
        },
        overlap,
    );
    // An integer target as its own index.
    let n = Tensor::from_slice(&[0i64, 1, 2], [3]);
    assert_eq!(
        n.index_put((&n,), &tensor(9i64), false).to_vec::<i64>(),
        [9, 9, 9]
    );
    assert_panics(
        || {
            n.index_put_((&n,), &tensor(9i64), false);
        },
        overlap,
    );
    assert_eq!(n.to_vec::<i64>(), [0, 1, 2]);
    // An expanded target: its elements share memory.
    let e = homura::ones([1]).expand([3]);
    assert_eq!(
        values(&e.index_put((&ints(&[1]),), &tensor(5.0), false)),
        [1.0, 5.0, 1.0]
    );
    assert_rejected(
        &e,
        || {
            e.index_put_((&ints(&[1]),), &tensor(5.0), false);
        },
        "more than one element of the written-to tensor may refer to a single memory location",
    );
    // Views of the same storage that don't touch the same bytes are fine; views that interleave
    // are rejected, conservatively.
    let base = homura::arange(6.0);
    base.narrow(0, 0, 3)
        .index_put_((&ints(&[0, 2]),), &base.narrow(0, 3, 2), false);
    assert_eq!(values(&base), [3.0, 1.0, 4.0, 3.0, 4.0, 5.0]);
    let even = base.index((0..).step_by(2));
    let odd = base.index((1..).step_by(2));
    assert_rejected(
        &even,
        || {
            even.index_put_((&ints(&[0, 1, 2]),), &odd, false);
        },
        overlap,
    );
    // The gradient of the out-of-place put accepts upstream gradients of any layout.
    let x = counting(&[3, 2]);
    let v = counting(&[2]);
    let y = x.index_put((&ints(&[1]),), &v, false);
    for g in [
        f64t(1.0f64).expand([3, 2]),
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).transpose(0, 1),
    ] {
        let r = grads(&y, &[&x, &v], &g, false);
        let gx = values(r[0].as_ref().unwrap());
        let gv = values(r[1].as_ref().unwrap());
        let gd = values(&g);
        assert_eq!(gx, [gd[0], gd[1], 0.0, 0.0, gd[4], gd[5]]);
        assert_eq!(gv, [gd[2], gd[3]]);
    }
}

// ---------------------------------------------------------------- layouts (R68)

/// `x` with `values` (`[len, x.shape[1]]`) written at `(rows[n], :, cols[n])` of the 3-D `x`, by
/// explicit loops.
fn put_loops(x: &Tensor, rows: &[usize], cols: &[usize], v: &Tensor, accumulate: bool) -> Vec<f64> {
    let s = x.shape();
    let mut out = values(&x.contiguous());
    let vals = values(v);
    for (n, (&r, &c)) in rows.iter().zip(cols).enumerate() {
        for k in 0..s[1] {
            let p = (r * s[1] + k) * s[2] + c;
            let value = vals[n * s[1] + k];
            out[p] = if accumulate { out[p] + value } else { value };
        }
    }
    out
}

#[test]
fn index_put_handles_any_target_layout() {
    let base = homura::arange(2.0 * 3.0 * 4.0 * 2.0).view([4, 3, 4]);
    let targets = [
        ("contiguous", homura::arange(48.0).view([4, 3, 4])),
        (
            "transposed",
            homura::arange(48.0).view([4, 4, 3]).transpose(1, 2),
        ),
        (
            "offset slice",
            homura::arange(60.0).view([5, 3, 4]).narrow(0, 1, 4),
        ),
        (
            "permuted",
            homura::arange(48.0).view([3, 4, 4]).permute([1, 0, 2]),
        ),
    ];
    let _ = base;
    let (rows, cols) = ([3usize, 0, 2], [1usize, 3, 0]);
    let (i, j) = (ints(&[3, 0, 2]), ints(&[1, 3, 0]));
    let v = Tensor::from_vec((0..9).map(|k| -1.0 - k as f32).collect(), [3, 3]);
    for (name, x) in &targets {
        for accumulate in [false, true] {
            let original = values(x);
            // Not adjacent: the broadcast dimension comes first, as `v` is laid out.
            let y = x.index_put((&i, .., &j), &v, accumulate);
            assert_eq!(y.shape(), [4, 3, 4]);
            assert_eq!(
                values(&y),
                put_loops(x, &rows, &cols, &v, accumulate),
                "{name}"
            );
            assert_eq!(values(x), original, "{name}");
            // Adjacent: the same positions through a transposed target.
            let xt = x.transpose(0, 1);
            let yt = xt.index_put((.., &i, &j), &v.transpose(0, 1), accumulate);
            assert_eq!(
                values(&yt.transpose(0, 1).contiguous()),
                values(&y),
                "{name}"
            );
            // In place, through a view of a larger base: the base sees the write.
            let big = homura::zeros([6, 3, 4]);
            let view = big.narrow(0, 1, 4);
            view.copy_(x);
            let probe_w = homura::ones([6, 3, 4]);
            probe_w.requires_grad_(true);
            let probe = (&probe_w * &big).sum();
            view.index_put_((&i, .., &j), &v, accumulate);
            let expected = put_loops(x, &rows, &cols, &v, accumulate);
            assert_eq!(values(&view), expected, "{name}");
            let whole = values(&big);
            assert!(whole[..12].iter().chain(&whole[60..]).all(|&e| e == 0.0));
            assert_panics(
                || {
                    drop(autograd::grad(
                        &[&probe],
                        &[&probe_w],
                        GradOptions::default(),
                    ))
                },
                "modified by an inplace operation",
            );
        }
    }
}

#[test]
fn in_place_index_put_writes_through_non_contiguous_targets() {
    // A stepped slice of a base with a non-zero offset, permuted: the write lands at the right
    // places of the base, and the elements between the slice's rows stay as they were.
    for adjacent in [false, true] {
        for accumulate in [false, true] {
            let base = homura::arange_with(120, DType::F64).view([6, 4, 5]);
            let target = base.index((1..6).step_by(2)).permute([2, 0, 1]);
            assert_eq!(target.shape(), [5, 3, 4]);
            assert!(!target.is_contiguous() && target.storage_offset() == 20);
            let (j, cols) = (ints(&[1, 3]), [1, 3]);
            // Target position `(a, b, c)` is base position `(1 + 2b, c, a)`.
            let (indices, v, at): (Vec<TensorIndex>, Tensor, Vec<[usize; 3]>) = if adjacent {
                let v = homura::arange_with(10, DType::F64)
                    .view([2, 5])
                    .transpose(0, 1);
                let at = (0..5)
                    .flat_map(|a| (0..2).map(move |n| [a, [2, 0][n], cols[n]]))
                    .collect();
                (
                    vec![full(), ints(&[2, 0]).into_index(), j.into_index()],
                    v,
                    at,
                )
            } else {
                let v = homura::arange_with(6, DType::F64)
                    .view([3, 2])
                    .transpose(0, 1);
                let at = (0..2)
                    .flat_map(|n| (0..3).map(move |b| [[4, 1][n], b, cols[n]]))
                    .collect();
                (
                    vec![ints(&[4, 1]).into_index(), full(), j.into_index()],
                    v,
                    at,
                )
            };
            let v = &v + 100.0;
            // `at` lists the written positions in the row-major order of `v`'s elements.
            let written = values(&v.contiguous());
            let mut expected = values(&base);
            for (k, [a, b, c]) in at.iter().enumerate() {
                let p = ((1 + 2 * b) * 4 + c) * 5 + a;
                expected[p] = if accumulate {
                    expected[p] + written[k]
                } else {
                    written[k]
                };
            }
            target.index_put_(indices, &v, accumulate);
            assert_eq!(
                values(&base),
                expected,
                "adjacent {adjacent}, accumulate {accumulate}"
            );
        }
    }
}

#[test]
fn advanced_indexing_results_are_new_tensors() {
    let x = homura::arange(24.0).view([2, 3, 4]);
    let i = ints(&[2, 0]);
    // The result is contiguous in the order the indexing works in; it doesn't keep the strides
    // of its input (PyTorch keeps them).
    let t = homura::arange(12.0).view([3, 4]).transpose(0, 1);
    let y = t.index_put((&ints(&[0]),), &tensor(9.0), false);
    assert_eq!((t.stride(), y.stride()), ([1, 4].into(), [3, 1].into()));
    let results = [
        x.index((.., &i)),
        x.index((&ints(&[1]), .., &ints(&[3]))),
        x.index(x.greater(5.0)),
        x.index(true),
        x.index_put((.., &i), &tensor(1.0), false),
        x.index_put((&ints(&[1]), .., &ints(&[3])), &tensor(1.0), true),
    ];
    for r in &results {
        // A later in-place write changes only the result.
        r.add_(100.0);
        assert_eq!(values(&x), (0..24).map(f64::from).collect::<Vec<_>>());
    }
    // Recorded results accept in-place operations as any non-view result does.
    let leaf = counting(&[2, 3]);
    let y = leaf.index((.., &i.index(..1)));
    y.mul_(2.0);
    let r = grads(&y, &[&leaf], &homura::ones_with([2, 1], DType::F64), false);
    assert_eq!(
        values(r[0].as_ref().unwrap()),
        [0.0, 0.0, 2.0, 0.0, 0.0, 2.0]
    );
}

// ---------------------------------------------------------------- saved index tensors

#[test]
fn saved_indices_are_version_checked_unless_consumed_by_value() {
    let modified = "modified by an inplace operation";
    // Masks and integer tensors are saved as given.
    let x = counting(&[3]);
    let m = mask(&[true, false, true]);
    let y = x.index(&m);
    m.fill_(false);
    assert_panics(
        || drop(grads(&y, &[&x], &homura::ones_with([2], DType::F64), false)),
        modified,
    );
    let i = ints(&[2, 0]);
    let y = x.index(&i);
    i.fill_(1);
    assert_panics(
        || drop(grads(&y, &[&x], &homura::ones_with([2], DType::F64), false)),
        modified,
    );
    // Zero-dimensional tensors in `index` are used up: changing them later is fine.
    for t in [tensor(1i64), tensor(true), Tensor::from_slice(&[1i32], [])] {
        let y = x.index(&t);
        t.fill_(0);
        let g = homura::ones_with(y.shape(), DType::F64);
        let r = grads(&y, &[&x], &g, false);
        let expected = if t.dtype() == DType::Bool {
            [1.0; 3]
        } else {
            [0.0, 1.0, 0.0]
        };
        assert_eq!(values(r[0].as_ref().unwrap()), expected);
    }
    // In index_put and index_put_, a zero-dimensional integer tensor is an index like any
    // other: saved and checked, whichever gradients a pass needs.
    for in_place in [false, true] {
        for accumulate in [false, true] {
            for dtype in [DType::I64, DType::I32] {
                let x = counting(&[3]);
                let v = counting(&[]);
                let i = Tensor::from_slice(&[1i64], []).to(dtype);
                let y = &x * 1.0;
                let y = if in_place {
                    y.index_put_((&i,), &v, accumulate);
                    y
                } else {
                    y.index_put((&i,), &v, accumulate)
                };
                assert_eq!(y.shape(), [3]);
                i.fill_(2);
                let g = homura::ones_with([3], DType::F64);
                for inputs in [&[&x, &v][..], &[&x][..], &[&v][..]] {
                    assert_panics(|| drop(grads(&y, inputs, &g, false)), modified);
                }
            }
        }
    }
    // The gradient of the values indexes the upstream gradient with the saved tensors again;
    // under create_graph that records an IndexBackward0 that saves them too.
    let x = counting(&[3]);
    let v = counting(&[]);
    let i = tensor(1i64);
    let y = x.index_put((&i,), &v, false);
    let g = homura::ones_with([3], DType::F64);
    g.requires_grad_(true);
    let gv = grads(&y, &[&v], &g, true).remove(0).unwrap();
    assert_eq!(chain(&gv)[0], "IndexBackward0");
    i.fill_(0);
    assert_panics(|| drop(grads(&gv, &[&g], &f64t(1.0f64), false)), modified);
}

#[test]
fn graphs_free_their_saved_indices() {
    let x = counting(&[4]);
    let y = x.index(mask(&[true, false, true, true]));
    let g = homura::ones_with([3], DType::F64);
    let retain = autograd::BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    };
    autograd::backward(&[&y], &[Some(&g)], retain);
    y.backward_with(&g);
    assert_eq!(values(&x.grad().unwrap()), [2.0, 0.0, 2.0, 2.0]);
    assert_panics(
        || y.backward_with(&g),
        "Trying to backward through the graph a second time",
    );
}

// ---------------------------------------------------------------- values and gradients

#[test]
fn accumulation_sums_in_f64_and_rounds_once() {
    // PyTorch rounds after each addition and gets 0 (see check_m2_index.py); homura's sum is
    // exact before the single rounding.
    for (dtype, start) in [
        (DType::F16, 2048.0),
        (DType::BF16, 256.0),
        (DType::F32, 16777216.0),
    ] {
        let x = tensor([start]).to(dtype);
        let v = tensor([1.0, -start]).to(dtype);
        let y = x.index_put((&ints(&[0, 0]),), &v, true);
        assert_eq!(values(&y), [1.0], "{dtype}");
    }
    let inf = f64::INFINITY;
    let x = Tensor::from_vec(vec![inf, 1.0, f64::NAN, 0.0], [4]);
    let v = Tensor::from_vec(vec![-inf, 2.0, 3.0, inf, inf], [5]);
    let y = x.index_put((&ints(&[0, 1, 2, 3, 3]),), &v, true);
    let r = values(&y);
    assert!(r[0].is_nan() && r[2].is_nan());
    assert_eq!((r[1], r[3]), (3.0, inf));
    // Integer sums wrap around, bool sums are a logical or.
    let n = Tensor::from_slice(&[i8::MAX], [1]);
    let y = n.index_put((&ints(&[0, 0]),), &Tensor::from_slice(&[1i8, 1], [2]), true);
    assert_eq!(y.to_vec::<i8>(), [i8::MIN + 1]);
    let b = mask(&[false, true, false]);
    let y = b.index_put((&ints(&[0, 0, 1]),), &mask(&[true, true, false]), true);
    assert_eq!(y.to_vec::<bool>(), [true, true, false]);
}

#[test]
fn overwriting_puts_zero_the_gradient_where_they_write() {
    // Repeated positions: one of the values is written (unspecified which), every one gets the
    // gradient of that position, and the target gets none there.
    let x = counting(&[3]);
    let v = Tensor::from_vec(vec![5.0, 6.0], [2]);
    v.requires_grad_(true);
    let y = x.index_put((&ints(&[1, 1]),), &v, false);
    assert!([5.0, 6.0].contains(&values(&y)[1]));
    let r = grads(&y, &[&x, &v], &f64t([1.0f64, 2.0, 3.0]), false);
    assert_eq!(values(r[0].as_ref().unwrap()), [1.0, 0.0, 3.0]);
    assert_eq!(values(r[1].as_ref().unwrap()), [2.0, 2.0]);
    // Written, not multiplied by zero: infinite and NaN gradients vanish there too.
    let v = counting(&[2]);
    let y = x.index_put((&ints(&[1, 2]),), &v, false);
    let g = Tensor::from_vec(vec![1.0, f64::INFINITY, f64::NAN], [3]);
    let r = grads(&y, &[&x, &v], &g, false);
    assert_eq!(values(r[0].as_ref().unwrap()), [1.0, 0.0, 0.0]);
    let gv = values(r[1].as_ref().unwrap());
    assert!(gv[0].is_infinite() && gv[1].is_nan());
}

#[test]
fn gradients_pass_gradcheck() {
    let options = GradcheckOptions::default;
    let check = |what: &str, f: &dyn Fn(&[Tensor]) -> Vec<Tensor>, inputs: &[&Tensor]| {
        gradcheck(f, inputs, options()).unwrap_or_else(|e| panic!("{what}: {e}"));
        gradgradcheck(f, inputs, options()).unwrap_or_else(|e| panic!("{what}: {e}"));
    };
    let x = counting(&[4, 3, 5]);
    let (i, j) = (ints(&[3, 0, 3]), ints(&[1, -1, 2]));
    let m = mask(&[true, false, true]);
    let i21 = Tensor::from_slice(&[0i64, 2], [2, 1]);
    check(
        "adjacent",
        &|t| vec![t[0].index((.., &ints(&[2, 0, 1]), &j))],
        &[&x],
    );
    check("non adjacent", &|t| vec![t[0].index((&i, .., &j))], &[&x]);
    check("mask", &|t| vec![t[0].index((.., &m))], &[&x]);
    check("mask broadcast", &|t| vec![t[0].index((&i21, &m))], &[&x]);
    check("true", &|t| vec![t[0].index((0, true, &j))], &[&x]);
    let v = counting(&[3, 3]);
    for accumulate in [false, true] {
        // Distinct positions: the overwriting put is smooth there.
        let k = ints(&[3, 0, 1]);
        check(
            "put",
            &|t| vec![t[0].index_put((&k, .., &j), &t[1], accumulate)],
            &[&x, &v],
        );
        let row = counting(&[3]);
        check(
            "put broadcast values",
            &|t| vec![t[0].index_put((&k, .., &j), &t[1], accumulate)],
            &[&x, &row],
        );
        let s = counting(&[]);
        check(
            "put mask",
            &|t| vec![t[0].index_put((.., &m), &t[1], accumulate)],
            &[&x, &s],
        );
    }
    check(
        "put repeated accumulate",
        &|t| vec![t[0].index_put((&ints(&[1, 1, 3]),), &t[1], true)],
        &[&x, &counting(&[3, 3, 5])],
    );
    // Values computed from the target itself: both paths to it get their gradients.
    let y = counting(&[2, 3]);
    let reversed = ints(&[1, 0]);
    for accumulate in [false, true] {
        check(
            "put values of self",
            &|t| {
                let put = t[0].index_put((&reversed,), &t[0].pow(2.0), accumulate);
                vec![put.index((.., &ints(&[2, 0]))).pow(2.0)]
            },
            &[&y],
        );
    }
}

#[test]
fn derivatives_connect_to_the_upstream_gradient() {
    // Indexing is linear: its gradient depends only on the upstream gradient `g`, its second
    // derivative only on the upstream gradient of the first, and so on.
    let x = counting(&[4]);
    let i = ints(&[1, 3, 1]);
    let y = x.index(&i);
    let g = f64t([1.0f64, 2.0, 4.0]);
    g.requires_grad_(true);
    let gx = grads(&y, &[&x], &g, true).remove(0).unwrap();
    assert_eq!(values(&gx), [0.0, 5.0, 0.0, 2.0]);
    assert_eq!(chain(&gx)[0], "IndexPutBackward0");
    let gg = f64t([1.0f64, 10.0, 100.0, 1000.0]);
    gg.requires_grad_(true);
    let r = grads(&gx, &[&x, &g], &gg, true);
    assert!(r[0].is_none());
    let dg = r[1].clone().unwrap();
    assert_eq!(values(&dg), [10.0, 1000.0, 10.0]);
    let w = f64t([1.0f64, 2.0, 3.0]);
    let third = grads(&dg, &[&gg], &w, false).remove(0).unwrap();
    assert_eq!(values(&third), [0.0, 4.0, 0.0, 2.0]);
    // index_put: the target's gradient and the values' both connect to `g`.
    let v = counting(&[2]);
    let y = x.index_put((&ints(&[0, 2]),), &v, false);
    let g = f64t([1.0f64, 2.0, 3.0, 4.0]);
    g.requires_grad_(true);
    let r = grads(&y, &[&x, &v], &g, true);
    let (gx, gv) = (r[0].clone().unwrap(), r[1].clone().unwrap());
    let total =
        (&gx * &f64t([1.0f64, 10.0, 100.0, 1000.0])).sum() + (&gv * &f64t([7.0f64, 9.0])).sum();
    let r = autograd::grad(
        &[&total],
        &[&x, &v, &g],
        GradOptions {
            allow_unused: true,
            ..Default::default()
        },
    );
    assert!(r[0].is_none() && r[1].is_none());
    assert_eq!(values(r[2].as_ref().unwrap()), [7.0, 10.0, 9.0, 1000.0]);
}

#[test]
fn degenerate_indices_keep_their_gradients() {
    let empty = Tensor::from_slice::<i64>(&[], [0]);
    type Indexing = Box<dyn Fn(&Tensor) -> Tensor>;
    let cases: Vec<(&str, Tensor, Indexing)> = vec![
        (
            "empty index",
            counting(&[3, 4]),
            Box::new(move |x| x.index((.., &empty))),
        ),
        (
            "all false",
            counting(&[3, 4]),
            Box::new(|x| x.index(mask(&[false; 3]))),
        ),
        (
            "empty dimension",
            counting(&[0, 4]),
            Box::new(|x| x.index((.., &ints(&[1, 3])))),
        ),
        (
            "whole",
            counting(&[3]),
            Box::new(|x| x.index(ints(&[2, 1, 0]))),
        ),
        ("false item", counting(&[3]), Box::new(|x| x.index(false))),
    ];
    for (name, x, f) in &cases {
        let y = f(x);
        let g = homura::ones_with(y.shape(), DType::F64);
        g.requires_grad_(true);
        let gx = grads(&y, &[x], &g, true).remove(0).unwrap();
        assert_eq!(gx.shape(), x.shape(), "{name}");
        assert_eq!(
            values(&gx).iter().sum::<f64>(),
            y.num_elements() as f64,
            "{name}"
        );
        let gg = homura::ones_with(x.shape(), DType::F64);
        let dg = grads(&gx, &[&g], &gg, false).remove(0).unwrap();
        assert_eq!(dg.shape(), y.shape(), "{name}");
    }
    // Scalar values and whole coverage in index_put.
    let x = counting(&[3]);
    let s = counting(&[]);
    let y = x.index_put((&ints(&[0, 1, 2]),), &s, false);
    let r = grads(&y, &[&x, &s], &f64t([1.0f64, 2.0, 3.0]), false);
    assert_eq!(values(r[0].as_ref().unwrap()), [0.0; 3]);
    assert_eq!(values(r[1].as_ref().unwrap()), [6.0]);
}

// ---------------------------------------------------------------- meta tensors (R70)

#[test]
fn meta_targets_index_by_shape() {
    let x = homura::zeros_with([4, 5, 6], Device::Meta);
    x.requires_grad_(true);
    let meta_i = homura::zeros_with([2], (DType::I64, Device::Meta));
    let cpu_j = Tensor::from_slice(&[0i64, 1, 2], [3, 1]);
    // Meta integer indices are not checked; CPU ones are, unlike PyTorch.
    assert_eq!(x.index((&meta_i, &cpu_j)).shape(), [3, 2, 6]);
    assert_panics(
        || drop(x.index((.., &ints(&[5])))),
        "index 5 is out of bounds for dimension 0 with size 5",
    );
    // CPU masks work on meta targets, with both gradients.
    let m = mask(&[true, false, true, true, false]);
    let y = x.index((.., &m));
    assert_eq!(y.shape(), [4, 3, 6]);
    let v = homura::zeros_with([3, 6], Device::Meta);
    v.requires_grad_(true);
    for accumulate in [false, true] {
        let y = x.index_put((.., &m), &v, accumulate);
        let g = homura::zeros_with([4, 5, 6], Device::Meta);
        let r = grads(&y, &[&x, &v], &g, false);
        assert_eq!(r[0].as_ref().unwrap().shape(), [4, 5, 6]);
        assert_eq!(r[1].as_ref().unwrap().shape(), [3, 6]);
    }
    // Meta masks and zero-dimensional meta tensors whose value is needed are rejected.
    let meta_mask = homura::zeros_with([5], (DType::Bool, Device::Meta));
    let rejected = "on the meta device is not supported";
    assert_panics(|| drop(x.index((.., &meta_mask))), rejected);
    assert_panics(|| drop(x.index_put((.., &meta_mask), &v, false)), rejected);
    assert_panics(
        || {
            x.detach().index_put_((.., &meta_mask), &v.detach(), false);
        },
        rejected,
    );
    assert_panics(
        || drop(x.index(homura::zeros_with([], (DType::Bool, Device::Meta)))),
        rejected,
    );
    let meta_scalar = homura::zeros_with([], (DType::I64, Device::Meta));
    assert_panics(|| drop(x.index(&meta_scalar)), rejected);
    // index_put keeps a zero-dimensional integer index as an index: shapes only, with first and
    // upstream second derivatives.
    let x1 = homura::zeros_with([3], Device::Meta);
    x1.requires_grad_(true);
    let s = homura::zeros_with([], Device::Meta);
    s.requires_grad_(true);
    for accumulate in [false, true] {
        let y = (&x1 * 1.0).index_put((&meta_scalar,), &s, accumulate);
        assert_eq!(y.shape(), [3]);
        let g = homura::zeros_with([3], Device::Meta);
        g.requires_grad_(true);
        let r = grads(&y, &[&x1, &s], &g, true);
        assert_eq!(r[0].as_ref().unwrap().shape(), [3]);
        let gs = r[1].clone().unwrap();
        assert_eq!(gs.shape(), []);
        let dg = grads(&gs, &[&g], &homura::zeros_with([], Device::Meta), false);
        assert_eq!(dg[0].as_ref().unwrap().shape(), [3]);
        let z = (&x1 * 1.0).detach();
        z.index_put_((&meta_scalar,), &s.detach(), accumulate);
    }
    // Large logical sizes allocate nothing on the host.
    let huge = homura::zeros_with([1 << 20, 1 << 20], Device::Meta);
    let column = homura::zeros_with([1 << 16, 1], DType::I64);
    let row = homura::zeros_with([1 << 16], DType::I64);
    assert_eq!(huge.index((&column, &row)).shape(), [1 << 16, 1 << 16]);
    // Values are on the device of the target (PyTorch also takes CPU scalars).
    assert_panics(
        || drop(x.index_put((&ints(&[0]),), &tensor(1.0), false)),
        "Expected all tensors to be on the same device",
    );
}

#[test]
fn bool_items_ignore_the_default_device() {
    let x = homura::arange(3.0);
    let i = ints(&[2]);
    let _meta = homura::device_scope(Device::Meta);
    let TensorIndex::Tensor(t) = true.into_index() else {
        unreachable!()
    };
    assert_eq!(t.device(), Device::Cpu);
    assert_eq!(values(&x.index(true)), [0.0, 1.0, 2.0]);
    assert_eq!(x.index((false, ..)).shape(), [0, 3]);
    assert_eq!(values(&x.index((&i, true))), [2.0]);
}

// ---------------------------------------------------------------- in-place rules

#[test]
fn index_put_follows_the_in_place_rules() {
    let i = ints(&[0, 2]);
    let leaf = counting(&[3]);
    assert_rejected(
        &leaf,
        || {
            leaf.index_put_((&i,), &f64t([1.0f64, 2.0]), false);
        },
        "a leaf Variable that requires grad is being used in an in-place operation.",
    );
    let y = &leaf * 1.0;
    let view = y.narrow(0, 0, 3);
    assert_panics(
        || {
            view.index_put_((&i,), &f64t([1.0f64, 2.0]), false);
        },
        "in-place operations on a view are not supported yet",
    );
    // In no-grad mode, parameters can be updated, also through views.
    {
        let _no_grad = homura::no_grad();
        leaf.index_put_((&i,), &f64t([10.0f64, 20.0]), true);
        leaf.index(0..2)
            .index_put_((&ints(&[1]),), &f64t(-1.0f64), false);
    }
    assert_eq!(values(&leaf), [10.0, -1.0, 22.0]);
    // A target that doesn't require grad joins the graph when the values do.
    let buffer = homura::zeros_with([3], DType::F64);
    let v = counting(&[2]);
    buffer.index_put_((&i,), &v, false);
    assert_eq!(chain(&buffer), ["IndexPutBackward0", "AccumulateGrad"]);
    let r = grads(&buffer, &[&v], &f64t([1.0f64, 2.0, 3.0]), false);
    assert_eq!(values(r[0].as_ref().unwrap()), [1.0, 3.0]);
    // A recorded write into a computed tensor: its old history gets the masked gradient.
    let x = counting(&[3]);
    let y = &x * 2.0;
    y.index_put_((&ints(&[1]),), &v.index(0..1), false);
    assert_eq!(chain(&y)[..2], ["IndexPutBackward0", "MulBackward0"]);
    let r = grads(&y, &[&x, &v], &f64t([1.0f64, 1.0, 1.0]), false);
    assert_eq!(values(r[0].as_ref().unwrap()), [2.0, 0.0, 2.0]);
    assert_eq!(values(r[1].as_ref().unwrap()), [1.0, 0.0]);
}
