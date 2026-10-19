//! View and indexing gradients: the nodes of `permute`, `transpose`, the squeezes and
//! `unsqueeze` and their second derivatives, the full slices that indexing skips (pinned PyTorch
//! node chains), view identity, base and grad mode of the new views, metadata-only backward
//! passes and empty inputs.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradFn, GradOptions};
use homura::{Device, Ellipsis, NewAxis, Tensor, TensorIndex};

const NO_DIMS: [i64; 0] = [];

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

fn grad_with(out: &Tensor, x: &Tensor, g: &Tensor, graph: bool) -> Tensor {
    let options = GradOptions {
        grad_outputs: vec![Some(g.clone())],
        create_graph: graph,
        ..Default::default()
    };
    autograd::grad(&[out], &[x], options).remove(0).unwrap()
}

/// The upstream gradient `1, 2, 3, ...` of the shape of `out`.
fn upstream(out: &Tensor) -> Tensor {
    let n = out.num_elements();
    Tensor::from_vec((1..=n).map(|i| i as f64).collect(), out.shape())
}

fn panic_message(f: impl FnOnce()) -> String {
    let p = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}

/// The names of the nodes from `t`'s `grad_fn` along the first edges, up to the leaf's
/// `AccumulateGrad` (left out).
fn chain(t: &Tensor) -> Vec<&'static str> {
    let mut names = vec![];
    let mut next = t.grad_fn();
    while let Some(f) = next {
        if f.name() == "AccumulateGrad" {
            break;
        }
        names.push(f.name());
        next = f
            .next_functions()
            .into_iter()
            .next()
            .flatten()
            .map(|(f, _)| f);
    }
    names
}

fn edge_names(f: &GradFn) -> Vec<Option<&'static str>> {
    f.next_functions()
        .iter()
        .map(|e| e.as_ref().map(|(f, _)| f.name()))
        .collect()
}

/// The element of a row-major tensor of `shape` at `index`.
fn at(values: &[f64], shape: &[usize], index: &[usize]) -> f64 {
    let flat = index.iter().zip(shape).fold(0, |acc, (&i, &n)| acc * n + i);
    values[flat]
}

/// Every index of a tensor of `shape`, in row-major order.
fn indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    for &n in shape {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                (0..n).map(move |i| {
                    let mut p = prefix.clone();
                    p.push(i);
                    p
                })
            })
            .collect();
    }
    all
}

// ---------------------------------------------------------------- formulas

#[test]
fn permute_and_transpose_send_each_gradient_element_back() {
    let x = counting(&[2, 3, 4]);
    // Element `i` of the result is element `source(i)` of `x`.
    type Case = (&'static str, Tensor, fn(&[usize]) -> Vec<usize>);
    let cases: [Case; 5] = [
        ("PermuteBackward0", x.permute([2, 0, 1]), |i| {
            vec![i[1], i[2], i[0]]
        }),
        ("PermuteBackward0", x.permute([-1, -3, 1]), |i| {
            vec![i[1], i[2], i[0]]
        }),
        ("TransposeBackward0", x.transpose(0, 2), |i| {
            vec![i[2], i[1], i[0]]
        }),
        ("TransposeBackward0", x.transpose(-1, 1), |i| {
            vec![i[0], i[2], i[1]]
        }),
        ("TransposeBackward0", x.transpose(1, 1), |i| i.to_vec()),
    ];
    for (name, out, source) in cases {
        let f = out.grad_fn().unwrap();
        assert_eq!(f.name(), name);
        assert_eq!(edge_names(&f), [Some("AccumulateGrad")]);
        let g = upstream(&out);
        let gx = grad_with(&out, &x, &g, false);
        assert_eq!(gx.shape(), [2, 3, 4]);
        let (gv, gx) = (g.to_vec::<f64>(), gx.to_vec::<f64>());
        let mut expected = vec![0.0; 24];
        for i in indices(&out.shape()) {
            let s = source(&i);
            expected[(s[0] * 3 + s[1]) * 4 + s[2]] = at(&gv, &out.shape(), &i);
        }
        assert_eq!(gx, expected, "{name}");
    }
}

#[test]
fn squeezes_insert_the_removed_dimensions_back() {
    let x = counting(&[2, 1, 3, 1]);
    let cases = [
        ("SqueezeBackward0", x.squeeze(), vec![2, 3]),
        ("SqueezeBackward1", x.squeeze_dim(1), vec![2, 3, 1]),
        ("SqueezeBackward1", x.squeeze_dim(-1), vec![2, 1, 3]),
        // Dimensions of other sizes are left alone.
        ("SqueezeBackward1", x.squeeze_dim(0), vec![2, 1, 3, 1]),
        ("SqueezeBackward2", x.squeeze_dim([1]), vec![2, 3, 1]),
        ("SqueezeBackward2", x.squeeze_dim([-1, 1]), vec![2, 3]),
        ("SqueezeBackward2", x.squeeze_dim([0, 2]), vec![2, 1, 3, 1]),
        // As in PyTorch, an empty list squeezes nothing.
        ("SqueezeBackward2", x.squeeze_dim(NO_DIMS), vec![2, 1, 3, 1]),
        ("UnsqueezeBackward0", x.unsqueeze(-1), vec![2, 1, 3, 1, 1]),
        ("UnsqueezeBackward0", x.unsqueeze(2), vec![2, 1, 1, 3, 1]),
    ];
    for (name, out, shape) in cases {
        assert_eq!(out.shape(), shape, "{name}");
        let f = out.grad_fn().unwrap();
        assert_eq!(f.name(), name);
        assert_eq!(edge_names(&f), [Some("AccumulateGrad")]);
        let g = upstream(&out);
        let gx = grad_with(&out, &x, &g, false);
        assert_eq!(gx.shape(), [2, 1, 3, 1], "{name} {shape:?}");
        assert_eq!(gx.to_vec::<f64>(), g.to_vec::<f64>(), "{name} {shape:?}");
    }
    assert_eq!(
        panic_message(|| {
            x.squeeze_dim([1, -3]);
        }),
        "dim 1 appears multiple times in the list of dims"
    );
}

#[test]
fn zero_dimensional_views() {
    let s = leaf(&[2.0], []);
    let cases = [
        ("SqueezeBackward0", s.squeeze()),
        ("SqueezeBackward1", s.squeeze_dim(0)),
        ("SqueezeBackward1", s.squeeze_dim(-1)),
        ("SqueezeBackward2", s.squeeze_dim([0])),
        ("UnsqueezeBackward0", s.unsqueeze(0)),
        ("UnsqueezeBackward0", s.unsqueeze(-1)),
        ("TransposeBackward0", s.transpose(0, -1)),
        ("PermuteBackward0", s.permute(NO_DIMS)),
    ];
    for (name, out) in cases {
        assert_eq!(out.grad_fn().unwrap().name(), name);
        let g = Tensor::from_slice(&[3.0], out.shape());
        let gs = grad_with(&out, &s, &g, false);
        assert_eq!(gs.shape(), [], "{name}");
        assert_eq!(gs.to_vec::<f64>(), [3.0], "{name}");
    }
}

#[test]
fn second_derivatives_go_through_the_inverse_views() {
    // The gradient of each view is computed by the inverse view, whose own node is recorded with
    // `create_graph`; its derivative with respect to the upstream gradient is the inverse again.
    let x = counting(&[2, 1, 3, 1]);
    type Case = (fn(&Tensor) -> Tensor, &'static [&'static str]);
    let cases: [Case; 7] = [
        (
            |x| x.squeeze(),
            &["UnsqueezeBackward0", "UnsqueezeBackward0"],
        ),
        (|x| x.squeeze_dim(1), &["UnsqueezeBackward0"]),
        (
            |x| x.squeeze_dim([1, 3]),
            &["UnsqueezeBackward0", "UnsqueezeBackward0"],
        ),
        (|x| x.unsqueeze(-1), &["SqueezeBackward1"]),
        (|x| x.permute([2, 0, 3, 1]), &["PermuteBackward0"]),
        (|x| x.transpose(0, 2), &["TransposeBackward0"]),
        // The alias passes the upstream gradient itself through.
        (|x| x.index((.., .., ..)), &[]),
    ];
    for (op, expected) in cases {
        let out = op(&x);
        let g = upstream(&out);
        g.requires_grad_(true);
        let gx = grad_with(&out, &x, &g, true);
        assert_eq!(chain(&gx), expected, "{:?}", out.grad_fn().unwrap().name());
        // d(gx)/dg applied to an upstream gradient of the shape of x: the forward view.
        let h = upstream(&gx);
        let gg = grad_with(&gx, &g, &h, false);
        let viewed = op(&h.detach());
        assert_eq!(gg.shape(), viewed.shape());
        assert_eq!(gg.to_vec::<f64>(), viewed.contiguous().to_vec::<f64>());
    }
    // A squeeze that removes nothing passes the gradient through untouched, as in PyTorch.
    let out = x.squeeze_dim(0);
    let g = upstream(&out);
    g.requires_grad_(true);
    let gx = grad_with(&out, &x, &g, true);
    assert!(gx.grad_fn().is_none() && gx.requires_grad());
    assert_eq!(gx.to_vec::<f64>(), g.to_vec::<f64>());
}

#[test]
fn views_save_nothing_so_backward_passes_can_repeat() {
    let x = counting(&[2, 1, 3]);
    let outputs = [
        x.permute([2, 0, 1]),
        x.transpose(0, 1),
        x.squeeze(),
        x.squeeze_dim(1),
        x.squeeze_dim([1]),
        x.unsqueeze(0),
        x.index((.., ..)),
    ];
    for out in outputs {
        let y = out.sum();
        y.backward();
        y.backward();
    }
    assert_eq!(x.grad().unwrap().to_vec::<f64>(), [14.0; 6]);
}

// ---------------------------------------------------------------- indexing

#[test]
fn indexing_skips_full_slices_like_pytorch() {
    // Node chains measured on the pinned PyTorch: among several items, a slice starting at 0
    // with a step of 1 and an end at or past the size is skipped; a lone slice is recorded.
    let x = counting(&[2, 3, 4]);
    let idx = homura::tensor([1i64, 0]);
    let index = &["IndexBackward0"][..];
    let cases: Vec<(&str, Tensor, &[&str])> = vec![
        ("x[:]", x.index(..), &["SliceBackward0"]),
        ("x[0:2]", x.index(0..2), &["SliceBackward0"]),
        ("x[:, :]", x.index((.., ..)), &["AliasBackward0"]),
        ("x[:, :, :]", x.index((.., .., ..)), &["AliasBackward0"]),
        ("x[...]", x.index(Ellipsis), &["AliasBackward0"]),
        (
            "x[()]",
            x.index(Vec::<TensorIndex>::new()),
            &["AliasBackward0"],
        ),
        ("x[..., :]", x.index((Ellipsis, ..)), &["AliasBackward0"]),
        ("x[:, 0:100]", x.index((.., 0..100)), &["AliasBackward0"]),
        ("x[:, 0:3]", x.index((.., 0..3)), &["AliasBackward0"]),
        (
            "x[:, 0:3:1]",
            x.index((.., (0..3).step_by(1))),
            &["AliasBackward0"],
        ),
        ("x[:, 0:=2]", x.index((.., 0..=2)), &["AliasBackward0"]),
        ("x[:, :=-1]", x.index((.., ..=-1)), &["AliasBackward0"]),
        ("x[:, -3:]", x.index((.., -3..)), &["SliceBackward0"]),
        (
            "x[:, ::2]",
            x.index((.., (0..).step_by(2))),
            &["SliceBackward0"],
        ),
        ("x[:, 0:2]", x.index((.., 0..2)), &["SliceBackward0"]),
        ("x[1:, :]", x.index((1.., ..)), &["SliceBackward0"]),
        ("x[:, 1:]", x.index((.., 1..)), &["SliceBackward0"]),
        ("x[:, 1]", x.index((.., 1)), &["SelectBackward0"]),
        ("x[0, :]", x.index((0, ..)), &["SelectBackward0"]),
        ("x[..., 1]", x.index((Ellipsis, 1)), &["SelectBackward0"]),
        (
            "x[:, None]",
            x.index((.., NewAxis)),
            &["UnsqueezeBackward0"],
        ),
        (
            "x[None, :]",
            x.index((NewAxis, ..)),
            &["UnsqueezeBackward0"],
        ),
        ("x[None]", x.index(NewAxis), &["UnsqueezeBackward0"]),
        (
            "x[:, :, None, :]",
            x.index((.., .., NewAxis, ..)),
            &["UnsqueezeBackward0"],
        ),
        (
            "x[:, 1:, None]",
            x.index((.., 1.., NewAxis)),
            &["UnsqueezeBackward0", "SliceBackward0"],
        ),
        ("x[:, idx]", x.index((.., &idx)), index),
        ("x[idx, :]", x.index((&idx, ..)), index),
        ("x[idx]", x.index(&idx), index),
        ("x[:, :, idx]", x.index((.., .., &idx)), index),
    ];
    for (name, out, expected) in cases {
        assert_eq!(chain(&out), expected, "{name}");
        assert!(!out.is_same(&x), "{name}");
        // `x` holds 0, 1, 2, ...: each element of the result names the element of `x` it
        // came from, which gets the upstream gradient of that position.
        let g = upstream(&out);
        let gx = grad_with(&out, &x, &g, false);
        let mut expected = vec![0.0; x.num_elements()];
        let picked = out.detach().contiguous().to_vec::<f64>();
        for (&p, &gi) in picked.iter().zip(&g.to_vec::<f64>()) {
            expected[p as usize] += gi;
        }
        assert_eq!(gx.to_vec::<f64>(), expected, "{name}");
    }
    let e = counting(&[2, 0]);
    assert_eq!(chain(&e.index((.., 0..0))), ["AliasBackward0"]);
    assert_eq!(chain(&e.index((.., ..))), ["AliasBackward0"]);
    assert_eq!(chain(&e.index((0..0, ..))), ["SliceBackward0"]);
    let s = leaf(&[2.0], []);
    assert_eq!(chain(&s.index(Ellipsis)), ["AliasBackward0"]);
    assert_eq!(chain(&s.index(NewAxis)), ["UnsqueezeBackward0"]);
    assert_eq!(
        chain(&s.index(Vec::<TensorIndex>::new())),
        ["AliasBackward0"]
    );
    // Review round 20: a `step_by` range with a single position becomes a slice with step 1,
    // which covers a dimension of size 1 and is skipped; Python's `x[:, 0:1:2]` and an explicit
    // slice with step 2 are recorded. Values and gradients are the same either way.
    let column = counting(&[2, 1]);
    let short = column.index((.., (0..1).step_by(2)));
    assert_eq!(chain(&short), ["AliasBackward0"]);
    let full = TensorIndex::Slice {
        start: None,
        end: None,
        step: 1,
    };
    let step = TensorIndex::Slice {
        start: Some(0),
        end: Some(1),
        step: 2,
    };
    let explicit = column.index(vec![full, step]);
    assert_eq!(chain(&explicit), ["SliceBackward0"]);
    for out in [&short, &explicit] {
        let gx = grad_with(out, &column, &upstream(out), false);
        assert_eq!(gx.to_vec::<f64>(), [1.0, 2.0]);
    }
}

#[test]
fn skipped_slices_keep_the_gradients() {
    let x = counting(&[2, 3]);
    let y = &x * 2.0;
    let out = y.index((.., .., NewAxis));
    assert_eq!(chain(&out), ["UnsqueezeBackward0", "MulBackward0"]);
    let out = y.index((.., ..));
    assert_eq!(chain(&out), ["AliasBackward0", "MulBackward0"]);
    out.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec::<f64>(), [2.0; 6]);
    // Gradients of skipped slices before a tensor index and a select.
    let gx = grad_with(
        &x.index((.., &homura::tensor([2i64, 2, 0]))),
        &x,
        &Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        false,
    );
    assert_eq!(gx.to_vec::<f64>(), [3.0, 0.0, 3.0, 6.0, 0.0, 9.0]);
    let gx = grad_with(
        &x.index((0..10, 1)),
        &x,
        &Tensor::from_slice(&[7.0, 8.0], [2]),
        false,
    );
    assert_eq!(gx.to_vec::<f64>(), [0.0, 7.0, 0.0, 0.0, 8.0, 0.0]);
}

// ---------------------------------------------------------------- views and grad mode

/// The views of `x` (shape `[2, 1, 3]`) that 6d connects, by name.
fn new_views(x: &Tensor) -> Vec<(&'static str, Tensor)> {
    vec![
        ("permute", x.permute([2, 0, 1])),
        ("transpose", x.transpose(0, 2)),
        ("squeeze", x.squeeze()),
        ("squeeze_dim", x.squeeze_dim(1)),
        ("squeeze_dims", x.squeeze_dim([1])),
        ("unsqueeze", x.unsqueeze(0)),
        ("alias", x.index((.., ..))),
        ("ellipsis", x.index(Ellipsis)),
    ]
}

#[test]
fn new_views_are_views_with_their_own_node() {
    let x = counting(&[2, 1, 3]);
    for (name, v) in new_views(&x) {
        assert!(!v.is_same(&x), "{name}");
        assert!(v.requires_grad() && !v.is_leaf(), "{name}");
        // A view shares the storage: a write in no-grad mode shows through `x`.
        {
            let _no_grad = homura::no_grad();
            v.index(Ellipsis).fill_(5.0);
        }
        assert!(x.to_vec::<f64>().iter().all(|&e| e == 5.0), "{name}");
        // In-place operations on views of a leaf that requires grad are rejected, as in PyTorch.
        assert_eq!(
            panic_message(|| {
                v.add_(1.0);
            }),
            "a view of a leaf Variable that requires grad is being used in an in-place operation."
        );
        // Views of these views reach `x` through both nodes.
        let w = v.unsqueeze(0);
        assert_eq!(chain(&w).len(), 2, "{name}");
        // The base stops requiring grad: the recorded view still does.
        let b = counting(&[2, 1, 3]);
        let recorded = new_views(&b)
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1;
        b.requires_grad_(false);
        assert!(recorded.requires_grad() && !recorded.is_leaf(), "{name}");
    }
}

#[test]
fn new_views_in_no_grad_mode_follow_their_base_but_drop_gradients() {
    let x = counting(&[2, 1, 3]);
    let quiet = {
        let _no_grad = homura::no_grad();
        new_views(&x)
    };
    for (name, v) in &quiet {
        assert!(
            v.requires_grad() && v.is_leaf() && v.grad_fn().is_none(),
            "{name}"
        );
        let y = v * 2.0;
        assert!(y.requires_grad(), "{name}");
        assert_eq!(edge_names(&y.grad_fn().unwrap()), [None, None], "{name}");
        y.sum().backward();
        assert!(x.grad().is_none() && v.grad().is_none(), "{name}");
        assert_eq!(
            panic_message(|| v.backward_with(&homura::ones_like(v))),
            "element 0 of tensors does not require grad and does not have a grad_fn",
            "{name}"
        );
    }
    x.requires_grad_(false);
    assert!(quiet.iter().all(|(_, v)| !v.requires_grad()));
}

// ---------------------------------------------------------------- meta and empty inputs

#[test]
fn gradients_of_views_on_meta_and_empty_inputs() {
    for device in [Device::Cpu, Device::Meta] {
        for shape in [[0, 1, 3], [1 << 40, 1, 3]] {
            if device == Device::Cpu && shape[0] != 0 {
                continue;
            }
            let x = homura::zeros_with(shape, device);
            x.requires_grad_(true);
            for (name, v) in new_views(&x) {
                let g = homura::ones_with(v.shape(), device);
                let gx = grad_with(&v, &x, &g, true);
                assert_eq!(gx.shape(), shape, "{name} on {device}");
                assert_eq!(gx.device(), device, "{name} on {device}");
                assert_eq!(gx.dtype(), x.dtype(), "{name} on {device}");
            }
        }
    }
}
