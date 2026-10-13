//! Reduction gradients: pinned PyTorch boundary semantics (ties and NaN of max and min, zeros of
//! prod, degrees of freedom of var and std), dimensions and keepdim, higher derivatives, saved
//! values and metadata-only backward passes.

use std::panic::{AssertUnwindSafe, catch_unwind};

use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::{DType, Device, Tensor};

const NO_DIMS: [i64; 0] = [];

fn leaf(values: &[f64], shape: impl homura::IntoShape) -> Tensor {
    let x = Tensor::from_slice(values, shape);
    x.requires_grad_(true);
    x
}

fn gradients(out: &Tensor, inputs: &[&Tensor], g: &Tensor, graph: bool) -> Vec<Tensor> {
    autograd::grad(
        &[out],
        inputs,
        GradOptions {
            grad_outputs: vec![Some(g.clone())],
            create_graph: graph,
            ..Default::default()
        },
    )
    .into_iter()
    .map(Option::unwrap)
    .collect()
}

/// The gradient of `out` with respect to `x` for the upstream gradient `g` of the shape of `out`.
fn grad_of(out: &Tensor, x: &Tensor, g: &[f64]) -> Tensor {
    let g = Tensor::from_slice(g, out.shape()).to(out.dtype());
    gradients(out, &[x], &g, false).remove(0)
}

/// Equal values within 1e-12 relative to the expected value; NaN equals NaN and infinities only
/// themselves.
fn same(t: &Tensor, expected: &[f64]) {
    let actual = t.to(DType::F64).to_vec::<f64>();
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (&a, &e) in actual.iter().zip(expected) {
        let close = if e.is_finite() {
            (a - e).abs() <= 1e-12 * e.abs().max(1.0)
        } else {
            a == e || (a.is_nan() && e.is_nan())
        };
        assert!(close, "{actual:?} vs {expected:?}");
    }
}

fn panic_message(f: impl FnOnce()) -> String {
    let p = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}

#[test]
fn sums_and_means_expand_the_gradient_back() {
    let inf = f64::INFINITY;
    let x = leaf(&[1., 2., 3., 4., 5., 6.], [2, 3]);
    same(
        &grad_of(&x.sum_dim(-1, false), &x, &[inf, -2.]),
        &[inf, inf, inf, -2., -2., -2.],
    );
    same(
        &grad_of(&x.sum_dim(0, true), &x, &[1., 2., 3.]),
        &[1., 2., 3., 1., 2., 3.],
    );
    same(&grad_of(&x.sum_dim([1, 0], false), &x, &[4.]), &[4.; 6]);
    let all = x.sum_dim(NO_DIMS, false);
    assert_eq!(all.shape(), []);
    same(&grad_of(&all, &x, &[3.]), &[3.; 6]);
    same(&grad_of(&x.sum_dim(NO_DIMS, true), &x, &[3.]), &[3.; 6]);
    same(&grad_of(&x.mean(), &x, &[3.]), &[0.5; 6]);
    same(
        &grad_of(&x.mean_dim(0, false), &x, &[2., inf, -4.]),
        &[1., inf, -2., 1., inf, -2.],
    );
    same(
        &grad_of(&x.mean_dim(-1, true), &x, &[3., 6.]),
        &[1., 1., 1., 2., 2., 2.],
    );
    same(&grad_of(&x.mean_dim(NO_DIMS, true), &x, &[3.]), &[0.5; 6]);
    // A zero-dimensional input reduces over dimension 0 or -1 to itself.
    let s = leaf(&[3.], ());
    for out in [s.sum_dim(0, false), s.mean_dim(-1, true), s.mean()] {
        same(&grad_of(&out, &s, &[2.]), &[2.]);
    }
    let names = [
        (x.sum_dim(0, false), "SumBackward1"),
        (x.sum_dim(NO_DIMS, false), "SumBackward1"),
        (x.mean(), "MeanBackward0"),
        (x.mean_dim(1, false), "MeanBackward1"),
        (x.mean_dim(NO_DIMS, false), "MeanBackward1"),
    ];
    for (out, name) in names {
        let node = out.grad_fn().unwrap();
        assert_eq!((node.name(), node.next_functions().len()), (name, 1));
    }
}

#[test]
fn prod_follows_the_reference_at_zeros_and_non_finite_values() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    // Without zeros the gradient is `g * (result / x)`; with any zero in the input, the products
    // of all other elements, also in rows without zeros. Two zeros of a whole product give exact
    // zeros outside grad mode, whatever the upstream gradient.
    for (values, g, expected) in [
        (vec![2., 3., 4.], 1., vec![12., 8., 6.]),
        (vec![2., 3., 4.], inf, vec![inf; 3]),
        (vec![2., inf, 4.], 1., vec![inf, nan, inf]),
        (vec![2., nan, 4.], 1., vec![nan; 3]),
        (vec![2., 0., 4.], 1., vec![0., 8., 0.]),
        (vec![2., 0., 4.], inf, vec![nan, inf, nan]),
        (vec![0., inf, 2.], 1., vec![inf, 0., nan]),
        (vec![0., nan, 2.], 1., vec![nan, 0., nan]),
        (vec![-0., 3., 2.], 1., vec![6., -0., -0.]),
        (vec![0., 3., 0.], 1., vec![0.; 3]),
        (vec![0., 3., 0.], inf, vec![0.; 3]),
        (vec![0., inf, 0.], nan, vec![0.; 3]),
    ] {
        let x = leaf(&values, [3]);
        same(&grad_of(&x.prod(), &x, &[g]), &expected);
    }
    let x = leaf(&[2., inf, 4., 0., 3., 5.], [2, 3]);
    same(
        &grad_of(&x.prod_dim(1, false), &x, &[1., 1.]),
        &[inf, 8., inf, 15., 0., 0.],
    );
    let x = leaf(&[2., inf, 4., 1., 3., 5.], [2, 3]);
    same(
        &grad_of(&x.prod_dim(-1, false), &x, &[1., 1.]),
        &[inf, nan, inf, 15., 5., 3.],
    );
    let x = leaf(&[0., inf, 0., 1., 3., 5.], [2, 3]);
    same(
        &grad_of(&x.prod_dim(1, false), &x, &[inf, 1.]),
        &[nan, nan, nan, 15., 5., 3.],
    );
    let x = leaf(&[0., 2., 3., 4.], [2, 2]);
    same(
        &grad_of(&x.prod_dim(0, true), &x, &[1., 2.]),
        &[3., 8., 0., 4.],
    );
    // The products of the other elements accumulate in f64 and are rounded to f32 before they
    // are multiplied: 1e30 * 1e30 is infinite in f32, and infinity times zero is NaN.
    let x = Tensor::from_slice(&[1e30f32, 1e30, 1e-30, 0.], [4]);
    x.requires_grad_(true);
    same(
        &grad_of(&x.prod(), &x, &[1.]),
        &[0., 0., nan, f64::from(1e30f32)],
    );
    // Grouped differently in f32, the products before the last element would overflow and
    // underflow (1e30 * 1e30 and 1e-30 * 1e-30) and give NaN instead of 1. The reversed row
    // checks the products after each element the same way.
    let v = [1e30f32, 1e30, 1e-30, 1e-30, 0.];
    let reversed: Vec<f32> = v.iter().rev().copied().collect();
    let x = Tensor::from_slice(&[&v[..], &reversed[..]].concat(), [2, 5]);
    x.requires_grad_(true);
    same(
        &grad_of(&x.prod_dim(1, false), &x, &[1., 1.]),
        &[0., 0., nan, 0., 1., 1., 0., nan, 0., 0.],
    );
    let x = Tensor::from_slice(&v, [5]);
    x.requires_grad_(true);
    same(&grad_of(&x.prod(), &x, &[1.]), &[0., 0., nan, 0., 1.]);
    for (out, name) in [
        (x.prod(), "ProdBackward0"),
        (x.prod_dim(0, false), "ProdBackward1"),
    ] {
        let node = out.grad_fn().unwrap();
        assert_eq!((node.name(), node.next_functions().len()), (name, 1));
    }
}

/// `values` repeated: `rep(&[(1e30, 11), (0., 1)])` is eleven 1e30 and a zero.
fn rep(runs: &[(f64, usize)]) -> Vec<f64> {
    runs.iter()
        .flat_map(|&(v, n)| std::iter::repeat_n(v, n))
        .collect()
}

#[test]
fn prod_with_zeros_accumulates_in_order_like_the_reference_cumprod() {
    // The products of the elements before and after each one are accumulated one element at a
    // time in PyTorch's accumulation type (f64 for f32 and f64, f32 for f16 and bf16), and each
    // is rounded to the dtype: intermediate products overflow or underflow in that type exactly
    // where the reference's do (review round 18, R35 and R36). Expected values are the pinned
    // reference's, for an upstream gradient of 1 and of Inf.
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    // (dtype, input, gradient for an upstream 1, gradient for an upstream Inf)
    type Case = (DType, Vec<f64>, Vec<f64>, Vec<f64>);
    let cases: Vec<Case> = vec![
        (
            DType::F64,
            vec![1e308, 1e308, 1e-308, 1e-308, 0.],
            vec![0., 0., nan, nan, inf],
            vec![nan, nan, nan, nan, inf],
        ),
        (
            DType::F64,
            vec![1e-308, 1e-308, 1e308, 1e308, 0.],
            vec![0.; 5],
            vec![nan; 5],
        ),
        // Rounding: the last element gets 1e308 * 1e-308 * 1e-308 * 1e308 in this order.
        (
            DType::F64,
            vec![1e308, 1e-308, 1e-308, 1e308, 0.],
            vec![0., 0., 0., 0., 0.9999999999999999],
            vec![nan, nan, nan, nan, inf],
        ),
        (
            DType::F32,
            rep(&[(1e30, 11), (1e-30, 11), (0., 1)]),
            rep(&[(0., 2), (nan, 20), (inf, 1)]),
            rep(&[(nan, 22), (inf, 1)]),
        ),
        (
            DType::F32,
            rep(&[(1e-30, 11), (1e30, 11), (0., 1)]),
            vec![0.; 23],
            vec![nan; 23],
        ),
        (
            DType::BF16,
            vec![1e30, 1e30, 1e-30, 0.],
            vec![0., 0., nan, inf],
            vec![nan, nan, nan, inf],
        ),
        (
            DType::BF16,
            vec![1e30, 1e30, 1e-30, 1e-30, 0.],
            vec![0., 0., nan, nan, inf],
            vec![nan, nan, nan, nan, inf],
        ),
        (
            DType::F16,
            rep(&[(1000., 14), (0.001, 14), (0., 1)]),
            rep(&[(0., 2), (nan, 26), (inf, 1)]),
            rep(&[(nan, 28), (inf, 1)]),
        ),
        // Each product is rounded to f16 when it is written, while the f32 accumulation goes on.
        (
            DType::F16,
            rep(&[(0.001, 14), (1000., 14), (0., 1)]),
            rep(&[(0., 28), (1.005859375, 1)]),
            rep(&[(nan, 28), (inf, 1)]),
        ),
    ];
    for (dtype, values, expected, expected_inf) in cases {
        let n = values.len();
        let what = format!("{dtype:?} {values:?}");
        let x = Tensor::from_slice(&values, [n]).to(dtype);
        x.requires_grad_(true);
        let check = |actual: &Tensor, expected: &[f64]| {
            assert_eq!(actual.dtype(), dtype, "{what}");
            let actual = actual.to(DType::F64).to_vec::<f64>();
            let equal = actual
                .iter()
                .zip(expected)
                .all(|(&a, &e)| a == e || (a.is_nan() && e.is_nan()));
            assert!(equal, "{what}: {actual:?} vs {expected:?}");
        };
        check(&grad_of(&x.prod(), &x, &[1.]), &expected);
        check(&grad_of(&x.prod(), &x, &[inf]), &expected_inf);
        // The same values in a row and reversed in another, in a non-contiguous tensor, give the
        // same products in reverse; the upstream gradient scales each row.
        let reversed: Vec<f64> = values.iter().rev().copied().collect();
        let rows = Tensor::from_slice(&[&reversed[..], &values[..]].concat(), [2, n])
            .to(dtype)
            .flip(0)
            .transpose(0, 1)
            .deep_clone()
            .transpose(0, 1);
        rows.requires_grad_(true);
        let mut both: Vec<f64> = expected.clone();
        both.extend(expected.iter().rev().map(|&e| -2. * e));
        check(&grad_of(&rows.prod_dim(1, false), &rows, &[1., -2.]), &both);
        check(&grad_of(&rows.prod_dim(-1, true), &rows, &[1., -2.]), &both);
    }
}

#[test]
fn prod_differentiates_again_without_zeros_and_rejects_zeros() {
    let x = leaf(&[2., 3., 4.], [3]);
    let gx = gradients(&x.prod(), &[&x], &homura::ones_like(&x.prod()), true).remove(0);
    same(&gx, &[12., 8., 6.]);
    same(&grad_of(&gx.sum(), &x, &[1.]), &[7., 6., 5.]);
    let x = leaf(&[2., 3., 4., 5.], [2, 2]);
    let g = Tensor::from_slice(&[1., 2.], [2]);
    let gx = gradients(&x.prod_dim(1, false), &[&x], &g, true).remove(0);
    same(&gx, &[3., 2., 10., 8.]);
    same(&grad_of(&gx.sum(), &x, &[1.]), &[1., 1., 2., 2.]);
    // With zeros, the reference differentiates its cumulative products again; homura's
    // gradient isn't differentiable there yet and says so.
    let unsupported = "computing the gradient of prod with create_graph for an input that \
                       contains zeros is not supported yet";
    for (x, dim) in [
        (leaf(&[0., 2., 3.], [3]), None),
        (leaf(&[0., 2., 0.], [3]), None),
        (leaf(&[0., 2., 3., 4.], [2, 2]), Some(1)),
    ] {
        let out = dim.map_or_else(|| x.prod(), |d| x.prod_dim(d, false));
        let g = homura::ones_like(&out);
        let message = panic_message(|| drop(gradients(&out, &[&x], &g, true)));
        assert_eq!(message, unsupported);
    }
    // Without a product to compute, the gradient is the upstream one, even with zeros.
    let x = leaf(&[0., 2.], [2, 1]);
    let g = leaf(&[3., 4.], [2]);
    let gx = gradients(&x.prod_dim(1, false), &[&x], &g, true).remove(0);
    same(&gx, &[3., 4.]);
    same(
        &gradients(&gx.sum(), &[&g], &homura::ones_like(&gx.sum()), false)[0],
        &[1., 1.],
    );
    let s = leaf(&[0.], ());
    let gs = gradients(&s.prod(), &[&s], &Tensor::from_slice(&[5.], ()), true).remove(0);
    same(&gs, &[5.]);
}

#[test]
fn max_and_min_split_ties_select_nan_and_return_indices() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    for (values, g, max, expected) in [
        (vec![1., 3., 3., 2.], 1., true, vec![0., 0.5, 0.5, 0.]),
        (vec![1., 3., 3., 2.], inf, true, vec![0., inf, inf, 0.]),
        (vec![1., 3., 3., 2.], nan, true, vec![0., nan, nan, 0.]),
        (vec![nan, 1., nan], 1., true, vec![0.5, 0., 0.5]),
        (vec![inf, 1., inf], 1., true, vec![0.5, 0., 0.5]),
        (vec![-inf, -inf], 3., true, vec![1.5, 1.5]),
        (vec![1., nan, 0.], 3., false, vec![0., 3., 0.]),
        (vec![-0., 0.], 1., true, vec![0.5, 0.5]),
        (vec![5., 5., 5.], 1., false, vec![1. / 3.; 3]),
    ] {
        let x = leaf(&values, [values.len()]);
        let out = if max { x.max() } else { x.min() };
        same(&grad_of(&out, &x, &[g]), &expected);
    }
    let x = leaf(&[1., 3., 3., nan, 2., nan], [2, 3]);
    same(
        &grad_of(&x.max_dim(1, false).0, &x, &[inf, 5.]),
        &[0., inf, 0., 5., 0., 0.],
    );
    let x = leaf(&[1., 3., 1., nan], [2, 2]);
    same(
        &grad_of(&x.min_dim(0, false).0, &x, &[2., inf]),
        &[2., 0., 0., inf],
    );
    let x = leaf(&[1., 3., 3., 4., 2., 1.], [2, 3]);
    same(
        &grad_of(&x.max_dim(-1, true).0, &x, &[1., 2.]),
        &[0., 1., 0., 2., 0., 0.],
    );
    let s = leaf(&[3.], ());
    same(&grad_of(&s.max_dim(0, false).0, &s, &[2.]), &[2.]);
    // The share of each tie depends on the upstream gradient, also in a second pass.
    let x = leaf(&[1., 3., 3.], [3]);
    let g = leaf(&[2.], ());
    let gx = gradients(&x.max(), &[&x], &g, true).remove(0);
    same(&gx, &[0., 1., 1.]);
    same(&grad_of(&gx.sum(), &g, &[1.]), &[1.]);
    for (out, name) in [
        (x.max(), "MaxBackward1"),
        (x.min(), "MinBackward1"),
        (x.max_dim(0, false).0, "MaxBackward0"),
        (x.min_dim(0, true).0, "MinBackward0"),
    ] {
        let node = out.grad_fn().unwrap();
        assert_eq!((node.name(), node.next_functions().len()), (name, 1));
    }
    let (values, indices) = x.max_dim(0, false);
    assert!(values.requires_grad() && !indices.requires_grad());
    assert!(indices.grad_fn().is_none());
}

#[test]
fn var_and_std_follow_the_whole_and_dimension_rules() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    type Reduce = fn(&Tensor, i64) -> Tensor;
    let whole_var: Reduce = |x, c| x.var(c);
    let dim_var: Reduce = |x, c| x.var_dim(0, c, false);
    let whole_std: Reduce = |x, c| x.std(c);
    let dim_std: Reduce = |x, c| x.std_dim(0, c, false);
    // Inputs 2, 4, ..., 2N with an upstream gradient of 1 (review/M1a.md measurements).
    let (third, sixth) = (4. / 3., 0.408_248_290_463_862_96);
    for (n, c, expected) in [
        (
            2,
            2,
            [
                vec![inf, inf],
                vec![-inf, inf],
                vec![nan, nan],
                vec![nan, nan],
            ],
        ),
        (
            2,
            3,
            [vec![inf, inf], vec![2., -2.], vec![nan, nan], vec![0., -0.]],
        ),
        (1, 1, [vec![nan], vec![nan], vec![nan], vec![nan]]),
        (
            3,
            0,
            [
                vec![-third, 0., third],
                vec![-third, 0., third],
                vec![-sixth, 0., sixth],
                vec![-sixth, 0., sixth],
            ],
        ),
    ] {
        let values: Vec<f64> = (1..=n).map(|i| 2. * i as f64).collect();
        for (f, expected) in [whole_var, dim_var, whole_std, dim_std]
            .iter()
            .zip(expected)
        {
            let x = leaf(&values, [n]);
            same(&grad_of(&f(&x, c), &x, &[1.]), &expected);
        }
    }
    // The whole variance with nonpositive degrees of freedom gives `g * Inf`, NaN where x equals
    // the mean; the dimension version divides by the nonpositive count.
    let x = leaf(&[1., 2., 3.], [3]);
    for (g, expected) in [
        (1., vec![inf, nan, inf]),
        (0., vec![nan; 3]),
        (-2., vec![-inf, nan, -inf]),
    ] {
        same(&grad_of(&x.var(3), &x, &[g]), &expected);
    }
    same(
        &grad_of(&x.var_dim(0, 3, false), &x, &[1.]),
        &[-inf, nan, inf],
    );
    same(
        &grad_of(&x.var_dim(0, 4, false), &x, &[1.]),
        &[2., -0., -2.],
    );
    same(
        &grad_of(&x.std_dim(0, 4, false), &x, &[1.]),
        &[0., -0., -0.],
    );
    for (f, c) in [(whole_std, 3), (whole_std, 4), (dim_std, 3)] {
        same(&grad_of(&f(&x, c), &x, &[1.]), &[nan; 3]);
    }
    same(&grad_of(&x.var(1), &x, &[inf]), &[-inf, nan, inf]);
    same(&grad_of(&x.var(-1), &x, &[1.]), &[-0.5, 0., 0.5]);
    for (values, expected) in [
        (vec![1., inf, 3.], vec![-inf, nan, -inf]),
        (vec![1., nan, 3.], vec![nan; 3]),
    ] {
        let x = leaf(&values, [3]);
        same(&grad_of(&x.var(1), &x, &[1.]), &expected);
    }
    // A zero standard deviation gives zero gradients, whatever the upstream gradient.
    let x = leaf(&[2., 2., 2.], [3]);
    for g in [1., inf, nan] {
        same(&grad_of(&x.std(1), &x, &[g]), &[0.; 3]);
        same(&grad_of(&x.std_dim(-1, 1, true), &x, &[g]), &[0.; 3]);
    }
    // Zero standard deviation isn't a smooth point: as in PyTorch, the second derivative with
    // respect to the upstream gradient is NaN there (0 / 0 behind the selected zero).
    let g = leaf(&[0.3], ());
    let gx = gradients(&x.std(1), &[&x], &g, true).remove(0);
    let ds = gradients(&gx.sum(), &[&x, &g], &Tensor::from_slice(&[1.], ()), false);
    same(&ds[0], &[0.; 3]);
    same(&ds[1], &[nan]);
    // Zero-dimensional inputs use the whole rule, also with a dimension.
    let s = leaf(&[3.], ());
    for out in [s.var(1), s.var_dim(0, 1, false)] {
        same(&grad_of(&out, &s, &[2.]), &[nan]);
    }
    for out in [
        s.var(0),
        s.var_dim(0, 0, true),
        s.std(0),
        s.std_dim(-1, 0, false),
    ] {
        same(&grad_of(&out, &s, &[2.]), &[0.]);
    }
    // An empty list of dimensions reduces every dimension, and gets the gradient of the whole
    // reduction. (The reference counts one reduced element for `dim=[]` and gives ±Inf.)
    let x = leaf(&[1., 2., 4., 8.], [2, 2]);
    let expected = grad_of(&x.var(1), &x, &[1.]).to_vec::<f64>();
    same(
        &grad_of(&x.var_dim(NO_DIMS, 1, false), &x, &[1.]),
        &expected,
    );
    same(&grad_of(&x.var_dim(NO_DIMS, 1, true), &x, &[1.]), &expected);
    let expected = grad_of(&x.std(1), &x, &[1.]).to_vec::<f64>();
    same(
        &grad_of(&x.std_dim(NO_DIMS, 1, false), &x, &[1.]),
        &expected,
    );
    same(
        &grad_of(&x.var_dim(NO_DIMS, 4, false), &x, &[1.]),
        &[inf; 4],
    );
    for (out, name) in [
        (x.var(1), "VarBackward0"),
        (x.var_dim(1, 0, true), "VarBackward0"),
        (x.std(1), "StdBackward0"),
        (x.std_dim([0, 1], 1, false), "StdBackward0"),
    ] {
        let node = out.grad_fn().unwrap();
        assert_eq!((node.name(), node.next_functions().len()), (name, 1));
    }
}

#[test]
fn reductions_pass_both_gradient_checks() {
    // Distinct values away from zero: smooth points of max, min and prod.
    let x = leaf(
        &[0.7, 1.2, 1.8, 2.3, 0.9, 1.5, 1.1, 2.6, 0.6, 1.4, 2.1, 1.3],
        [3, 4],
    )
    .detach()
    .transpose(0, 1);
    x.requires_grad_(true);
    let s = leaf(&[1.7], ());
    type F = fn(&[Tensor]) -> Vec<Tensor>;
    let cases: &[(&str, F)] = &[
        ("sum_dim", |t| {
            vec![
                t[0].sum_dim(0, false),
                t[0].sum_dim(-1, true),
                t[0].sum_dim([1, 0], false),
                t[0].sum_dim(NO_DIMS, true),
            ]
        }),
        ("mean", |t| {
            vec![
                t[0].mean(),
                t[0].mean_dim(1, false),
                t[0].mean_dim([-1, 0], true),
                t[0].mean_dim(NO_DIMS, false),
            ]
        }),
        ("prod", |t| {
            vec![
                t[0].prod(),
                t[0].prod_dim(0, false),
                t[0].prod_dim(-1, true),
            ]
        }),
        ("max and min", |t| vec![t[0].max(), t[0].min()]),
        ("max_dim and min_dim", |t| {
            vec![
                t[0].max_dim(0, false).0,
                t[0].min_dim(-1, true).0,
                t[0].max_dim(1, true).0,
            ]
        }),
        ("var", |t| {
            vec![
                t[0].var(1),
                t[0].var(0),
                t[0].var_dim(0, 1, false),
                t[0].var_dim([0, 1], 2, true),
                t[0].var_dim(NO_DIMS, 1, false),
            ]
        }),
        ("std", |t| {
            vec![
                t[0].std(1),
                t[0].std_dim(1, 0, true),
                t[0].std_dim(0, 1, false),
            ]
        }),
        ("zero dimensions", |t| {
            vec![
                t[1].sum_dim(0, true),
                t[1].mean_dim(NO_DIMS, false),
                t[1].prod(),
                t[1].prod_dim(0, false),
                t[1].max(),
                t[1].min_dim(0, false).0,
                t[1].var(0),
            ]
        }),
    ];
    for (name, f) in cases {
        let inputs = [&x, &s];
        gradcheck(f, &inputs, GradcheckOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        gradgradcheck(f, &inputs, GradcheckOptions::default())
            .unwrap_or_else(|e| panic!("{name} second order: {e}"));
    }
}

#[test]
fn empty_and_meta_inputs_get_gradients_of_their_shape() {
    for (device, n) in [(Device::Cpu, 0), (Device::Meta, 1usize << 40)] {
        let x = homura::zeros_with([n, 3], (DType::F32, device));
        x.requires_grad_(true);
        let mut outs = vec![
            x.sum_dim(0, false),
            x.sum_dim(1, true),
            x.mean(),
            x.mean_dim(0, true),
            x.mean_dim(1, false),
            x.prod(),
            x.prod_dim(0, false),
            x.prod_dim(1, true),
            x.max_dim(1, false).0,
            x.min_dim(1, true).0,
            x.var(1),
            x.var_dim(0, 1, false),
            x.var_dim(1, 0, true),
            x.std(1),
            x.std_dim(0, 1, true),
        ];
        if device == Device::Meta {
            outs.extend([x.max(), x.min(), x.max_dim(0, false).0]);
        }
        for out in outs {
            let g = homura::ones_like(&out);
            let d = gradients(&out, &[&x], &g, true).remove(0);
            assert_eq!(
                (d.shape(), d.dtype(), d.device()),
                (x.shape(), DType::F32, device)
            );
        }
    }
}

#[test]
fn new_nodes_check_and_release_saved_values() {
    type F = fn(&Tensor) -> Tensor;
    // (node, operation, saves anything, saves the input): max_dim and min_dim save the indices
    // only, sums and means nothing, so that they can run again.
    let cases: &[(&str, F, bool, bool)] = &[
        ("SumBackward1", |x| x.sum_dim(1, false), false, false),
        ("MeanBackward0", |x| x.mean(), false, false),
        ("MeanBackward1", |x| x.mean_dim(1, true), false, false),
        ("ProdBackward0", |x| x.prod(), true, true),
        ("ProdBackward1", |x| x.prod_dim(1, false), true, true),
        ("MaxBackward1", |x| x.max(), true, true),
        ("MinBackward1", |x| x.min(), true, true),
        ("MaxBackward0", |x| x.max_dim(1, false).0, true, false),
        ("MinBackward0", |x| x.min_dim(0, true).0, true, false),
        ("VarBackward0", |x| x.var(1), true, true),
        ("VarBackward0", |x| x.var_dim(1, 1, false), true, true),
        ("StdBackward0", |x| x.std(1), true, true),
        ("StdBackward0", |x| x.std_dim(0, 1, false), true, true),
    ];
    for &(name, f, saves, saves_input) in cases {
        let x = leaf(&[1., 3., 2., 4., 6., 5.], [2, 3]);
        let out = f(&x);
        assert_eq!(out.grad_fn().unwrap().name(), name);
        let loss = out.sum();
        loss.backward();
        let second = catch_unwind(AssertUnwindSafe(|| loss.backward()));
        assert_eq!(second.is_err(), saves, "{name}");
        let x = leaf(&[1., 3., 2., 4., 6., 5.], [2, 3]);
        let loss = f(&x).sum();
        {
            let _guard = homura::no_grad();
            x.add_(1.);
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| loss.backward()));
        assert_eq!(outcome.is_err(), saves_input, "{name}");
    }
    // Saved results and indices are checked too, described like PyTorch does.
    for (f, message) in [
        (
            (|x| x.prod()) as F,
            "[f64 []], which is output 0 of Prod, is at version 1",
        ),
        (
            |x| x.max(),
            "[f64 []], which is output 0 of Max1, is at version 1",
        ),
        (
            |x| x.std(1),
            "[f64 []], which is output 0 of Std, is at version 1",
        ),
    ] {
        let x = leaf(&[1., 3., 2., 4., 6., 5.], [2, 3]);
        let out = f(&x);
        {
            let _guard = homura::no_grad();
            out.add_(1.);
        }
        let actual = panic_message(|| out.backward());
        assert!(actual.contains(message), "{actual}");
    }
    let x = leaf(&[1., 3., 2., 4., 6., 5.], [2, 3]);
    let (values, indices) = x.max_dim(1, false);
    {
        let _guard = homura::no_grad();
        indices.add_(1i64);
    }
    let actual = panic_message(|| values.sum().backward());
    assert!(
        actual.contains("[i64 [2]] is at version 1; expected version 0 instead."),
        "{actual}"
    );
}
