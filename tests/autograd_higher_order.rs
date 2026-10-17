//! Typical uses of higher-order gradients: derivatives of a cube up to the fourth, Hessian-vector
//! products (elementwise, and quadratic forms with coupled coordinates), a gradient penalty, and
//! two-stage vector-Jacobian products through dtype conversions. Reference values are asserted by
//! `tools/golden/check_m1_higher_order.py`, block by block in the same order. Run in debug and
//! release profiles.

use homura::autograd::{self, GradOptions, GradcheckOptions, gradcheck, gradgradcheck};
use homura::{DType, Device, Tensor};

fn t(values: &[f64]) -> Tensor {
    Tensor::from_slice(values, [values.len()])
}

fn leaf(values: &[f64]) -> Tensor {
    let x = t(values);
    x.requires_grad_(true);
    x
}

fn vals(t: &Tensor) -> Vec<f64> {
    t.to(DType::F64).to_vec()
}

/// The gradients of `out` with respect to `inputs`, for the upstream gradient `upstream` (ones
/// for a single element when `None`).
fn grad(
    out: &Tensor,
    inputs: &[&Tensor],
    upstream: Option<&Tensor>,
    create_graph: bool,
) -> Vec<Tensor> {
    let options = GradOptions {
        grad_outputs: vec![upstream.cloned()],
        create_graph,
        ..Default::default()
    };
    autograd::grad(&[out], inputs, options)
        .into_iter()
        .map(|g| g.expect("a gradient"))
        .collect()
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-13 * e.abs(),
            "{what}: element {i} is {a:e}, expected {e:e}"
        );
    }
}

#[test]
fn derivatives_of_a_cube_up_to_the_fourth() {
    let x = leaf(&[0.5, -1.25, 2.0]);
    let product = &x * &x * &x;
    let power = x.pow(3.0);
    for (name, y) in [("product", product), ("pow", power)] {
        let g1 = grad(&y.sum(), &[&x], None, true).remove(0);
        let g2 = grad(&g1.sum(), &[&x], None, true).remove(0);
        let g3 = grad(&g2.sum(), &[&x], None, true).remove(0);
        assert_eq!(vals(&g1), [0.75, 4.6875, 12.0], "{name}");
        assert_eq!(vals(&g2), [3.0, -7.5, 12.0], "{name}");
        assert_eq!(vals(&g3), [6.0, 6.0, 6.0], "{name}");
        if name == "product" {
            // The derivatives of products are sums of constants times the factors: the third has
            // no graph left.
            assert!(!g3.requires_grad() && g3.grad_fn().is_none());
        } else {
            // pow multiplies by x.pow(2), x.pow(1) and then x.pow(0), which still depends on x:
            // the fourth derivative is a zero gradient, not an error.
            assert_eq!(g3.grad_fn().unwrap().name(), "MulBackward0");
            let g4 = grad(&g3.sum(), &[&x], None, false).remove(0);
            assert_eq!(vals(&g4), [0.0, 0.0, 0.0]);
        }
    }
}

fn second_derivative(x: f64) -> f64 {
    2.0 * x.sin() + 4.0 * x * x.cos() - x * x * x.sin()
}

fn third_derivative(x: f64) -> f64 {
    6.0 * x.cos() - 6.0 * x * x.sin() - x * x * x.cos()
}

#[test]
fn hessian_vector_products_of_an_elementwise_function() {
    // f(x) = Σ sin(x)·x², whose Hessian is diagonal.
    let (xs, vs, ws) = (
        [0.3, -1.1, 2.0, 0.7],
        [1.0, -0.5, 0.25, 2.0],
        [0.5, 1.5, -1.0, 0.75],
    );
    let (x, v, w) = (leaf(&xs), leaf(&vs), t(&ws));
    let g = grad(&(x.sin() * &x * &x).sum(), &[&x], None, true).remove(0);
    let hv = grad(&g, &[&x], Some(&v), true).remove(0);
    let expected: Vec<f64> = (0..4).map(|i| second_derivative(xs[i]) * vs[i]).collect();
    assert_close(&vals(&hv), &expected, "Hv");
    // The product is linear in v, with derivative H w, and its derivative in x has the third
    // derivatives.
    let [hw_v, hw_x] = grad(&hv, &[&v, &x], Some(&w), false).try_into().unwrap();
    let expected: Vec<f64> = (0..4).map(|i| second_derivative(xs[i]) * ws[i]).collect();
    assert_close(&vals(&hw_v), &expected, "Hw");
    let expected: Vec<f64> = (0..4)
        .map(|i| third_derivative(xs[i]) * vs[i] * ws[i])
        .collect();
    assert_close(&vals(&hw_x), &expected, "third derivatives");
}

#[test]
fn hessian_vector_products_of_a_quadratic_form() {
    // f(x) = xᵀMx / 2 + bᵀx, whose Hessian (M + Mᵀ) / 2 couples the coordinates. The values are
    // exact.
    let x = leaf(&[1.0, -2.0, 0.5]);
    let b = t(&[0.5, -1.0, 2.0]);
    let v = t(&[0.5, 1.0, -1.0]);
    let matrix = |rows: [[f64; 3]; 3]| Tensor::from_slice(rows.as_flattened(), [3, 3]);
    let symmetric = matrix([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.25], [0.5, 0.25, 1.0]]);
    let general = matrix([[2.0, -1.0, 0.5], [0.0, 3.0, 0.25], [1.5, -0.75, 1.0]]);
    for (m, expected_g, expected_hv) in [
        (symmetric, [4.75, -7.875, 2.5], [-0.5, 2.25, -0.5]),
        (general, [4.0, -7.625, 4.0], [-0.5, 3.0, -0.75]),
    ] {
        let f = x.matmul(&m.matmul(&x)) * 0.5 + b.matmul(&x);
        let g = grad(&f, &[&x], None, true).remove(0);
        let hv = grad(&g, &[&x], Some(&v), false).remove(0);
        assert_eq!(vals(&g), expected_g);
        assert_eq!(vals(&hv), expected_hv);
    }
}

/// `Σ (∂y/∂X)²` for the network `y = Σ tanh(X W1 + b1) w2`, as a function of `[X, W1, b1, w2]`.
fn penalty(t: &[Tensor]) -> Tensor {
    let (x, w1, b1, w2) = (&t[0], &t[1], &t[2], &t[3]);
    let y = (x.matmul(w1) + b1).tanh().matmul(w2).sum();
    let gx = grad(&y, &[x], None, true).remove(0);
    (&gx * &gx).sum()
}

#[test]
fn gradient_penalty_of_a_small_network() {
    let input = |values: &[f64], shape: &[usize]| {
        let x = Tensor::from_slice(values, shape);
        x.requires_grad_(true);
        x
    };
    let x = input(&[0.5, -1.0, 0.25, 1.5, 0.75, -0.5], &[2, 3]);
    let w1 = input(&[0.3, -0.6, 0.8, 0.1, -0.4, 0.5], &[3, 2]);
    let b1 = leaf(&[0.1, -0.2]);
    let w2 = leaf(&[0.7, -1.3]);
    let params = [x, w1, b1, w2];
    let p = penalty(&params);
    assert_close(&vals(&p), &[1.3059498758318049], "penalty");
    let refs: Vec<&Tensor> = params.iter().collect();
    let grads = grad(&p, &refs, None, false);
    let expected: [&[f64]; 4] = [
        &[
            -0.639555733871122,
            0.7780865554975789,
            0.4144641821735521,
            -0.23447136145361183,
            -0.08458010976176594,
            0.21721491944734928,
        ],
        &[
            0.9870142936783939,
            -0.5815705400743374,
            -0.6102296086399324,
            -1.8600872720033894,
            -0.4797688917101419,
            1.8622800716570966,
        ],
        &[0.6445120743314008, 1.7789678627069234],
        &[1.1059625572918017, -1.4136353550456529],
    ];
    for ((name, g), e) in ["X", "W1", "b1", "w2"].iter().zip(&grads).zip(expected) {
        assert_close(&vals(g), e, name);
    }
    // The numerical checks of the penalty need its second and third derivatives to be right.
    let f = |t: &[Tensor]| vec![penalty(t)];
    gradcheck(f, &refs, GradcheckOptions::default()).unwrap();
    gradgradcheck(f, &refs, GradcheckOptions::default()).unwrap();
}

#[test]
fn two_stage_products_across_dtypes() {
    // The numerical checkers work in f64 only, so conversions are checked analytically:
    // y = x.to(dst)², g = ∂y/∂x · u = 2xu converted back, and then g's own vector-Jacobian
    // product with respect to x and u. The values are exact in every dtype.
    for (src, dst) in [
        (DType::F32, DType::F64),
        (DType::F64, DType::F32),
        (DType::F64, DType::F16),
        (DType::F64, DType::BF16),
    ] {
        let of = |values: &[f64], dtype: DType| t(values).to(dtype);
        let x = of(&[1.5, -0.75, 2.0], src);
        x.requires_grad_(true);
        let u = of(&[0.5, 2.0, -1.0], dst);
        u.requires_grad_(true);
        let w = of(&[1.0, 0.25, -2.0], src);
        let y = x.to(dst).pow(2.0);
        let g = grad(&y, &[&x], Some(&u), true).remove(0);
        let what = format!("{src:?} -> {dst:?}");
        assert_eq!(g.dtype(), src, "{what}");
        assert_eq!(g.grad_fn().unwrap().name(), "ToCopyBackward0", "{what}");
        assert_eq!(vals(&g), [1.5, -3.0, -4.0], "{what}");
        let [gx, gu] = grad(&g, &[&x, &u], Some(&w), false).try_into().unwrap();
        for (r, dtype, expected) in [(&gx, src, [1.0, 1.0, 4.0]), (&gu, dst, [3.0, -0.375, -8.0])] {
            assert_eq!(r.dtype(), dtype, "{what}");
            assert_eq!(r.shape(), [3], "{what}");
            assert_eq!(r.device(), Device::Cpu, "{what}");
            assert_eq!(vals(r), expected, "{what}");
        }
    }
}
