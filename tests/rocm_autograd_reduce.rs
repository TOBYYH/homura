//! M1's reduction and softmax formulas on a ROCm GPU (M4 step 4a, item 4): first and second
//! order gradients computed on the GPU match the CPU's, within `(1e-10, 1e-12)` for F64 (the M4
//! acceptance tolerance): sums, means, products (with zeros too, first order), max and min over
//! all elements (ties included), var and std, softmax and log-softmax (the second-order graphs
//! of M1's R40), and the gradients of broadcast operands, which reduce (step 3 left them to this
//! step). The gradients of `max_dim` and `min_dim` scatter into their indices: step 4b.
#![cfg(feature = "rocm")]

mod common;
#[path = "common/rocm.rs"]
mod rocm;

use homura::autograd::{self, GradOptions};
use homura::{Device, Tensor};
use rocm::{GPU, Words};

fn inputs(n: usize, low: f64, high: f64, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    let v: Vec<f64> = (0..n)
        .map(|_| low + (w.next() >> 11) as f64 / (1u64 << 53) as f64 * (high - low))
        .collect();
    Tensor::from_slice(&v, [n])
}

/// `d<y, v>/dx` (and, with `create_graph`, its graph) for `y = f(x)`, `v` of `y`'s shape.
fn first(y: &Tensor, x: &Tensor, seed: u64, create_graph: bool) -> Tensor {
    let v = inputs(y.num_elements(), -1.0, 1.0, seed)
        .reshape(y.shape())
        .to(y.device());
    autograd::grad(
        &[y],
        &[x],
        GradOptions {
            grad_outputs: vec![Some(v)],
            create_graph,
            ..Default::default()
        },
    )
    .remove(0)
    .expect("a gradient")
}

/// The gradients of `f` at `x` on `device` up to `order` (1 or 2), as CPU tensors.
fn gradients(
    f: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
    device: Device,
    order: usize,
) -> Vec<Tensor> {
    let x = x.to(device);
    x.requires_grad_(true);
    let y = f(&x);
    let g = first(&y, &x, 5, order > 1);
    let mut out = vec![g.to(Device::Cpu)];
    if order > 1 {
        out.push(first(&g, &x, 6, false).to(Device::Cpu));
    }
    out
}

fn compare(name: &str, f: &dyn Fn(&Tensor) -> Tensor, x: &Tensor, order: usize, bitwise: bool) {
    let cpu = gradients(f, x, Device::Cpu, order);
    let gpu = gradients(f, x, GPU, order);
    for (k, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let what = format!("{name}: order {}", k + 1);
        if bitwise {
            let (a, b) = (g.to_vec::<f64>(), c.to_vec::<f64>());
            for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                assert!(
                    p.to_bits() == q.to_bits(),
                    "{what}: element {i}: {p:e} vs {q:e}"
                );
            }
        } else {
            common::assert_close_tol(g, c, 1e-10, 1e-12, &what);
        }
    }
}

type Op = Box<dyn Fn(&Tensor) -> Tensor>;

#[test]
fn reduction_formulas() {
    // 6 x 7 x 8, reshaped by each formula.
    let x = inputs(336, -2.0, 2.0, 1);
    let positive = inputs(336, 0.5, 1.5, 2);
    let r = |t: &Tensor| t.reshape([6, 7, 8]);
    let formulas: Vec<(&str, Op, &Tensor)> = vec![
        ("sum", Box::new(|t: &Tensor| t.mul(t).sum()), &x),
        (
            "sum_dim",
            Box::new(move |t: &Tensor| r(t).mul(r(t)).sum_dim(&[0, 2][..], false)),
            &x,
        ),
        (
            "sum_dim keepdim",
            Box::new(move |t: &Tensor| r(t).exp().sum_dim(1, true)),
            &x,
        ),
        ("mean", Box::new(|t: &Tensor| t.mul(t).mean()), &x),
        (
            "mean_dim",
            Box::new(move |t: &Tensor| r(t).mul(r(t)).mean_dim(&[1][..], false)),
            &x,
        ),
        ("prod", Box::new(|t: &Tensor| t.prod()), &positive),
        (
            "prod_dim",
            Box::new(move |t: &Tensor| r(t).prod_dim(2, false)),
            &positive,
        ),
        ("max", Box::new(|t: &Tensor| t.mul(t).max()), &x),
        ("min", Box::new(|t: &Tensor| t.mul(t).min()), &x),
        ("var", Box::new(|t: &Tensor| t.var(1)), &x),
        (
            "var_dim",
            Box::new(move |t: &Tensor| r(t).var_dim(&[0, 2][..], 0, false)),
            &x,
        ),
        ("std", Box::new(|t: &Tensor| t.std(1)), &x),
        (
            "std_dim",
            Box::new(move |t: &Tensor| r(t).std_dim(1, 1, true)),
            &x,
        ),
        ("softmax", Box::new(move |t: &Tensor| r(t).softmax(2)), &x),
        (
            "softmax dim 0",
            Box::new(move |t: &Tensor| r(t).softmax(0)),
            &x,
        ),
        (
            "log_softmax",
            Box::new(move |t: &Tensor| r(t).log_softmax(-1)),
            &x,
        ),
        (
            "log_softmax dim 1",
            Box::new(move |t: &Tensor| r(t).log_softmax(1)),
            &x,
        ),
    ];
    for (name, f, input) in &formulas {
        compare(name, f.as_ref(), input, 2, false);
    }
}

/// Ties share the gradient of max and min evenly, on both sides.
#[test]
fn ties_share_the_gradient() {
    let v: Vec<f64> = (0..60).map(|i| ((i * 7) % 5) as f64).collect();
    let x = Tensor::from_slice(&v, [60]);
    let max: Op = Box::new(|t: &Tensor| t.max());
    let min: Op = Box::new(|t: &Tensor| t.mul(-1.0).min());
    compare("max with ties", max.as_ref(), &x, 1, true);
    compare("min with ties", min.as_ref(), &x, 1, true);
}

/// The gradient of prod for inputs with zeros (first order only, as on the CPU) multiplies the
/// exclusive cumulative products: the same bits.
#[test]
fn prod_with_zeros() {
    let v: Vec<f64> = (0..48)
        .map(|i| {
            if i % 11 == 3 {
                0.0
            } else {
                0.5 + (i % 7) as f64 * 0.25
            }
        })
        .collect();
    let x = Tensor::from_slice(&v, [48]);
    let one_zero = Tensor::from_slice(&[1.5, 0.0, 2.0, 0.75], [4]);
    let prod_dim: Op = Box::new(|t: &Tensor| t.reshape([6, 8]).prod_dim(1, false));
    let prod_dim0: Op = Box::new(|t: &Tensor| t.reshape([6, 8]).prod_dim(0, true));
    let prod: Op = Box::new(|t: &Tensor| t.prod());
    compare("prod_dim with zeros", prod_dim.as_ref(), &x, 1, true);
    compare("prod_dim 0 with zeros", prod_dim0.as_ref(), &x, 1, true);
    compare("prod with one zero", prod.as_ref(), &one_zero, 1, true);
}

/// Differentiating the gradient of prod for an input with zeros is refused on the GPU as on the
/// CPU, with the same message (the migration table's error case for the prod row).
#[test]
fn prod_with_zeros_refuses_create_graph() {
    let unsupported = "computing the gradient of prod with create_graph for an input that \
                       contains zeros is not supported yet";
    for device in [Device::Cpu, GPU] {
        let x = Tensor::from_slice(&[0.0, 2.0, 3.0, 4.0], [2, 2]).to(device);
        x.requires_grad_(true);
        let y = x.prod_dim(1, false);
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| first(&y, &x, 8, true)))
            .unwrap_err();
        let message = e
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| e.downcast_ref::<&str>().copied());
        assert_eq!(message, Some(unsupported), "{device}");
    }
}

/// The gradients of broadcast operands sum over the broadcast dimensions.
#[test]
fn broadcast_operands() {
    let x = inputs(40, 0.5, 2.0, 3);
    let row = inputs(8, 0.5, 2.0, 4);
    let col = inputs(5, 0.5, 2.0, 5);
    let with = |o: &Tensor, shape: [usize; 2], f: fn(&Tensor, &Tensor) -> Tensor| -> Op {
        let o = o.reshape(shape);
        Box::new(move |t: &Tensor| f(&t.reshape([5, 8]), &o.to(t.device())))
    };
    let formulas: Vec<(&str, Op)> = vec![
        ("add a row", with(&row, [1, 8], |t, o| t.add(o).mul(t))),
        ("mul a column", with(&col, [5, 1], |t, o| t.mul(o).mul(t))),
        ("div by a row", with(&row, [1, 8], |t, o| o.div(t))),
    ];
    for (name, f) in &formulas {
        compare(name, f.as_ref(), &x, 2, false);
    }
    // The broadcast operand's own gradient.
    let x = x.reshape([5, 8]);
    let grad_of_row = |device: Device| {
        let o = row.reshape([1, 8]).to(device);
        o.requires_grad_(true);
        let y = x.to(device).mul(&o).mul(&o);
        first(&y, &o, 7, false).to(Device::Cpu)
    };
    common::assert_close_tol(
        &grad_of_row(GPU),
        &grad_of_row(Device::Cpu),
        1e-10,
        1e-12,
        "row gradient",
    );
}
