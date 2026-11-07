//! M1's elementwise formulas on a ROCm GPU (M4 step 3, item 5): first and second order
//! gradients of the unary, binary and selection operations computed on the GPU match the CPU's.
//! The graphs avoid reductions (step 4a), so the operands have equal shapes and the gradients
//! of the outputs are given explicitly.
#![cfg(feature = "rocm")]

mod common;
#[path = "common/rocm.rs"]
mod rocm;

use homura::autograd::{self, GradOptions};
use homura::{DType, GeluApproximate, Tensor};
use rocm::{GPU, Words};

fn inputs(n: usize, low: f64, high: f64, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    let v: Vec<f64> = (0..n)
        .map(|_| low + (w.next() >> 11) as f64 / (1u64 << 53) as f64 * (high - low))
        .collect();
    Tensor::from_slice(&v, [n])
}

/// `d<y, v>/dx` (and, with `create_graph`, its graph) for `y = f(x)`.
fn first(y: &Tensor, x: &Tensor, v: &Tensor, create_graph: bool) -> Tensor {
    autograd::grad(
        &[y],
        &[x],
        GradOptions {
            grad_outputs: vec![Some(v.clone())],
            create_graph,
            ..Default::default()
        },
    )
    .remove(0)
    .expect("a gradient")
}

/// The first and second order gradients of `f` at `x` on `device`, as CPU tensors.
fn gradients(f: &dyn Fn(&Tensor) -> Tensor, x: &Tensor, device: homura::Device) -> [Tensor; 2] {
    let x = x.to(device);
    x.requires_grad_(true);
    let v = inputs(x.num_elements(), -1.0, 1.0, 5).to(device);
    let w = inputs(x.num_elements(), -1.0, 1.0, 6).to(device);
    let y = f(&x);
    let g = first(&y, &x, &v, true);
    let gg = first(&g, &x, &w, false);
    [g, gg].map(|t| t.to(homura::Device::Cpu))
}

/// F64 gradients: within `(1e-10, 1e-12)` (the M4 acceptance tolerance for F64), and the same
/// bits for the formulas built only from IEEE operations.
fn compare(name: &str, f: &dyn Fn(&Tensor) -> Tensor, x: &Tensor, bitwise: bool) {
    let cpu = gradients(f, x, homura::Device::Cpu);
    let gpu = gradients(f, x, GPU);
    for (order, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        let what = format!("{name}: order {}", order + 1);
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

#[test]
fn unary_formulas() {
    let x = inputs(1031, 0.25, 3.0, 1);
    let signed = inputs(1031, -3.0, 3.0, 2);
    type Op = fn(&Tensor) -> Tensor;
    let ieee: [(&str, Op); 6] = [
        ("neg", |t| t.neg().mul(t)),
        ("abs", |t| t.abs().mul(t)),
        ("sqrt", |t| t.sqrt()),
        ("rsqrt", |t| t.rsqrt()),
        ("reciprocal", |t| t.reciprocal()),
        ("relu", |t| t.relu().mul(t)),
    ];
    for (name, f) in ieee {
        compare(name, &f, &x, true);
        compare(name, &f, &signed.abs().add(0.1), true);
    }
    let library: [(&str, Op); 11] = [
        ("exp", |t| t.exp()),
        ("log", |t| t.log()),
        ("sin", |t| t.sin()),
        ("cos", |t| t.cos()),
        ("erf", |t| t.erf()),
        ("tanh", |t| t.tanh()),
        ("sigmoid", |t| t.sigmoid()),
        ("log_sigmoid", |t| t.log_sigmoid()),
        ("silu", |t| t.silu()),
        ("gelu", |t| t.gelu(GeluApproximate::None)),
        ("gelu_tanh", |t| t.gelu(GeluApproximate::Tanh)),
    ];
    for (name, f) in library {
        compare(name, &f, &x, false);
        if name != "log" {
            compare(name, &f, &signed, false);
        }
    }
}

#[test]
fn binary_and_selection_formulas() {
    let x = inputs(1031, 0.25, 3.0, 3);
    let other = inputs(1031, 0.25, 3.0, 4).to(DType::F64);
    type Op = Box<dyn Fn(&Tensor) -> Tensor>;
    let with = |f: fn(&Tensor, &Tensor) -> Tensor| -> Op {
        let o = other.clone();
        Box::new(move |t: &Tensor| f(t, &o.to(t.device())))
    };
    let ieee: Vec<(&str, Op)> = vec![
        ("add", with(|t, o| t.add(o).mul(t))),
        ("sub", with(|t, o| t.sub(o).mul(t))),
        ("mul", with(|t, o| t.mul(o).mul(t))),
        ("div", with(|t, o| o.div(t))),
        ("maximum", with(|t, o| t.maximum(o).mul(t))),
        ("minimum", with(|t, o| t.minimum(o).mul(t))),
        (
            "where",
            with(|t, o| homura::where_(&t.greater(o), t.mul(t), o.mul(t))),
        ),
        ("clamp", Box::new(|t: &Tensor| t.clamp(0.5, 2.0).mul(t))),
        (
            "masked_fill",
            with(|t, o| t.mul(t).masked_fill(&t.less(o), 0.5)),
        ),
        (
            "triu",
            Box::new(|t: &Tensor| {
                let m = t.reshape([32, 32]);
                m.mul(&m).triu(1).reshape([1024])
            }),
        ),
        ("flip", Box::new(|t: &Tensor| t.mul(t).flip([0]))),
    ];
    for (name, f) in &ieee {
        // The triangle is of a 32 x 32 matrix.
        let x = if *name == "triu" {
            x.narrow(0, 0, 1024).contiguous()
        } else {
            x.clone()
        };
        compare(name, f.as_ref(), &x, true);
    }
    let pow: Vec<(&str, Op)> = vec![
        ("pow", with(|t, o| t.pow(o))),
        ("pow scalar", Box::new(|t: &Tensor| t.pow(2.5))),
        ("rpow", with(|t, o| o.pow(t))),
    ];
    for (name, f) in &pow {
        compare(name, f.as_ref(), &x, false);
    }
}
