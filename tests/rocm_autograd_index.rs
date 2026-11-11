//! The gradients of indexing on a ROCm GPU (M4 step 4b): index_select, gather, index_add,
//! scatter_add, advanced indexing and index_put (with and without accumulation), Embedding,
//! nll_loss, and max_dim / min_dim (whose gradient scatters into the indices; step 4a left it to
//! this step). Their first and second order gradients are made of the indexing operations and
//! sums in f64 in the CPU's order, so they have the CPU's bits.
#![cfg(feature = "rocm")]

#[path = "common/rocm.rs"]
mod rocm;

use homura::autograd::{self, GradOptions};
use homura::nn::functional as F;
use homura::{DType, Device, Tensor};
use rocm::{GPU, Words, assert_bitwise};

fn inputs(n: usize, seed: u64) -> Tensor {
    let mut w = Words::new(seed);
    let v: Vec<f64> = (0..n)
        .map(|_| (w.next() >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0)
        .collect();
    Tensor::from_slice(&v, [n])
}

fn ints(v: &[i64]) -> Tensor {
    Tensor::from_slice(v, [v.len()])
}

/// `d<y, v>/dx` for `y = f(x)`, with its graph when `create_graph`.
fn first(y: &Tensor, x: &Tensor, seed: u64, create_graph: bool) -> Tensor {
    let v = inputs(y.num_elements(), seed)
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

/// The first and second order gradients of `f` at `x` on `device`; `f` gets the device.
fn gradients(f: &dyn Fn(&Tensor, Device) -> Tensor, x: &Tensor, device: Device) -> Vec<Tensor> {
    let x = x.to(device);
    x.requires_grad_(true);
    // f(x)^2 keeps the second order gradient nonzero for linear f.
    let y = f(&x, device);
    let y = y.mul(&y);
    let g = first(&y, &x, 5, true);
    let h = first(&g, &x, 6, false);
    vec![g, h]
}

#[track_caller]
fn compare(name: &str, f: &dyn Fn(&Tensor, Device) -> Tensor, x: &Tensor) {
    let cpu = gradients(f, x, Device::Cpu);
    let gpu = gradients(f, x, GPU);
    for (k, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        assert_bitwise(g, c, &format!("{name}: order {}", k + 1));
    }
}

#[test]
fn indexing_gradients_match_the_cpu() {
    let x = inputs(5 * 6, 1).reshape([5, 6]);
    let rep = || ints(&[4, 0, 4, 2, 4]);
    compare("index_select", &|x, d| x.index_select(0, &rep().to(d)), &x);
    let gi = || Tensor::from_slice(&[0i64, 5, 5, 1, 2, 5, 3, 3], [2, 4]);
    compare("gather", &|x, d| x.gather(1, &gi().to(d)), &x);
    let src = inputs(5 * 6, 2).reshape([5, 6]);
    compare(
        "index_add (self)",
        &|x, d| x.index_add(0, &rep().to(d), &src.to(d)),
        &x,
    );
    compare(
        "index_add (source)",
        &|s, d| homura::zeros_with([5, 6], (DType::F64, d)).index_add(0, &rep().to(d), s),
        &src,
    );
    compare(
        "scatter_add (source)",
        &|s, d| {
            homura::zeros_with([2, 6], (DType::F64, d)).scatter_add(
                1,
                &gi().to(d),
                &s.narrow(0, 0, 2).narrow(1, 0, 4),
            )
        },
        &src,
    );
    compare(
        "advanced index",
        &|x, d| x.index((&ints(&[1, 1, -1]).to(d), &ints(&[0, 5, 0]).to(d))),
        &x,
    );
    compare(
        "advanced index with a mask",
        &|x, d| {
            let m = x.detach().greater(0.0);
            x.index((&m.to(d),))
        },
        &x,
    );
    let v = inputs(3, 3);
    for accumulate in [false, true] {
        compare(
            &format!("index_put (target), accumulate {accumulate}"),
            &|x, d| {
                let i = (&ints(&[2, 2, 0]).to(d),);
                let x = x.reshape([-1]).narrow(0, 0, 5);
                if accumulate {
                    x.index_put_accumulate(i, &v.to(d))
                } else {
                    x.index_put(i, &v.to(d))
                }
            },
            &x,
        );
        compare(
            &format!("index_put (values), accumulate {accumulate}"),
            &|v, d| {
                let x = homura::zeros_with([5], (DType::F64, d));
                let i = (&ints(&[2, 2, 0]).to(d),);
                if accumulate {
                    x.index_put_accumulate(i, v)
                } else {
                    x.index_put(i, v)
                }
            },
            &v,
        );
    }
}

#[test]
fn max_dim_and_min_dim_gradients_scatter_into_the_indices() {
    // Ties: the gradient goes to the position the forward reported.
    let v: Vec<f64> = (0..4 * 7).map(|i| ((i * 5) % 3) as f64).collect();
    let x = Tensor::from_slice(&v, [4, 7]);
    for keepdim in [false, true] {
        for dim in [0, 1, -1] {
            compare(
                &format!("max_dim {dim} keepdim {keepdim}"),
                &|x, _| x.max_dim(dim, keepdim).values,
                &x,
            );
            compare(
                &format!("min_dim {dim} keepdim {keepdim}"),
                &|x, _| x.min_dim(dim, keepdim).values,
                &x,
            );
        }
    }
}

#[test]
fn embedding_and_nll_gradients_match_the_cpu() {
    let weight = inputs(6 * 4, 4).reshape([6, 4]);
    let input = || Tensor::from_slice(&[5i64, 0, 5, 2, 5, 1], [2, 3]);
    for padding_idx in [None, Some(5)] {
        compare(
            &format!("embedding padding {padding_idx:?}"),
            &|w, d| F::embedding(&input().to(d), w, padding_idx),
            &weight,
        );
    }
    // Log-probabilities given directly: no transcendental functions, so the gradients have the
    // CPU's bits.
    let logp = inputs(5 * 4, 5).reshape([5, 4]);
    let target = || ints(&[3, -100, 0, 3, 1]);
    let w = inputs(4, 6).abs();
    for reduction in [F::Reduction::Sum, F::Reduction::Mean, F::Reduction::None] {
        compare(
            &format!("nll_loss {reduction:?}"),
            &|x, d| F::nll_loss(x, &target().to(d), None, -100, reduction),
            &logp,
        );
        compare(
            &format!("nll_loss weighted {reduction:?}"),
            &|x, d| F::nll_loss(x, &target().to(d), Some(&w.to(d)), -100, reduction),
            &logp,
        );
    }
}
