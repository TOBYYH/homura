//! Acceptance tests of a Llama decoder layer written with plain tensor operations
//! (`src/autograd/test_support/llama.rs`): the forward pass (M0) and a backward pass (M1) compared
//! with PyTorch (`tools/golden/gen_llama_layer.py`), and both run again on the meta device. The
//! release check of the layer's graph is in `src/autograd/llama_tests.rs`.

mod common;
#[path = "../src/autograd/test_support/llama.rs"]
mod llama;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{Golden, assert_close_tol};
use homura::autograd::{self, BackwardOptions};
use homura::{DType, Device, Tensor};
use llama::{BATCH, HEADS, HIDDEN, Layer, SEQ, WEIGHT_NAMES};

fn load(g: &Golden) -> Layer {
    Layer::from_fn(|name| g.get(&format!("layer.{name}.weight")))
}

/// `t` as a new leaf that requires grad.
fn leaf(t: &Tensor) -> Tensor {
    let t = t.detach();
    t.requires_grad_(true);
    t
}

fn panic_message(f: impl FnOnce()) -> String {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic message")
}

#[test]
fn llama_decoder_layer_matches_pytorch() {
    let g = Golden::load("llama_layer.safetensors");
    let f = load(&g).forward(&g.get("input"));
    assert_close_tol(
        &f.h,
        &g.get("attention_residual"),
        1e-5,
        1e-5,
        "attention residual",
    );
    assert_close_tol(&f.out, &g.get("output"), 1e-5, 1e-5, "layer output");
}

#[test]
fn llama_decoder_layer_on_meta_device() {
    let g = Golden::load("llama_layer.safetensors");
    let layer = load(&g).map(|w| w.to(Device::Meta));
    let x = g.get("input").to(Device::Meta);
    let f = layer.forward(&x);
    for t in [&f.h, &f.out] {
        assert_eq!(t.shape(), [BATCH, SEQ, HIDDEN]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.device(), Device::Meta);
    }
}

#[test]
fn llama_decoder_layer_gradients_match_pytorch() {
    let g = Golden::load("llama_layer.safetensors");
    let grads = Golden::load("llama_layer_grads.safetensors");
    let layer = load(&g).map(leaf);
    let x = leaf(&g.get("input"));
    let f = layer.forward(&x);
    f.h.retain_grad();
    f.probs.retain_grad();
    // The reference gets the same gradients from torch.autograd.grad.
    autograd::backward(
        &[&f.h, &f.out],
        &[
            Some(&grads.get("grad_output.attention_residual")),
            Some(&grads.get("grad_output.output")),
        ],
        BackwardOptions::default(),
    );
    // The chain is long, so the tolerance is the one 12.3.5 sets for this test, wider than for
    // single operations: `atol = 1e-5, rtol = 1e-4`. Measured when written, homura's worst
    // element (down_proj) used 0.095 of it, and the f32 reference itself, compared with the same
    // computation in f64, 0.097 (printed by the generator).
    let check = |name: &str, actual: &Tensor, key: &str| {
        assert_close_tol(actual, &grads.get(key), 1e-4, 1e-5, name);
    };
    check("input", &x.grad().unwrap(), "grad.input");
    for (name, w) in WEIGHT_NAMES.iter().zip(layer.weights()) {
        check(
            name,
            &w.grad().unwrap(),
            &format!("grad.layer.{name}.weight"),
        );
    }
    // The total gradient of the attention residual, from both roots, and of the probabilities.
    check(
        "attention residual",
        &f.h.grad().unwrap(),
        "retained.attention_residual",
    );
    check("probabilities", &f.probs.grad().unwrap(), "retained.probs");
}

#[test]
fn llama_decoder_layer_gradients_on_meta_leaves() {
    // Leaves created on the meta device: every gradient has the shape, dtype and device of its
    // tensor.
    let g = Golden::load("llama_layer.safetensors");
    let layer = load(&g).map(|w| leaf(&w.to(Device::Meta)));
    let x = leaf(&g.get("input").to(Device::Meta));
    let f = layer.forward(&x);
    f.h.retain_grad();
    f.probs.retain_grad();
    let upstream = homura::ones_with([BATCH, SEQ, HIDDEN], Device::Meta);
    autograd::backward(
        &[&f.h, &f.out],
        &[Some(&upstream), Some(&upstream)],
        BackwardOptions::default(),
    );
    let tensors = std::iter::once(&x)
        .chain(layer.weights())
        .chain([&f.h, &f.probs]);
    for t in tensors {
        let grad = t.grad().expect("a gradient");
        assert_eq!(grad.shape(), t.shape());
        assert_eq!(grad.dtype(), DType::F32);
        assert_eq!(grad.device(), Device::Meta);
    }
    assert_eq!(f.probs.shape(), [BATCH, HEADS, SEQ, SEQ]);
}

#[test]
fn llama_decoder_layer_backward_from_meta_to_cpu_leaves_fails() {
    // CPU leaves moved to the meta device: the backward pass has no data to copy back, and fails
    // like the forward conversion would.
    let g = Golden::load("llama_layer.safetensors");
    let layer = load(&g).map(leaf);
    let x = leaf(&g.get("input"));
    let f = layer
        .map(|w| w.to(Device::Meta))
        .forward(&x.to(Device::Meta));
    assert_eq!(f.out.device(), Device::Meta);
    assert_eq!(
        panic_message(|| f.out.sum().backward()),
        "Cannot copy out of meta tensor; no data!"
    );
    assert!(x.grad().is_none());
}
