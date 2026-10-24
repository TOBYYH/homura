//! A Llama decoder layer written with plain tensor operations, shared by the acceptance tests in
//! `tests/llama_layer.rs` (compared with `tools/golden/gen_llama_layer.py`) and the crate's own
//! release check in `src/autograd/llama_tests.rs`, which needs crate-internal weak handles.
//!
//! The file lives under `src/` so that it ships with the package (Cargo excludes `/tests`), but it
//! is compiled only into tests: it uses only the public API, through `homura::` paths (the crate
//! names itself `homura` under `cfg(test)`), and loads no golden data.

// Each test target uses a different part of it.
#![allow(dead_code)]

use homura::{DType, Device, Tensor};

// Keep in sync with tools/golden/gen_llama_layer.py.
pub const BATCH: usize = 2;
pub const SEQ: usize = 7;
pub const HIDDEN: usize = 64;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 16;
pub const INTERMEDIATE: usize = 160;
pub const EPS: f64 = 1e-6;
pub const ROPE_THETA: f64 = 10000.0;

/// The names of the weights, in the order of [`Layer::weights`], as the golden file spells them
/// (after `layer.` and before `.weight`).
pub const WEIGHT_NAMES: [&str; 9] = [
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

pub struct Layer {
    pub input_norm: Tensor,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub post_norm: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

/// The results of [`Layer::forward`].
pub struct Forward {
    /// The hidden state after the attention block (with its residual).
    pub h: Tensor,
    /// The attention probabilities, `[batch, heads, seq, seq]`.
    pub probs: Tensor,
    /// The output of the layer.
    pub out: Tensor,
}

impl Layer {
    /// A layer with the weights `weight(name)`, named as in [`WEIGHT_NAMES`].
    pub fn from_fn(mut weight: impl FnMut(&str) -> Tensor) -> Layer {
        let [
            input_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_norm,
            gate_proj,
            up_proj,
            down_proj,
        ] = WEIGHT_NAMES.map(&mut weight);
        Layer {
            input_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_norm,
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// A layer with deterministic `f32` weights of the right shapes that doesn't use the random
    /// number generator, whose state other tests share.
    pub fn deterministic() -> Layer {
        let filled = |shape: [usize; 2], seed: f64| {
            let n = (shape[0] * shape[1]) as i64;
            (homura::arange_with(n, DType::F32) * 0.37 + seed).sin() * 0.08
        };
        let norm =
            |seed: f64| (homura::arange_with(HIDDEN as i64, DType::F32) + seed).cos() * 0.1 + 1.0;
        let (q, kv) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
        // A copy, so that the weight isn't a view of the flat tensor.
        let matrix = |rows: usize, cols: usize, seed: f64| {
            filled([rows, cols], seed).view([rows, cols]).deep_clone()
        };
        Layer {
            input_norm: norm(0.5),
            q_proj: matrix(q, HIDDEN, 1.0),
            k_proj: matrix(kv, HIDDEN, 2.0),
            v_proj: matrix(kv, HIDDEN, 3.0),
            o_proj: matrix(HIDDEN, q, 4.0),
            post_norm: norm(1.5),
            gate_proj: matrix(INTERMEDIATE, HIDDEN, 5.0),
            up_proj: matrix(INTERMEDIATE, HIDDEN, 6.0),
            down_proj: matrix(HIDDEN, INTERMEDIATE, 7.0),
        }
    }

    /// The weights in the order of [`WEIGHT_NAMES`].
    pub fn weights(&self) -> [&Tensor; 9] {
        [
            &self.input_norm,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.post_norm,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ]
    }

    /// The layer with every weight converted by `f`.
    pub fn map(&self, f: impl Fn(&Tensor) -> Tensor) -> Layer {
        let weights = self.weights();
        let mut next = weights.iter().map(|w| f(w));
        Layer::from_fn(|_| next.next().unwrap())
    }

    pub fn forward(&self, x: &Tensor) -> Forward {
        let (batch, seq) = (x.size(0), x.size(1));

        let h = rms_norm(x, &self.input_norm);
        let q = linear(&h, &self.q_proj)
            .view([batch, seq, HEADS, HEAD_DIM])
            .transpose(1, 2);
        let k = linear(&h, &self.k_proj)
            .view([batch, seq, KV_HEADS, HEAD_DIM])
            .transpose(1, 2);
        let v = linear(&h, &self.v_proj)
            .view([batch, seq, KV_HEADS, HEAD_DIM])
            .transpose(1, 2);

        let (cos, sin) = rope(seq, x.device());
        let q = &q * &cos + rotate_half(&q) * &sin;
        let k = &k * &cos + rotate_half(&k) * &sin;
        let k = repeat_kv(&k, HEADS / KV_HEADS);
        let v = repeat_kv(&v, HEADS / KV_HEADS);

        let scores = q.matmul(&k.transpose(2, 3)) / (HEAD_DIM as f64).sqrt();
        let causal = homura::ones_with([seq, seq], (DType::Bool, x.device())).triu(1);
        let probs = scores.masked_fill(&causal, f64::NEG_INFINITY).softmax(-1);
        let attn = probs
            .matmul(&v)
            .transpose(1, 2)
            .reshape([batch, seq, HEADS * HEAD_DIM]);
        let h = x + linear(&attn, &self.o_proj);

        let h2 = rms_norm(&h, &self.post_norm);
        let mlp = linear(&h2, &self.gate_proj).silu() * linear(&h2, &self.up_proj);
        let out = &h + linear(&mlp, &self.down_proj);
        Forward { h, probs, out }
    }
}

fn linear(x: &Tensor, weight: &Tensor) -> Tensor {
    x.matmul(&weight.transpose(0, 1))
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Tensor {
    let variance = x.pow(2).mean_dim(-1, true);
    weight * (x * (variance + EPS).rsqrt())
}

/// `cos` and `sin` tables of shape `[seq, head_dim]`.
fn rope(seq: usize, device: Device) -> (Tensor, Tensor) {
    let exponents =
        homura::arange_step_with(0, HEAD_DIM as i64, 2, device).to(DType::F32) / HEAD_DIM as f64;
    let inv_freq = 1.0 / homura::full_with([HEAD_DIM / 2], ROPE_THETA, device).pow(&exponents);
    let positions = homura::arange_with(seq as i64, (DType::F32, device));
    let freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
    let emb = homura::concat(&[&freqs, &freqs], -1);
    (emb.cos(), emb.sin())
}

fn rotate_half(x: &Tensor) -> Tensor {
    let half = HEAD_DIM / 2;
    let x1 = x.index((homura::Ellipsis, ..half));
    let x2 = x.index((homura::Ellipsis, half..));
    homura::concat(&[-&x2, x1], -1)
}

/// `[batch, kv_heads, seq, d]` → `[batch, kv_heads * n_rep, seq, d]`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    let s = x.shape();
    let (b, kv, seq, d) = (s[0], s[1], s[2], s[3]);
    x.unsqueeze(2)
        .expand([b, kv, n_rep, seq, d])
        .reshape([b, kv * n_rep, seq, d])
}
