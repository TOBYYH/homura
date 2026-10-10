//! M0 acceptance test: a Llama decoder layer written with plain tensor operations, compared with
//! PyTorch (`tools/golden/gen_llama_layer.py`), and run again on the meta device.

mod common;

use common::{Golden, assert_close_tol};
use homura::{DType, Device, Tensor};

// Keep in sync with tools/golden/gen_llama_layer.py.
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

struct Layer {
    input_norm: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    post_norm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Layer {
    fn load(g: &Golden) -> Layer {
        let w = |name: &str| g.get(&format!("layer.{name}.weight"));
        Layer {
            input_norm: w("input_layernorm"),
            q_proj: w("self_attn.q_proj"),
            k_proj: w("self_attn.k_proj"),
            v_proj: w("self_attn.v_proj"),
            o_proj: w("self_attn.o_proj"),
            post_norm: w("post_attention_layernorm"),
            gate_proj: w("mlp.gate_proj"),
            up_proj: w("mlp.up_proj"),
            down_proj: w("mlp.down_proj"),
        }
    }

    fn to(&self, device: Device) -> Layer {
        let t = |x: &Tensor| x.to(device);
        Layer {
            input_norm: t(&self.input_norm),
            q_proj: t(&self.q_proj),
            k_proj: t(&self.k_proj),
            v_proj: t(&self.v_proj),
            o_proj: t(&self.o_proj),
            post_norm: t(&self.post_norm),
            gate_proj: t(&self.gate_proj),
            up_proj: t(&self.up_proj),
            down_proj: t(&self.down_proj),
        }
    }

    /// Returns the hidden state after the attention block and the final output.
    fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let (batch, seq) = (x.size(0) as i64, x.size(1) as i64);
        let (heads, kv_heads, head_dim) = (HEADS as i64, KV_HEADS as i64, HEAD_DIM as i64);

        let h = rms_norm(x, &self.input_norm);
        let q = linear(&h, &self.q_proj)
            .view([batch, seq, heads, head_dim])
            .transpose(1, 2);
        let k = linear(&h, &self.k_proj)
            .view([batch, seq, kv_heads, head_dim])
            .transpose(1, 2);
        let v = linear(&h, &self.v_proj)
            .view([batch, seq, kv_heads, head_dim])
            .transpose(1, 2);

        let (cos, sin) = rope(seq as usize, x.device());
        let q = &q * &cos + rotate_half(&q) * &sin;
        let k = &k * &cos + rotate_half(&k) * &sin;
        let k = repeat_kv(&k, HEADS / KV_HEADS);
        let v = repeat_kv(&v, HEADS / KV_HEADS);

        let scores = q.matmul(&k.transpose(2, 3)) / (HEAD_DIM as f64).sqrt();
        let causal =
            homura::ones_with([seq as usize, seq as usize], (DType::Bool, x.device())).triu(1);
        let probs = scores.masked_fill(&causal, f64::NEG_INFINITY).softmax(-1);
        let attn = probs
            .matmul(&v)
            .transpose(1, 2)
            .reshape([batch, seq, heads * head_dim]);
        let h = x + linear(&attn, &self.o_proj);

        let h2 = rms_norm(&h, &self.post_norm);
        let mlp = linear(&h2, &self.gate_proj).silu() * linear(&h2, &self.up_proj);
        let out = &h + linear(&mlp, &self.down_proj);
        (h, out)
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
    let half = (HEAD_DIM / 2) as i64;
    let x1 = x.index((homura::Ellipsis, ..half));
    let x2 = x.index((homura::Ellipsis, half..));
    homura::concat(&[-&x2, x1], -1)
}

/// `[batch, kv_heads, seq, d]` → `[batch, kv_heads * n_rep, seq, d]`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    let s = x.shape();
    let (b, kv, seq, d) = (s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64);
    x.unsqueeze(2)
        .expand([b, kv, n_rep as i64, seq, d])
        .reshape([b, kv * n_rep as i64, seq, d])
}

#[test]
fn llama_decoder_layer_matches_pytorch() {
    let g = Golden::load("llama_layer.safetensors");
    let layer = Layer::load(&g);
    let (h, out) = layer.forward(&g.get("input"));
    assert_close_tol(
        &h,
        &g.get("attention_residual"),
        1e-5,
        1e-5,
        "attention residual",
    );
    assert_close_tol(&out, &g.get("output"), 1e-5, 1e-5, "layer output");
}

#[test]
fn llama_decoder_layer_on_meta_device() {
    let g = Golden::load("llama_layer.safetensors");
    let layer = Layer::load(&g).to(Device::Meta);
    let x = g.get("input").to(Device::Meta);
    let (h, out) = layer.forward(&x);
    for t in [&h, &out] {
        assert_eq!(t.shape(), [2, 7, 64]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.device(), Device::Meta);
    }
}
