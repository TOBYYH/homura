"""Generates tests/golden/llama_layer.safetensors, one Llama decoder layer forward pass, and
tests/golden/llama_layer_grads.safetensors, a backward pass through it.

The reference is written with plain tensor operations and follows the math of Hugging Face's
`LlamaDecoderLayer` (RMSNorm, grouped-query attention with RoPE and a causal mask, SwiGLU MLP).
src/autograd/test_support/llama.rs implements the same layer with homura, and tests/llama_layer.rs
compares the outputs and the gradients.

Usage: python tools/golden/gen_llama_layer.py (needs torch and safetensors).
"""

import math
from pathlib import Path

import torch
from safetensors.torch import save_file

import reference

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden"
OUT = GOLDEN / "llama_layer.safetensors"
GRADS_OUT = GOLDEN / "llama_layer_grads.safetensors"

# Keep in sync with src/autograd/test_support/llama.rs.
BATCH, SEQ, HIDDEN = 2, 7, 64
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 16
INTERMEDIATE = 160
EPS = 1e-6
ROPE_THETA = 10000.0

reference.check()
torch.manual_seed(0)


def weight(*shape, std=0.08):
    return torch.randn(*shape) * std


w = {
    "input_layernorm.weight": 1 + 0.1 * torch.randn(HIDDEN),
    "self_attn.q_proj.weight": weight(HEADS * HEAD_DIM, HIDDEN),
    "self_attn.k_proj.weight": weight(KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.v_proj.weight": weight(KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.o_proj.weight": weight(HIDDEN, HEADS * HEAD_DIM),
    "post_attention_layernorm.weight": 1 + 0.1 * torch.randn(HIDDEN),
    "mlp.gate_proj.weight": weight(INTERMEDIATE, HIDDEN),
    "mlp.up_proj.weight": weight(INTERMEDIATE, HIDDEN),
    "mlp.down_proj.weight": weight(HIDDEN, INTERMEDIATE),
}
x = torch.randn(BATCH, SEQ, HIDDEN)


def rms_norm(h, weight):
    variance = h.pow(2).mean(-1, keepdim=True)
    return weight * (h * torch.rsqrt(variance + EPS))


def rotate_half(h):
    h1, h2 = h[..., : HEAD_DIM // 2], h[..., HEAD_DIM // 2 :]
    return torch.cat((-h2, h1), dim=-1)


def rope(dtype):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).to(dtype) / HEAD_DIM))
    positions = torch.arange(SEQ).to(dtype)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def layer(x, w):
    """Returns the hidden state after the attention block, the attention probabilities and the
    output; tests/llama_layer.rs keeps the probabilities' gradient too."""
    cos, sin = rope(x.dtype)
    residual = x
    h = rms_norm(x, w["input_layernorm.weight"])
    q = (h @ w["self_attn.q_proj.weight"].T).view(BATCH, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    k = (h @ w["self_attn.k_proj.weight"].T).view(BATCH, SEQ, KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = (h @ w["self_attn.v_proj.weight"].T).view(BATCH, SEQ, KV_HEADS, HEAD_DIM).transpose(1, 2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    rep = HEADS // KV_HEADS
    k = k[:, :, None].expand(BATCH, KV_HEADS, rep, SEQ, HEAD_DIM).reshape(BATCH, HEADS, SEQ, HEAD_DIM)
    v = v[:, :, None].expand(BATCH, KV_HEADS, rep, SEQ, HEAD_DIM).reshape(BATCH, HEADS, SEQ, HEAD_DIM)
    scores = q @ k.transpose(2, 3) / math.sqrt(HEAD_DIM)
    causal = torch.ones(SEQ, SEQ, dtype=torch.bool).triu(1)
    probs = scores.masked_fill(causal, float("-inf")).softmax(-1)
    attn = (probs @ v).transpose(1, 2).reshape(BATCH, SEQ, HEADS * HEAD_DIM)
    h = residual + attn @ w["self_attn.o_proj.weight"].T

    residual = h
    h2 = rms_norm(h, w["post_attention_layernorm.weight"])
    gate = torch.nn.functional.silu(h2 @ w["mlp.gate_proj.weight"].T)
    mlp = (gate * (h2 @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
    out = residual + mlp
    return h, probs, out


# The forward pass, recorded: requiring gradients doesn't change any value.
x.requires_grad_(True)
for t in w.values():
    t.requires_grad_(True)
h, probs, out = layer(x, w)

data = {f"layer.{k}": t.detach().contiguous() for k, t in w.items()}
data["input"] = x.detach()
data["attention_residual"] = h.detach().contiguous()
data["output"] = out.detach().contiguous()
OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT))
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")

# The backward pass (M1): a vector-Jacobian product through both results, drawn after everything
# above so that llama_layer.safetensors stays the same. The gradients of the attention residual
# (the total that reaches it, from both roots) and of the probabilities are kept with retain_grad.
grad_h = torch.randn(BATCH, SEQ, HIDDEN)
grad_out = torch.randn(BATCH, SEQ, HIDDEN)
leaves = [x, *w.values()]


def vjp(h, probs, out, leaves):
    h.retain_grad()
    probs.retain_grad()
    grads = torch.autograd.grad([h, out], leaves, [grad_h.to(h.dtype), grad_out.to(out.dtype)])
    return grads, h.grad, probs.grad


grads, retained_h, retained_probs = vjp(h, probs, out, leaves)
grad_data = {"grad_output.attention_residual": grad_h, "grad_output.output": grad_out}
grad_data["grad.input"] = grads[0].contiguous()
for k, g in zip(w, grads[1:]):
    grad_data[f"grad.layer.{k}"] = g.contiguous()
grad_data["retained.attention_residual"] = retained_h.contiguous()
grad_data["retained.probs"] = retained_probs.contiguous()
save_file(grad_data, str(GRADS_OUT), metadata=reference.metadata())
print(f"wrote {len(grad_data)} tensors to {GRADS_OUT} ({GRADS_OUT.stat().st_size} bytes)")

# For the tolerance of tests/llama_layer.rs: how far the f32 gradients above are from the same
# computation in f64 (printed only).
leaves64 = [t.detach().double().requires_grad_(True) for t in leaves]
w64 = dict(zip(w, leaves64[1:]))
grads64, retained_h64, retained_probs64 = vjp(*layer(leaves64[0], w64), leaves64)
names = ["input", *w, "retained attention_residual", "retained probs"]
worst = 0.0
for name, g32, g64 in zip(
    names, [*grads, retained_h, retained_probs], [*grads64, retained_h64, retained_probs64]
):
    diff = (g32.double() - g64).abs()
    # The share of `atol + rtol * |expected|` that the worst element uses, with the test's tolerance.
    share = (diff / (1e-5 + 1e-4 * g64.abs())).max().item()
    worst = max(worst, share)
    print(f"  f32 vs f64 {name}: max |diff| {diff.max().item():.3g}, tolerance share {share:.3g}")
print(f"worst tolerance share of the f32 reference: {worst:.3g}")
