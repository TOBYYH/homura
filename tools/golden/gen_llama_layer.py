"""Generates tests/golden/llama_layer.safetensors: one Llama decoder layer forward pass.

The reference is written with plain tensor operations and follows the math of Hugging Face's
`LlamaDecoderLayer` (RMSNorm, grouped-query attention with RoPE and a causal mask, SwiGLU MLP).
tests/llama_layer.rs implements the same layer with homura and compares the outputs.

Usage: python tools/golden/gen_llama_layer.py (needs torch and safetensors).
"""

import math
from pathlib import Path

import torch
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "llama_layer.safetensors"

# Keep in sync with tests/llama_layer.rs.
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


inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM))
positions = torch.arange(SEQ).float()
freqs = positions[:, None] * inv_freq[None, :]
emb = torch.cat((freqs, freqs), dim=-1)
cos, sin = emb.cos(), emb.sin()

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

data = {f"layer.{k}": t.contiguous() for k, t in w.items()}
data["input"] = x
data["attention_residual"] = h.contiguous()
data["output"] = out.contiguous()
OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT))
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
