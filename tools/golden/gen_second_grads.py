"""Generates tests/golden/second_grads.safetensors: second derivatives of softmax and log_softmax.

The reference records the first derivative as its own node (SoftmaxBackwardDataBackward0 and
LogSoftmaxBackwardDataBackward0) whose formulas fix the order of operations, so that large finite
upstream gradients overflow to NaN or stay finite in particular places (review round 26, R40).
Each case "<dtype>.<op>_<dim>_<kind>" stores the input "x", the upstream gradients "g" (first
pass) and "r" (second pass), the first derivative "a" = grad(y, x, g, create_graph=True), and
"dx", "dy" and "dg", the gradients of a with respect to x, the result y and g for upstream r.
"<dim>" is "last" (shape [2, 4], dim 1) or "first" (shape [4, 2], dim 0); "<kind>" is "normal"
(random values) or "large" (equal finite values of g near the top of the dtype's range).

homura's half precision forward computes in f64 and rounds once (M0), the reference's in f32, so
the half precision inputs are drawn until both forwards agree; the gradients then depend on the
backward formulas only. tests/autograd_softmax.rs reads the file.

Usage: python tools/golden/gen_second_grads.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "second_grads.safetensors"

reference.check()
torch.manual_seed(0)
data = {}

DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}
# Large upstream gradients: softmax's second derivatives with respect to x are NaN, log_softmax's
# are finite, although multiplying r by the sum of g first would overflow.
LARGE = {
    ("f16", "softmax"): 60000.0,
    ("bf16", "softmax"): 3e38,
    ("f32", "softmax"): 3e38,
    ("f64", "softmax"): 1e308,
    ("f16", "log_softmax"): 15000.0,
    ("bf16", "log_softmax"): 5e37,
    ("f32", "log_softmax"): 5e37,
    ("f64", "log_softmax"): 2.5e307,
}


def forward(x, op, dim):
    return x.log_softmax(dim) if op == "log_softmax" else x.softmax(dim)


def agrees(x, op, dim):
    """Whether homura's forward (in f64, rounded once) gives the reference's result."""
    return torch.equal(forward(x.double(), op, dim).to(x.dtype), forward(x, op, dim))


def case(name, x, g, r, op, dim):
    x = x.detach().clone().requires_grad_()
    g = g.detach().clone().requires_grad_()
    y = forward(x, op, dim)
    (a,) = torch.autograd.grad(y, x, g, create_graph=True)
    dx, dy, dg = torch.autograd.grad(a, [x, y, g], r)
    for k, t in dict(x=x, g=g, r=r, a=a, dx=dx, dy=dy, dg=dg).items():
        data[f"{name}/{k}"] = t.detach().contiguous().clone()


for tag, dtype in DTYPES.items():
    for op in ["softmax", "log_softmax"]:
        for where, shape, dim in [("last", (2, 4), 1), ("first", (4, 2), 0)]:
            for _ in range(1000):
                x = torch.randn(shape, dtype=torch.float64).to(dtype)
                if agrees(x, op, dim):
                    break
            else:
                raise SystemExit(f"no agreeing input for {tag} {op}")
            g = torch.randn(shape, dtype=torch.float64).to(dtype)
            r = torch.randn(shape, dtype=torch.float64).to(dtype)
            case(f"{tag}.{op}_{where}_normal", x, g, r, op, dim)
            x = torch.zeros(shape, dtype=dtype)
            assert agrees(x, op, dim)
            g = torch.full(shape, LARGE[tag, op], dtype=dtype)
            r = torch.tensor([[0.0, 2.0, 0.0, 1.0], [2.0, 0.0, -1.0, 0.0]], dtype=dtype)
            if dim == 0:
                r = r.t()
            case(f"{tag}.{op}_{where}_large", x, g, r, op, dim)

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
