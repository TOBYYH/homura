"""Generates tests/golden/grads.safetensors: inputs, upstream gradients and PyTorch's gradients for
the operations whose backward formulas are implemented.

Each case "<category>.<op>" stores its inputs, the upstream gradient "g" (random, so that mistakes
in transposes or broadcasting directions show) and "grad_<input>" for each floating point input,
computed with torch.autograd.grad(out, inputs, grad_outputs=g). A gradient that PyTorch returns as
None is not stored. The Rust tests in tests/autograd_golden.rs run the same operation, so case
names and arguments must stay in sync with that file. Categories: "grad" (f32, compared with
atol = rtol = 1e-5), "grad_f16" and "grad_bf16" (a few representative cases, compared within the
dtype's precision), and "special" (non-finite inputs and upstream gradients, compared exactly).

Usage: python tools/golden/gen_grads.py (needs torch and safetensors).
"""

import math
from pathlib import Path

import torch
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "grads.safetensors"

reference.check()
torch.manual_seed(0)
data = {}


def randn(*shape, dtype=torch.float32):
    return torch.randn(shape, dtype=torch.float64).to(dtype)


def case(name, op, g=None, **inputs):
    """Stores the inputs, the upstream gradient and the gradients of `op(**inputs)`."""
    leaves = {k: v.detach().clone().requires_grad_(v.is_floating_point()) for k, v in inputs.items()}
    out = op(**leaves)
    if g is None:
        g = randn(*out.shape, dtype=out.dtype)
    variables = [k for k, v in leaves.items() if v.requires_grad]
    grads = torch.autograd.grad(out, [leaves[k] for k in variables], g, allow_unused=True)
    tensors = dict(inputs, g=g)
    for k, gk in zip(variables, grads):
        if gk is not None:
            tensors[f"grad_{k}"] = gk
    for k, t in tensors.items():
        key = f"{name}/{k}"
        assert key not in data, key
        data[key] = t.detach().contiguous().clone()


def ints(*values):
    return torch.tensor(values, dtype=torch.int64)


nan, inf = math.nan, math.inf
C = 2 / math.sqrt(math.pi)

# ---------------------------------------------------------------- f32
case("grad.neg", lambda x: -x, x=randn(3, 4))
case("grad.exp", lambda x: x.exp(), x=randn(3, 4))
case("grad.erf", lambda x: x.erf(), x=randn(3, 4))
case("grad.sign", lambda x: x.sign(), x=randn(3, 4))
case("grad.add", lambda x, y: x + y, x=randn(3, 1, 4), y=randn(2, 4))
case("grad.sub", lambda x, y: x - y, x=randn(2, 4), y=randn(3, 1, 4))
case("grad.mul", lambda x, y: x * y, x=randn(3, 1, 4), y=randn(2, 4))
case("grad.mul_self", lambda x: x * x, x=randn(5))
case("grad.add_scalar", lambda x: x + 2.5, x=randn(3, 4))
case("grad.rsub_scalar", lambda x: 1.5 - x, x=randn(3, 4))
case("grad.mul_scalar", lambda x: 1.7 * x, x=randn(3, 4))
case("grad.mul_int", lambda x, y: x * y, x=randn(2, 3), y=torch.randint(-3, 4, (3,)))
case("grad.sum", lambda x: x.sum(), x=randn(3, 4))
case("grad.sum_to_size", lambda x: x.sum_to_size(1, 4), x=randn(3, 4))
case("grad.expand", lambda x: x.expand(2, 3, 4), x=randn(3, 1))
case("grad.view", lambda x: x.view(4, 3), x=randn(3, 4))
case("grad.reshape_copy", lambda x: x[:, ::2].reshape(6), x=randn(3, 4))
case("grad.select", lambda x: x.select(-1, -2), x=randn(3, 4))
case("grad.narrow", lambda x: x.narrow(0, 1, 2), x=randn(3, 4))
case("grad.slice_step", lambda x: x[1:, ::3], x=randn(3, 7))
case("grad.index_int", lambda x, idx: x[idx], x=randn(4, 3), idx=torch.tensor([[-1, 0], [0, 2]]))
case("grad.index_select", lambda x, idx: x.index_select(1, idx), x=randn(3, 4), idx=ints(3, 0, 3, 3))
case("grad.as_strided", lambda x: x.as_strided((3, 2), (1, 1), 1), x=randn(6))
case(
    "grad.index_add",
    lambda x, idx, source: x.index_add(0, idx, source),
    x=randn(3, 2),
    idx=ints(2, 0, 2, 2),
    source=randn(4, 2),
)
case("grad.gather", lambda x, idx: x.gather(1, idx), x=randn(3, 4), idx=torch.tensor([[3, 3], [0, 1], [2, 3]]))
case(
    "grad.scatter_add",
    lambda x, idx, src: x.scatter_add(0, idx, src),
    x=randn(3, 4),
    idx=torch.tensor([[2, 0, 2, 1], [2, 2, 0, 0]]),
    src=randn(2, 4),
)
case("grad.select_scatter", lambda x, src: x.select_scatter(src, 1, -1), x=randn(3, 4), src=randn(3))
case(
    "grad.slice_scatter",
    lambda x, src: x.slice_scatter(src, 1, 1, None, 2),
    x=randn(3, 5),
    src=randn(3, 2),
)
case("grad.deep_clone", lambda x: x.clone(), x=randn(3, 4))
case("grad.to_f64", lambda x: x.to(torch.float64), x=randn(3, 4))
case("grad.to_f16", lambda x: x.to(torch.float16), x=randn(3, 4, dtype=torch.float64))

# ---------------------------------------------------------------- f16 and bf16
for dtype, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    case(f"grad_{tag}.mul_scalar", lambda x: x * C, x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.erf", lambda x: x.erf(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.exp", lambda x: x.exp(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.add", lambda x, y: x + y, x=randn(3, 1, 4, dtype=dtype), y=randn(2, 4, dtype=dtype))
    case(f"grad_{tag}.mul", lambda x, y: x * y, x=randn(3, 4, dtype=dtype), y=randn(4, dtype=dtype))
    case(
        f"grad_{tag}.gather",
        lambda x, idx: x.gather(1, idx),
        x=randn(3, 4, dtype=dtype),
        idx=torch.tensor([[1, 3], [0, 0], [2, 1]]),
    )

# ---------------------------------------------------------------- special values (exact)
f = torch.tensor
case(
    "special.mul",
    lambda x, y: x * y,
    x=f([0.0, inf, nan, 1.0, -0.0, 2.0]),
    y=f([inf, 0.0, 1.0, nan, 2.0, -inf]),
    g=f([1.0, 1.0, 1.0, 1.0, inf, 0.0]),
)
case("special.exp", lambda x: x.exp(), x=f([inf, -inf, nan, 0.0, 100.0]), g=f([1.0, inf, 1.0, nan, 0.0]))
case("special.erf", lambda x: x.erf(), x=f([inf, -inf, nan, 0.0, 30.0]), g=f([inf, 1.0, 1.0, nan, inf]))
case("special.sign", lambda x: x.sign(), x=f([0.0, -0.0, nan, inf, -2.0]), g=f([inf, nan, 1.0, -inf, nan]))
case("special.neg", lambda x: -x, x=f([1.0, 2.0, 3.0]), g=f([nan, inf, -0.0]))
case("special.sum", lambda x: x.sum(), x=f([1.0, inf, nan]), g=f(nan))
case("special.add_inf", lambda x, y: x + y, x=f([[1.0], [2.0]]), y=f([inf, -inf]), g=f([[inf, -inf], [1.0, nan]]))
case(
    "special.index_select",
    lambda x, idx: x.index_select(0, idx),
    x=f([1.0, 2.0, 3.0]),
    idx=ints(0, 0, 2),
    g=f([inf, -inf, nan]),
)
case(
    "special.as_strided",
    lambda x: x.as_strided((3,), (0,), 1),
    x=f([1.0, 2.0, 3.0]),
    g=f([inf, 1.0, 2.0]),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
