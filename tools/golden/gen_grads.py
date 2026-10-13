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

import itertools
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "grads.safetensors"

reference.check()
torch.manual_seed(0)
data = {}


def randn(*shape, dtype=torch.float32):
    return torch.randn(shape, dtype=torch.float64).to(dtype)


def positive(*shape, dtype=torch.float32):
    """Random values between 0.1 and about 4, for functions defined for positive inputs only."""
    return (torch.randn(shape, dtype=torch.float64).abs() + 0.1).to(dtype)


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

# ---------------------------------------------------------------- step 6a
# Appended after the earlier random cases, so that those keep their inputs.
# Unary functions, and the div and where formulas they use.
case("grad.abs", lambda x: x.abs(), x=randn(3, 4))
case("grad.relu", lambda x: x.relu(), x=randn(3, 4))
case("grad.log", lambda x: x.log(), x=positive(3, 4))
case("grad.sqrt", lambda x: x.sqrt(), x=positive(3, 4))
case("grad.rsqrt", lambda x: x.rsqrt(), x=positive(3, 4))
case("grad.reciprocal", lambda x: x.reciprocal(), x=randn(3, 4))
case("grad.sin", lambda x: x.sin(), x=randn(3, 4))
case("grad.cos", lambda x: x.cos(), x=randn(3, 4))
case("grad.tanh", lambda x: x.tanh(), x=randn(3, 4))
case("grad.sigmoid", lambda x: x.sigmoid(), x=randn(3, 4))
case("grad.silu", lambda x: F.silu(x), x=randn(3, 4))
case("grad.gelu", lambda x: F.gelu(x), x=randn(3, 4))
case("grad.gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), x=randn(3, 4))
case("grad.div", lambda x, y: x / y, x=randn(3, 1, 4), y=randn(2, 4))
case("grad.div_scalar", lambda x: x / 1.7, x=randn(3, 4))
# Python computes `1.5 / x` as `x.reciprocal() * 1.5`; homura divides (equal within tolerance).
case("grad.rdiv_scalar", lambda x: 1.5 / x, x=randn(3, 4))
case("grad.div_int", lambda x, y: x / y, x=randn(2, 3), y=ints(3, -2, 5))
case("grad.int_div", lambda x, y: x / y, x=torch.tensor([[3, -2, 5]]), y=randn(2, 3))
case(
    "grad.where",
    lambda cond, x, y: torch.where(cond, x, y),
    cond=torch.tensor([[True], [False], [True]]),
    x=randn(4),
    y=randn(3, 4),
)
case(
    "grad.where_scalar",
    lambda cond, x: torch.where(cond, 0.5, x),
    cond=torch.tensor([True, False, True, True]),
    x=randn(3, 4),
)

# Half precision: PyTorch computes the tanh, sigmoid and silu gradients in one kernel that rounds
# once, while homura rounds after each operation. tanh and sigmoid still agree within a rounding
# step; silu is left out because near x = -1.28, where `1 + x * (1 - sigmoid(x))` cancels, the
# small results differ by several steps. gelu follows below.
for dtype, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    case(f"grad_{tag}.relu", lambda x: x.relu(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.sqrt", lambda x: x.sqrt(), x=positive(3, 4, dtype=dtype))
    case(f"grad_{tag}.reciprocal", lambda x: x.reciprocal(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.sin", lambda x: x.sin(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.tanh", lambda x: x.tanh(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.sigmoid", lambda x: x.sigmoid(), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.div", lambda x, y: x / y, x=randn(3, 1, 4, dtype=dtype), y=randn(2, 4, dtype=dtype))
    case(f"grad_{tag}.div_scalar", lambda x: x / 1.7, x=randn(3, 4, dtype=dtype))
    case(
        f"grad_{tag}.where",
        lambda cond, x, y: torch.where(cond, x, y),
        cond=torch.tensor([[True], [False], [True]]),
        x=randn(4, dtype=dtype),
        y=randn(3, 4, dtype=dtype),
    )

# Half precision gelu (review round 14): like PyTorch's kernel, homura computes these gradients in
# f32 and rounds them once; the inputs include +-300, where x² overflows f16. The cases use
# PyTorch's native kernels: on CPUs like this one, the default dispatch sends the exact form to
# oneDNN, which rejects half precision backward passes. oneDNN is disabled for these cases only;
# for the tanh approximation both paths give the same results.
with torch.backends.mkldnn.flags(enabled=False):
    for dtype, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
        x = torch.cat([randn(10, dtype=dtype), torch.tensor([-300.0, 300.0], dtype=dtype)]).view(3, 4)
        case(f"grad_{tag}.gelu", lambda x: F.gelu(x), x=x)
        case(f"grad_{tag}.gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), x=x)

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
# Step 6a. Results are exact: zeros, infinities, NaN, or values that are exact in f32.
case(
    "special.abs",
    lambda x: x.abs(),
    x=f([0.0, -0.0, -1.0, 2.0, nan, inf, -inf]),
    g=f([inf, nan, inf, nan, 1.0, 2.0, 3.0]),
)
case(
    "special.relu",
    lambda x: x.relu(),
    x=f([-1.0, 0.0, -0.0, 2.0, nan, inf, -inf, 1e-30, -1e-30]),
    g=f([inf, nan, inf, nan, 3.0, inf, nan, inf, inf]),
)
case(
    "special.log",
    lambda x: x.log(),
    x=f([0.0, -0.0, -1.0, inf, -inf, nan, 2.0]),
    g=f([1.0, 1.0, inf, 1.0, 1.0, 1.0, 0.0]),
)
case("special.sqrt", lambda x: x.sqrt(), x=f([0.0, -0.0, -1.0, inf, 4.0]), g=f([1.0, 1.0, 1.0, 1.0, inf]))
case("special.rsqrt", lambda x: x.rsqrt(), x=f([0.0, -0.0, -1.0, inf, 4.0]), g=f([1.0, 1.0, 1.0, 1.0, inf]))
case(
    "special.reciprocal",
    lambda x: x.reciprocal(),
    x=f([0.0, -0.0, inf, -inf, nan, 2.0]),
    g=f([1.0, 1.0, 1.0, 1.0, 1.0, inf]),
)
case("special.sin", lambda x: x.sin(), x=f([inf, nan, 0.0, -0.0]), g=f([1.0, 1.0, inf, nan]))
case("special.cos", lambda x: x.cos(), x=f([inf, nan, 0.0, -0.0]), g=f([1.0, 1.0, inf, 2.0]))
case(
    "special.tanh",
    lambda x: x.tanh(),
    x=f([inf, -inf, nan, 0.0, 20.0]),
    g=f([1.0, inf, 1.0, inf, inf]),
)
case(
    "special.sigmoid",
    lambda x: x.sigmoid(),
    x=f([inf, -inf, nan, 0.0, 100.0, -100.0]),
    g=f([1.0, inf, 1.0, inf, inf, 1.0]),
)
case("special.silu", lambda x: F.silu(x), x=f([inf, -inf, nan, 0.0]), g=f([1.0, 1.0, 1.0, inf]))
case("special.gelu", lambda x: F.gelu(x), x=f([inf, -inf, nan, 0.0]), g=f([1.0, 1.0, 1.0, inf]))
case(
    "special.gelu_tanh",
    lambda x: F.gelu(x, approximate="tanh"),
    x=f([inf, -inf, nan, 0.0]),
    g=f([1.0, 1.0, 1.0, inf]),
)
case(
    "special.div",
    lambda x, y: x / y,
    x=f([1.0, 0.0, inf, 1.0, -1.0, 2.0]),
    y=f([0.0, 0.0, inf, inf, -0.0, 4.0]),
    g=f([1.0, 1.0, 1.0, 1.0, 1.0, inf]),
)
case("special.rdiv_zero", lambda x: 0.0 / x, x=f([0.0, 1.0, inf]), g=f([1.0, 1.0, 1.0]))
case(
    "special.where",
    lambda cond, x, y: torch.where(cond, x, y),
    cond=f([True, False, True, False]),
    x=f([1.0, 2.0, 3.0, 4.0]),
    y=f([5.0, 6.0, 7.0, 8.0]),
    g=f([nan, inf, -inf, nan]),
)

# Step 6b is appended after all earlier cases to preserve their random inputs and gradients.
case("grad.pow", lambda x, y: x.pow(y), x=positive(3, 4), y=randn(4))
case("grad.pow_scalar", lambda x: x.pow(2.5), x=positive(3, 4))
case("grad.pow_zero", lambda x: x.pow(0), x=randn(3, 4))
case("grad.pow_int", lambda x, y: x.pow(y), x=randn(3, 4), y=ints(0, 1, 2, 3))
case("grad.int_pow", lambda x, y: x.pow(y), x=ints(1, 2, 3, 4), y=randn(3, 4))
for name, op in [("maximum", torch.maximum), ("minimum", torch.minimum)]:
    case(f"grad.{name}", lambda x, y, op=op: op(x, y), x=randn(3, 1), y=randn(4))
    case(f"grad.{name}_scalar", lambda x, op=op: op(x, torch.tensor(0.5)), x=randn(3, 4))
case("grad.clamp", lambda x, lo, hi: x.clamp(lo, hi), x=randn(3, 4), lo=randn(4), hi=randn(3, 1))
case("grad.clamp_scalar", lambda x: x.clamp(-0.5, 0.5), x=randn(3, 4))
case("grad.clamp_min", lambda x, y: x.clamp_min(y), x=randn(3, 1), y=randn(4))
case("grad.clamp_max", lambda x, y: x.clamp_max(y), x=randn(3, 4), y=randn())
case("grad.clamp_min_scalar", lambda x: x.clamp_min(-0.5), x=randn(3, 4))
case("grad.clamp_max_scalar", lambda x: x.clamp_max(0.5), x=randn(3, 4))
case(
    "grad.clamp_mixed_min",
    lambda x, hi: x.clamp(torch.tensor(-0.5, dtype=x.dtype), hi),
    x=randn(3, 4),
    hi=randn(4, dtype=torch.float64),
)
case(
    "grad.clamp_mixed_max",
    lambda x, lo: x.clamp(lo, torch.tensor(0.5, dtype=lo.dtype)),
    x=randn(3, 4),
    lo=randn(3, 1, dtype=torch.float64),
)
case(
    "grad.masked_fill",
    lambda x, value, cond: x.masked_fill(cond, value),
    x=randn(3, 4),
    value=randn(dtype=torch.float64),
    cond=torch.rand(4) > 0.5,
)
case(
    "grad.masked_fill_scalar",
    lambda x, cond: x.masked_fill(cond, 0.5),
    x=randn(3, 4),
    cond=torch.rand(3, 1) > 0.5,
)
case(
    "grad.addcmul",
    lambda x, y, z: x.addcmul(y, z, value=1.7),
    x=randn(3, 4),
    y=randn(4),
    z=randn(3, 1),
)
case(
    "grad.addcdiv",
    lambda x, y, z: x.addcdiv(y, z, value=-0.3),
    x=randn(3, 4),
    y=randn(4),
    z=positive(3, 1),
)
case("grad.lerp", lambda x, y, z: x.lerp(y, z), x=randn(3, 4), y=randn(4), z=randn(3, 1))
for w, suffix in [(0.2, "low"), (0.8, "high")]:
    case(f"grad.lerp_{suffix}", lambda x, y, w=w: x.lerp(y, w), x=randn(3, 4), y=randn(4))

for dtype, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    case(
        f"grad_{tag}.pow",
        lambda x, y: x.pow(y),
        x=positive(3, 4, dtype=dtype),
        y=randn(4, dtype=dtype),
    )
    case(f"grad_{tag}.pow_scalar", lambda x: x.pow(2.5), x=positive(3, 4, dtype=dtype))
    case(
        f"grad_{tag}.maximum",
        lambda x, y: torch.maximum(x, y),
        x=randn(3, 1, dtype=dtype),
        y=randn(4, dtype=dtype),
    )
    case(
        f"grad_{tag}.minimum",
        lambda x, y: torch.minimum(x, y),
        x=randn(3, 1, dtype=dtype),
        y=randn(4, dtype=dtype),
    )
    case(
        f"grad_{tag}.clamp",
        lambda x, lo, hi: x.clamp(lo, hi),
        x=randn(3, 4, dtype=dtype),
        lo=randn(4, dtype=dtype),
        hi=randn(3, 1, dtype=dtype),
    )

# Full elementwise boundary grids prevent broadcasting reductions from hiding tie allocation
# mistakes. Every triple is tested with zero, negative, infinite and NaN upstream gradients.
values = [-inf, -1.0, -0.0, 0.0, 1.0, inf, nan]
triples = [(*t, g) for t in itertools.product(values, repeat=3) for g in [0.0, -2.0, inf, nan]]
x, lo, hi, g = (f(v) for v in zip(*triples))
case("special.clamp", lambda x, lo, hi: x.clamp(lo, hi), x=x, lo=lo, hi=hi, g=g)
pairs = [(*t, g) for t in itertools.product(values, repeat=2) for g in [0.0, -2.0, inf, nan]]
x, y, g = (f(v) for v in zip(*pairs))
for name, op in [
    ("maximum", torch.maximum),
    ("minimum", torch.minimum),
    ("clamp_min", torch.clamp_min),
    ("clamp_max", torch.clamp_max),
]:
    case(f"special.{name}", lambda x, y, op=op: op(x, y), x=x, y=y, g=g)
for name, op in [
    ("clamp_scalar", lambda x: x.clamp(-0.5, 0.5)),
    ("clamp_min_scalar", lambda x: x.clamp_min(-0.5)),
    ("clamp_max_scalar", lambda x: x.clamp_max(0.5)),
    ("clamp_equal", lambda x: x.clamp(0.5, 0.5)),
    ("clamp_reversed", lambda x: x.clamp(0.5, -0.5)),
    ("clamp_nan_min", lambda x: x.clamp(nan, 0.5)),
    ("clamp_nan_max", lambda x: x.clamp(-0.5, nan)),
]:
    case(
        f"special.{name}",
        op,
        x=f([-1., -0.5, 0., 0.5, 1., nan, inf, -inf]),
        g=f([inf, nan, -2., inf, nan, 3., 4., 5.]),
    )
case(
    "special.pow",
    lambda x, y: x.pow(y),
    x=f([0., 0., 0., 0., -2., nan, inf]),
    y=f([0., 1., 2., -1., 2., 0., 0.]),
    g=f([inf, 2., 3., 1., 1., nan, inf]),
)
case("special.pow_zero", lambda x: x.pow(0), x=f([0., -2., nan, inf]), g=f([inf, nan, inf, nan]))
case(
    "special.pow_scalar",
    lambda x: x.pow(2.5),
    x=f([0., -0., -2., nan, inf, -inf]),
    g=f([1., inf, 1., 1., 0., 1.]),
)
for name, op in [
    ("addcmul", lambda x, y, z: x.addcmul(y, z, value=1.7)),
    ("addcdiv", lambda x, y, z: x.addcdiv(y, z, value=-0.3)),
    ("lerp", lambda x, y, z: x.lerp(y, z)),
]:
    case(
        f"special.{name}",
        op,
        x=f([0., inf, nan, 1., -inf]),
        y=f([inf, 0., 1., nan, 2.]),
        z=f([0., 1., inf, -inf, nan]),
        g=f([1., inf, nan, -inf, 0.]),
    )
case(
    "special.masked_fill",
    lambda x, value, cond: x.masked_fill(cond, value),
    x=f([nan, inf, -inf, 1.]),
    value=f(nan),
    cond=torch.tensor([True, False, True, False]),
    g=f([inf, -inf, nan, 2.]),
)
# Mixed bounds use equivalent explicit tensor constants in PyTorch. The minimum scalar was
# rounded in the first forward step; the maximum scalar in the second step's result dtype.
case(
    "special.clamp_mixed_round",
    lambda x, hi: x.clamp(torch.tensor(1.0002, dtype=x.dtype), hi),
    x=f([0.5, 1., 2., nan], dtype=torch.float16),
    hi=f([3., 3., 3., 3.], dtype=torch.float64),
    g=f([inf, 4., -2., 5.], dtype=torch.float64),
)
case(
    "special.clamp_mixed_min",
    lambda x, hi: x.clamp(torch.tensor(-0.5, dtype=x.dtype), hi),
    x=f([-1., -0.5, 0., 1., nan, inf, -inf]),
    hi=f([1., 1., -0.5, -1., -1., inf, nan]),
    g=f([inf, 4., -2., nan, 5., -inf, 3.]),
)
case(
    "special.clamp_mixed_max",
    lambda x, lo: x.clamp(lo, torch.tensor(0.5, dtype=lo.dtype)),
    x=f([-1., -0.5, 0., 1., nan, inf, -inf]),
    lo=f([-0.5, -0.5, 0.5, 1., 1., inf, nan]),
    g=f([inf, 4., -2., nan, 5., -inf, 3.]),
)

# Review fix: scale derivative factors before multiplying the upstream gradient. Append after
# every earlier case so their inputs and gradients stay byte-for-byte unchanged.
case(
    "special.addcmul_zero_large",
    lambda x, y, z: x.addcmul(y, z, value=0.0),
    x=f([0.0]), y=f([1.0]), z=f([3e38]), g=f([10.0]),
)
case(
    "special.addcmul_scaled",
    lambda x, y, z: x.addcmul(y, z, value=1e-10),
    x=f([0.0]), y=f([1.0]), z=f([1e20]), g=f([1e20]),
)
case(
    "special.addcmul_underflow",
    lambda x, y, z: x.addcmul(y, z, value=1e20),
    x=f([0.0]), y=f([1.0]), z=f([1e-30]), g=f([1e-20]),
)
for dtype, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    case(
        f"grad_{tag}.addcmul_small",
        lambda x, y, z: x.addcmul(y, z, value=0.001),
        x=f([0.0, 0.5, -1.0, 2.0], dtype=dtype),
        y=f([1.0, 2.0, -3.0, 300.0], dtype=dtype),
        z=f([300.0, -300.0, 150.0, 1.0], dtype=dtype),
        g=f([300.0, 300.0, -400.0, 200.0], dtype=dtype),
    )
    case(
        f"grad_{tag}.addcdiv_small",
        lambda x, y, z: x.addcdiv(y, z, value=0.001),
        x=f([0.0, 0.5, -1.0, 2.0], dtype=dtype),
        y=f([1.0, 2.0, -3.0, 300.0], dtype=dtype),
        z=f([0.001, -0.001, 0.002, 300.0], dtype=dtype),
        g=f([1000.0, -1000.0, 300.0, 300.0], dtype=dtype),
    )
    for value, suffix in [(0.1, "tenth"), (0.3, "three_tenths"), (1.7, "factor")]:
        # Random half samples also distinguish C++ Scalar/Tensor coefficient rounding from
        # Python's reciprocal-times-scalar path, which uses different rounding points.
        case(
            f"grad_{tag}.addcdiv_{suffix}",
            lambda x, y, z, value=value: x.addcdiv(y, z, value=value),
            x=randn(3, 4, dtype=dtype),
            y=randn(4, dtype=dtype),
            z=positive(3, 1, dtype=dtype),
        )

# ---------------------------------------------------------------- step 6c
# Reductions, appended after every earlier case so that those keep their inputs.
def with_zeros(x, *positions):
    for p in positions:
        x[p] = 0.0
    return x


case("grad.sum_dim", lambda x: x.sum(1), x=randn(3, 4))
case("grad.sum_dims_keep", lambda x: x.sum((0, 2), keepdim=True), x=randn(2, 3, 4))
case("grad.mean", lambda x: x.mean(), x=randn(3, 4))
case("grad.mean_dim", lambda x: x.mean(-1), x=randn(3, 4))
case("grad.mean_dims_keep", lambda x: x.mean((0, 1), keepdim=True), x=randn(2, 3, 4))
case("grad.prod", lambda x: x.prod(), x=randn(3, 4))
case("grad.prod_dim", lambda x: x.prod(1), x=randn(3, 4))
case("grad.prod_dim_keep", lambda x: x.prod(0, keepdim=True), x=randn(3, 4))
# With zeros, the gradient is the product of the other elements in every slice.
case("grad.prod_zero", lambda x: x.prod(), x=with_zeros(randn(3, 4), (1, 2)))
case("grad.prod_dim_zeros", lambda x: x.prod(1), x=with_zeros(randn(3, 4), (0, 1), (0, 3), (2, 0)))
case("grad.max", lambda x: x.max(), x=randn(3, 4))
case("grad.min", lambda x: x.min(), x=randn(3, 4))
case("grad.max_dim", lambda x: x.max(1)[0], x=randn(3, 4))
case("grad.min_dim_keep", lambda x: x.min(0, keepdim=True)[0], x=randn(3, 4))
case("grad.var", lambda x: x.var(), x=randn(3, 4))
case("grad.var_dim", lambda x: x.var(1), x=randn(3, 4))
case("grad.var_dims_keep", lambda x: x.var((0, 1), keepdim=True, correction=0), x=randn(2, 3, 4))
case("grad.std", lambda x: x.std(), x=randn(3, 4))
case("grad.std_dim", lambda x: x.std(0, correction=2), x=randn(4, 3))

# Random half precision prod is left out: without zeros, the gradient `g * (result / x)` inherits
# the documented difference of the forward results (homura's product is computed in f64 and
# rounded once, PyTorch's is rounded at each step). The gradient with zeros doesn't use the
# result, and has exact half precision cases below (review round 18).
for dtype, tag in [(torch.float16, "f16"), (torch.bfloat16, "bf16")]:
    case(f"grad_{tag}.sum_dim", lambda x: x.sum(1), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.mean_dim", lambda x: x.mean(-1), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.max_dim", lambda x: x.max(1)[0], x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.var_dim", lambda x: x.var(1), x=randn(3, 4, dtype=dtype))
    case(f"grad_{tag}.std", lambda x: x.std(), x=randn(3, 4, dtype=dtype))

# Boundaries, compared exactly.
case("special.sum_dim", lambda x: x.sum(1), x=f([[1.0, inf], [nan, 2.0]]), g=f([nan, -inf]))
case("special.mean_dim", lambda x: x.mean(-1), x=f([[1.0, inf], [nan, 2.0]]), g=f([inf, -2.0]))
case("special.prod", lambda x: x.prod(), x=f([0.0, inf, 2.0]), g=f(1.0))
case("special.prod_two_zeros", lambda x: x.prod(), x=f([0.0, nan, 0.0, inf]), g=f(nan))
case("special.prod_overflow", lambda x: x.prod(), x=f([1e30, 1e30, 1e-30, 0.0]), g=f(1.0))
case(
    "special.prod_dim",
    lambda x: x.prod(1),
    x=f([[2.0, inf, 4.0], [0.0, 3.0, 5.0], [0.0, inf, 0.0], [nan, 1.0, 2.0], [-0.0, 3.0, 2.0]]),
    g=f([1.0, inf, inf, -2.0, 1.0]),
)
case("special.prod_dim_keep", lambda x: x.prod(0, keepdim=True), x=f([[2.0, inf], [nan, 3.0]]), g=f([[inf, 2.0]]))
case("special.max", lambda x: x.max(), x=f([nan, 1.0, nan, inf]), g=f(inf))
case("special.min", lambda x: x.min(), x=f([1.0, -inf, 3.0, -inf, -0.0]), g=f(-3.0))
case("special.max_ties", lambda x: x.max(), x=f([-0.0, 0.0, -1.0]), g=f(nan))
case(
    "special.max_dim",
    lambda x: x.max(1)[0],
    x=f([[1.0, 3.0, 3.0], [nan, 2.0, nan], [-inf, -inf, -inf]]),
    g=f([inf, nan, -2.0]),
)
case("special.min_dim_keep", lambda x: x.min(0, keepdim=True)[0], x=f([[1.0, 3.0], [1.0, nan]]), g=f([[2.0, inf]]))
case("special.var", lambda x: x.var(), x=f([1.0, inf, 3.0]), g=f(1.0))
case("special.var_c2", lambda x: x.var(correction=2), x=f([2.0, 2.0]), g=f(1.0))
case("special.var_c4", lambda x: x.var(correction=4), x=f([1.0, 2.0, 3.0]), g=f(-2.0))
case("special.var_dim_c3", lambda x: x.var(1, correction=3), x=f([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), g=f([1.0, -2.0]))
case("special.var_dim_c4", lambda x: x.var(1, correction=4), x=f([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]), g=f([1.0, inf]))
case("special.std", lambda x: x.std(), x=f([2.0, 2.0, 2.0]), g=f(inf))
case("special.std_c3", lambda x: x.std(correction=3), x=f([1.0, 2.0, 3.0]), g=f(1.0))
case("special.std_dim", lambda x: x.std(0, correction=2), x=f([[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]]), g=f([nan, inf]))
case("special.std_dim_c4", lambda x: x.std(1, correction=4), x=f([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]), g=f([1.0, 1.0]))

# Products of the other elements accumulate in f64, as PyTorch's cumprod does: grouped in f32,
# those before the last element of each row would overflow and underflow and give NaN, not 1.
case(
    "special.prod_dim_overflow",
    lambda x: x.prod(1),
    x=f([[1e30, 1e30, 1e-30, 1e-30, 0.0], [0.0, 1e-30, 1e-30, 1e30, 1e30]]),
    g=f([1.0, 1.0]),
)

# Review round 18 (R35, R36): the products of the other elements are accumulated in order, in
# PyTorch's accumulation type (f64 for f32 and f64, f32 for f16 and bf16), and rounded to the
# dtype one by one, so the intermediate products overflow and underflow where the reference's do.
def run(*pairs):
    return [v for v, n in pairs for _ in range(n)]


case(
    "special.prod_dim_order_f64",
    lambda x: x.prod(1),
    x=f([[1e308, 1e308, 1e-308, 1e-308, 0.0], [1e-308, 1e-308, 1e308, 1e308, 0.0], [1e308, 1e-308, 1e-308, 1e308, 0.0]], dtype=torch.float64),
    g=f([1.0, inf, -2.0], dtype=torch.float64),
)
case(
    "special.prod_dim_order_f32",
    lambda x: x.prod(1),
    x=f([run((1e30, 11), (1e-30, 11), (0.0, 1)), run((1e-30, 11), (1e30, 11), (0.0, 1))]),
    g=f([1.0, -2.0]),
)
case(
    "special.prod_dim_order_f16",
    lambda x: x.prod(1),
    x=f([run((1000.0, 14), (0.001, 14), (0.0, 1)), run((0.001, 14), (1000.0, 14), (0.0, 1))], dtype=torch.float16),
    g=f([1.0, inf], dtype=torch.float16),
)
case(
    "special.prod_dim_order_bf16",
    lambda x: x.prod(1),
    x=f([[1e30, 1e30, 1e-30, 1e-30, 0.0], [0.0, 1e-30, 1e-30, 1e30, 1e30]], dtype=torch.bfloat16),
    g=f([1.0, -2.0], dtype=torch.bfloat16),
)
case("special.prod_order_bf16", lambda x: x.prod(), x=f([1e30, 1e30, 1e-30, 0.0], dtype=torch.bfloat16), g=f(1.0, dtype=torch.bfloat16))

# ---------------------------------------------------------------- step 6d
# Views and indexing, appended after every earlier case so that those keep their inputs. Among
# several index items, PyTorch skips slices that cover their dimension; the gradients are the same
# either way, the node chains are checked in tests/autograd_views.rs.
case("grad.permute", lambda x: x.permute(2, 0, 1), x=randn(2, 3, 4))
case("grad.transpose", lambda x: x.transpose(-1, 0), x=randn(2, 3, 4))
case("grad.squeeze", lambda x: x.squeeze(), x=randn(2, 1, 3, 1))
case("grad.squeeze_dim", lambda x: x.squeeze(-1), x=randn(2, 1, 3, 1))
case("grad.squeeze_dims", lambda x: x.squeeze((1, 3)), x=randn(2, 1, 3, 1))
case("grad.squeeze_none", lambda x: x.squeeze(()), x=randn(2, 1, 3))
case("grad.unsqueeze", lambda x: x.unsqueeze(1), x=randn(2, 3))
case("grad.flatten", lambda x: x.flatten(1, 2), x=randn(2, 3, 4))
case("grad.index_alias", lambda x: x[:, :], x=randn(2, 3))
case("grad.index_mixed", lambda x: x[:, 1:, None, ::2], x=randn(2, 3, 4))
case("grad.index_full_tensor", lambda x, idx: x[:, idx], x=randn(3, 4), idx=ints(3, 0, 3, -1))
for tag, dtype in [("f16", torch.float16), ("bf16", torch.bfloat16)]:
    case(f"grad_{tag}.permute", lambda x: x.permute(2, 0, 1), x=randn(2, 3, 4, dtype=dtype))
    case(f"grad_{tag}.squeeze_dims", lambda x: x.squeeze((1, 3)), x=randn(2, 1, 3, 1, dtype=dtype))
    case(f"grad_{tag}.index_mixed", lambda x: x[:, 1:, None, ::2], x=randn(2, 3, 4, dtype=dtype))
# Views pass non-finite upstream gradients through unchanged.
case(
    "special.permute",
    lambda x: x.permute(2, 0, 1),
    x=f([[[1.0, 2.0], [3.0, 4.0]]]),
    g=f([[[inf, nan]], [[-inf, -0.0]]]),
)
case("special.unsqueeze", lambda x: x.unsqueeze(0), x=f([inf, nan]), g=f([[nan, -inf]]))
case(
    "special.index_mixed",
    lambda x: x[:, 1:, None, ::2],
    x=f([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
    g=f([[[[inf, -0.0]]]]),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
