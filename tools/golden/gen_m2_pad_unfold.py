"""Generates tests/golden/m2_pad_unfold.safetensors: F.pad in constant mode and Tensor.unfold
(M2 step 8a), with PyTorch's outputs and gradients. tests/nn_pad_unfold.rs reads the file.

Case "<function>_<dtype>.<name>", function pad or unfold, stores the input "x", the options as
I64 / F64 tensors, the upstream gradient "g", the output "out" and the gradient "grad_x" from
torch.autograd.grad(out, x, g). pad stores "opt.pad" (1-D, possibly empty) and "opt.value"
(0-D F64); unfold stores "opt.unfold", one row (dim, size, step) per unfold applied in order.
With "opt.transpose" (0-D, any value) the function is applied to x.transpose(-2, -1), a
non-contiguous view of x. Cases of SECOND (f64 only) also
store "gg_x" and the second derivative "grad2_g", the gradient of sum(grad_x * gg_x) with
respect to g; both functions are linear, so x has no second derivative.

Case unfold_<dtype>.order adds windows whose sum depends on the order of the additions in the
dtype: position 2 gets big + 1 + 1, position 4 gets 1 + 1 + big, where big is the power of two
whose spacing is 2 in the dtype.

Usage: python tools/golden/gen_m2_pad_unfold.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_pad_unfold.safetensors"

reference.check()
torch.manual_seed(0)
data = {}
DTYPES = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
INF, NAN = float("inf"), float("nan")
# The power of two at which the spacing of the dtype is 2.
BIG = {"f64": 2.0**53, "f32": 2.0**24, "f16": 2048.0, "bf16": 256.0}


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def randn(*shape):
    return torch.randn(shape, dtype=torch.float64)


# Each case: x (f64), options, and optionally a fixed upstream gradient (f64, or a function of
# the dtype name).
CASES = {}


def case(function, name, x, g=None, transpose=False, **options):
    CASES[f"{function}.{name}"] = (function, x, g, transpose, options)


case("pad", "1d", randn(5), pad=[2, 1], value=0.0)
case("pad", "2d", randn(3, 4), pad=[1, 2, 0, 1], value=0.5)
case("pad", "crop", randn(3, 5), pad=[-1, -2], value=9.0)
case("pad", "mixed", randn(2, 3, 4), pad=[-1, 2, 1, -1], value=-1.5)
case("pad", "three_dims", randn(2, 2, 3), pad=[1, 1, 0, 0, 2, 0], value=3.0)
case("pad", "zero", randn(2, 3), pad=[0, 0], value=1.0)
case("pad", "crop_to_empty", randn(2, 3), pad=[-2, -1], value=0.0)
case("pad", "empty_input", randn(0, 3), pad=[1, 1, 1, 1], value=2.0)
case("pad", "transposed", randn(3, 4), transpose=True, pad=[1, -1, 0, 2], value=0.25)
case("pad", "scalar", randn(), pad=[], value=0.0)
case("pad", "nonfinite", torch.tensor([[1.0, NAN, -INF], [INF, 2.0, -0.0]]),
     g=torch.tensor([[INF, 1.0, NAN, 2.0], [3.0, -0.0, 4.0, NAN], [1.0, 1.0, 1.0, 1.0]]),
     pad=[1, 0, 0, 1], value=-INF)

case("unfold", "overlap", randn(2, 5), unfold=[(1, 3, 1)])
case("unfold", "stride", randn(2, 7), unfold=[(1, 3, 2)])
case("unfold", "gap", randn(2, 7), unfold=[(1, 2, 3)])
case("unfold", "exact", randn(3, 6), unfold=[(1, 2, 2)])
case("unfold", "dim0", randn(5, 3), unfold=[(0, 2, 1)])
case("unfold", "negative_dim", randn(2, 3, 6), unfold=[(-1, 4, 1)])
case("unfold", "two_dims", randn(4, 5), unfold=[(0, 2, 1), (1, 3, 1)])
case("unfold", "two_dims_strided", randn(5, 6), unfold=[(0, 3, 2), (1, 2, 1)])
case("unfold", "windows_of_windows", randn(10), unfold=[(0, 4, 2), (1, 2, 1)])
case("unfold", "transposed", randn(6, 4), transpose=True, unfold=[(1, 3, 1)])
case("unfold", "size0", randn(2, 3), unfold=[(1, 0, 1)])
case("unfold", "whole", randn(2, 4), unfold=[(1, 4, 1)])
case("unfold", "scalar", randn(), unfold=[(0, 1, 1)])
case("unfold", "nonfinite", torch.tensor([[1.0, NAN, 3.0, -INF, 5.0]]),
     g=torch.tensor([[[INF, 1.0, 2.0], [3.0, NAN, 4.0], [-0.0, 5.0, -INF]]]),
     unfold=[(1, 3, 1)])


def order_gradient(dtype_name):
    g = torch.zeros(5, 3, dtype=torch.float64)
    big = BIG[dtype_name]
    g[0, 2], g[1, 1], g[2, 0] = big, 1, 1
    g[2, 2], g[3, 1], g[4, 0] = 1, 1, big
    return g


case("unfold", "order", torch.zeros(7), g=order_gradient, unfold=[(0, 3, 1)])

HALF = ["pad.2d", "pad.mixed", "pad.transposed", "pad.nonfinite", "unfold.overlap",
        "unfold.two_dims", "unfold.windows_of_windows", "unfold.transposed", "unfold.order"]
SECOND = ["pad.2d", "pad.mixed", "pad.transposed", "pad.crop", "unfold.overlap",
          "unfold.two_dims", "unfold.windows_of_windows", "unfold.transposed", "unfold.gap"]


def compute(function, x, options):
    if function == "pad":
        return F.pad(x, options["pad"], value=options["value"])
    for dim, size, step in options["unfold"]:
        x = x.unfold(dim, size, step)
    return x


def generate(key, dtype_name):
    function, x, g, transpose, options = CASES[key]
    dtype = DTYPES[dtype_name]
    name = f"{function}_{dtype_name}.{key.split('.', 1)[1]}"
    x = x.to(dtype).requires_grad_()
    put(name, "x", x)
    if function == "pad":
        put(name, "opt.pad", torch.tensor(options["pad"], dtype=torch.int64))
        put(name, "opt.value", torch.tensor(options["value"], dtype=torch.float64))
    else:
        put(name, "opt.unfold", torch.tensor(options["unfold"], dtype=torch.int64))
    if transpose:
        put(name, "opt.transpose", torch.tensor(1, dtype=torch.int64))
    out = compute(function, x.transpose(-2, -1) if transpose else x, options)
    put(name, "out", out)
    second = dtype_name == "f64" and key in SECOND
    if callable(g):
        g = g(dtype_name)
    g = (randn(*out.shape) if g is None else g).to(dtype).requires_grad_(second)
    assert g.shape == out.shape, key
    put(name, "g", g)
    (grad,) = torch.autograd.grad(out, x, g, create_graph=second)
    put(name, "grad_x", grad)
    if second:
        gg = randn(*grad.shape)
        put(name, "gg_x", gg)
        grad2 = torch.autograd.grad((grad * gg).sum(), [x, g], allow_unused=True)
        assert grad2[0] is None, key
        put(name, "grad2_g", grad2[1])


for key in CASES:
    names = ["f64", "f32"] + (["f16", "bf16"] if key in HALF else [])
    for dtype_name in names:
        generate(key, dtype_name)

OUT.parent.mkdir(exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT}")
