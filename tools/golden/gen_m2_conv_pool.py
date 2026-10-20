"""Generates tests/golden/m2_conv_pool.safetensors: F.conv2d, F.max_pool2d and
F.adaptive_avg_pool2d (M2 step 8b), with PyTorch's outputs and gradients. tests/nn_conv.rs and
tests/nn_pool.rs read the file.

Case "<function>_<dtype>.<name>", function conv, maxpool or adaptive, stores the inputs, the
options as I64 tensors ("opt.<name>"), the upstream gradient "g", the output "out" and the
gradient "grad_<input>" of each floating point input, from torch.autograd.grad(out, inputs, g).
With "opt.transpose" (0-D, any value) the function gets x.transpose(-2, -1), a non-contiguous
view; "opt.transpose_w" does the same for a convolution weight.

conv stores "x", "w" and optionally "b"; "opt.stride", "opt.dilation" ([2]), "opt.groups" (0-D)
and either "opt.padding" ([2]), "opt.valid" or "opt.same" (0-D, any value). Integer cases store
only the inputs and "out". This CPU can't compute PyTorch's half precision convolution gradients
(DNNL rejects them, asserted below and in check_m2_conv_pool.py), so the F16 / BF16 gradients are
derived: PyTorch's gradients of the inputs and "g" converted to F32, rounded to the dtype. The
F16 / BF16 "out" is PyTorch's own.

maxpool stores "x", "opt.kernel", "opt.padding", "opt.dilation" ([2]), "opt.ceil" (0-D, 0 or 1)
and optionally "opt.stride" ([2]; the kernel size without it), and the outputs "out" and
"indices". adaptive stores "x" and "opt.output" ([2]).

Cases of SECOND (f64 only) also store "gg_<input>" for each floating point input and the second
derivatives "grad2_<variable>": the gradients of sum(grad_<input> * gg_<input>), with the first
gradients computed with create_graph and "g" as a variable too; a variable whose second
derivative is None (the convolution's bias) has no entry. Pooling gives its input zeros.

Usage: python tools/golden/gen_m2_conv_pool.py (needs torch and safetensors).
"""

import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_conv_pool.safetensors"

reference.check()
# "same" padding warns about even kernels.
warnings.simplefilter("ignore")
torch.manual_seed(0)
data = {}
DTYPES = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i64": torch.int64,
    "u8": torch.uint8,
}
INF, NAN = float("inf"), float("nan")


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def randn(*shape):
    return torch.randn(shape, dtype=torch.float64)


def distinct(*shape):
    """A random permutation of distinct values (no ties for max pooling), scaled to about 1."""
    n = 1
    for s in shape:
        n *= s
    return (torch.randperm(n).to(torch.float64) / n * 4 - 2).reshape(shape)


CASES = {}


def case(function, name, inputs, dtypes, second=False, g=None, **options):
    CASES[f"{function}.{name}"] = (function, inputs, dtypes, second, g, options)


FLOAT = ["f64", "f32"]
ALL_FLOAT = ["f64", "f32", "f16", "bf16"]

# F.conv2d
case("conv", "basic", {"x": randn(2, 3, 6, 6), "w": randn(4, 3, 3, 3), "b": randn(4)}, ALL_FLOAT + ["i64"],
     second=True)
case("conv", "no_bias", {"x": randn(2, 3, 5, 5), "w": randn(2, 3, 3, 3)}, FLOAT)
case("conv", "stride_padding", {"x": randn(1, 2, 7, 6), "w": randn(3, 2, 3, 2), "b": randn(3)}, ALL_FLOAT,
     second=True, stride=[2, 1], padding=[1, 0])
case("conv", "padding2", {"x": randn(1, 2, 4, 4), "w": randn(2, 2, 3, 3), "b": randn(2)}, FLOAT, padding=[2, 2])
case("conv", "dilation", {"x": randn(2, 2, 7, 7), "w": randn(3, 2, 3, 2), "b": randn(3)}, FLOAT, second=True,
     dilation=[2, 3], padding=[1, 1])
case("conv", "groups", {"x": randn(2, 4, 5, 5), "w": randn(6, 2, 3, 3), "b": randn(6)}, ALL_FLOAT + ["i64"],
     second=True, groups=2)
case("conv", "depthwise", {"x": randn(1, 3, 5, 5), "w": randn(3, 1, 3, 3)}, FLOAT, groups=3, padding=[1, 1])
case("conv", "valid", {"x": randn(1, 2, 5, 6), "w": randn(2, 2, 2, 3), "b": randn(2)}, FLOAT, valid=1)
case("conv", "same_odd", {"x": randn(1, 2, 5, 5), "w": randn(3, 2, 3, 3), "b": randn(3)}, FLOAT, same=1)
case("conv", "same_even", {"x": randn(1, 2, 5, 6), "w": randn(2, 2, 2, 4), "b": randn(2)}, ALL_FLOAT,
     second=True, same=1)
case("conv", "same_dilation", {"x": randn(1, 1, 6, 6), "w": randn(2, 1, 2, 3)}, FLOAT, same=1,
     dilation=[3, 2])
case("conv", "unbatched", {"x": randn(3, 5, 5), "w": randn(2, 3, 3, 3), "b": randn(2)}, FLOAT, second=True,
     padding=[1, 1])
case("conv", "transposed", {"x": randn(2, 3, 6, 5), "w": randn(4, 3, 2, 3), "b": randn(4)}, ALL_FLOAT,
     transpose=1, transpose_w=1, stride=[1, 2])
case("conv", "one_by_one", {"x": randn(2, 3, 5, 5), "w": randn(4, 3, 1, 1), "b": randn(4)}, FLOAT,
     stride=[2, 2])
case("conv", "whole_input", {"x": randn(2, 2, 3, 3), "w": randn(3, 2, 3, 3), "b": randn(3)}, FLOAT)
case("conv", "empty_batch", {"x": randn(0, 3, 5, 5), "w": randn(4, 3, 3, 3), "b": randn(4)}, FLOAT)
# Products that cancel: the result is small next to the terms (for the F32 tolerance).
# The second pair of input channels is the first one negated, with the same weights, plus a
# little: the large terms cancel.
cancel_x = randn(1, 2, 4, 4) * 1e3
cancel_w = randn(2, 2, 3, 3)
case("conv", "cancellation", {"x": torch.cat([cancel_x, -cancel_x + randn(1, 2, 4, 4) * 1e-2], 1),
                              "w": torch.cat([cancel_w, cancel_w], 1), "b": randn(2)}, FLOAT)

# F.max_pool2d
case("maxpool", "basic", {"x": distinct(2, 3, 6, 6)}, ALL_FLOAT + ["i64", "u8"], second=True, kernel=[2, 2])
case("maxpool", "stride_padding", {"x": distinct(1, 2, 7, 7)}, ALL_FLOAT, second=True, kernel=[3, 3],
     stride=[2, 2], padding=[1, 1])
case("maxpool", "ceil", {"x": distinct(1, 2, 5, 7)}, FLOAT, kernel=[2, 3], stride=[2, 2], ceil=1)
case("maxpool", "ceil_padding", {"x": distinct(1, 1, 6, 6)}, FLOAT, kernel=[3, 3], stride=[2, 2],
     padding=[1, 1], ceil=1)
case("maxpool", "dilation", {"x": distinct(1, 2, 7, 8)}, FLOAT, second=True, kernel=[2, 3], stride=[1, 2],
     padding=[1, 1], dilation=[2, 2])
case("maxpool", "overlap", {"x": distinct(2, 2, 5, 5)}, ALL_FLOAT, second=True, kernel=[3, 3], stride=[1, 1])
case("maxpool", "rect", {"x": distinct(1, 3, 6, 8)}, FLOAT, kernel=[2, 3], stride=[1, 2], padding=[1, 1])
case("maxpool", "unbatched", {"x": distinct(3, 6, 6)}, FLOAT, kernel=[2, 2])
case("maxpool", "transposed", {"x": distinct(2, 2, 6, 5)}, ALL_FLOAT, transpose=1, kernel=[3, 2],
     stride=[2, 1], padding=[1, 1])
# Review round 65, R81: windows that select one position although the stride equals the kernel.
case("maxpool", "dilation_repeats", {"x": torch.tensor([[[[0.0, 0.0, 10.0, 0.0, 0.0]]]], dtype=torch.float64)},
     ALL_FLOAT, kernel=[1, 2], stride=[1, 2], dilation=[1, 2])
# Overlapping windows select one position repeatedly; the gradients add in output order, which
# matters in half precision (2048 + 1 + 1 vs 1 + 1 + 2048).
case("maxpool", "repeats_order", {"x": torch.tensor([[[[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 9.0]]]], dtype=torch.float64)},
     ALL_FLOAT, g=lambda dt: torch.tensor([[[[2048.0, 1.0, 1.0, 1.0, 1.0]]]], dtype=dt), kernel=[1, 3],
     stride=[1, 1])
case("maxpool", "nonfinite", {"x": torch.tensor([[[[NAN, 1.0, 2.0, -INF], [NAN, 3.0, INF, -INF],
                                                   [0.5, -1.0, -INF, -INF]]]], dtype=torch.float64)},
     ALL_FLOAT, g=lambda dt: torch.tensor([[[[INF, NAN], [-1.0, 2.0]]]], dtype=dt), kernel=[2, 2],
     stride=[1, 2])

# F.adaptive_avg_pool2d
case("adaptive", "shrink", {"x": randn(2, 3, 5, 7)}, ALL_FLOAT, second=True, output=[3, 4])
case("adaptive", "grow", {"x": randn(1, 2, 2, 3)}, ALL_FLOAT, second=True, output=[3, 5])
case("adaptive", "row", {"x": randn(2, 2, 4, 6)}, FLOAT, output=[1, 2])
case("adaptive", "column", {"x": randn(2, 2, 6, 4)}, FLOAT, output=[3, 1])
case("adaptive", "same_size", {"x": randn(1, 2, 3, 4)}, FLOAT, output=[3, 4])
case("adaptive", "unbatched", {"x": randn(3, 7, 5)}, FLOAT, output=[2, 3])
case("adaptive", "transposed", {"x": randn(2, 2, 6, 5)}, ALL_FLOAT, second=True, transpose=1, output=[4, 3])
case("adaptive", "large", {"x": randn(1, 2, 11, 13) * 10}, ALL_FLOAT, output=[4, 6])
# Review round 64, R79: the sum of the window cancels exactly.
case("adaptive", "cancellation", {"x": torch.tensor([[[[1e8, -1e8], [1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float64)},
     FLOAT, output=[2, 1])
case("adaptive", "mean", {"x": randn(2, 3, 4, 5)}, ALL_FLOAT, second=True, output=[1, 1])
case("adaptive", "empty_batch", {"x": randn(0, 2, 4, 4)}, FLOAT, output=[2, 3])

SPATIAL = ["kernel", "stride", "padding", "dilation", "output"]


def compute(function, args, options):
    x = args["x"]
    if "transpose" in options:
        x = x.transpose(-2, -1)
    if function == "conv":
        w = args["w"]
        if "transpose_w" in options:
            w = w.transpose(-2, -1)
        padding = "valid" if "valid" in options else "same" if "same" in options else options.get("padding", [0, 0])
        return F.conv2d(x, w, args.get("b"), stride=options.get("stride", [1, 1]), padding=padding,
                        dilation=options.get("dilation", [1, 1]), groups=options.get("groups", 1))
    if function == "maxpool":
        return F.max_pool2d(x, options["kernel"], stride=options.get("stride", []),
                            padding=options.get("padding", [0, 0]), dilation=options.get("dilation", [1, 1]),
                            ceil_mode=bool(options.get("ceil", 0)), return_indices=True)
    assert function == "adaptive"
    return F.adaptive_avg_pool2d(x, options["output"])


def store_options(name, function, options):
    defaults = {"conv": {"stride": [1, 1], "dilation": [1, 1], "groups": 1},
                "maxpool": {"padding": [0, 0], "dilation": [1, 1], "ceil": 0}}.get(function, {})
    if function == "conv" and not ({"valid", "same"} & options.keys()):
        defaults["padding"] = [0, 0]
    for k, v in {**defaults, **options}.items():
        put(name, f"opt.{k}", torch.tensor(v, dtype=torch.int64))


def generate(key, dtype_name):
    function, inputs, _, second_order, fixed_g, options = CASES[key]
    dtype = DTYPES[dtype_name]
    name = f"{function}_{dtype_name}.{key.split('.', 1)[1]}"
    floating = dtype.is_floating_point
    second = second_order and dtype_name == "f64"
    half = dtype_name in ("f16", "bf16")
    if function == "conv" and not floating:
        # Small integers, so that the products don't wrap.
        inputs = {k: (v * 3).round() for k, v in inputs.items()}
    args = {k: v.to(dtype).requires_grad_(floating) for k, v in inputs.items()}
    for k, v in args.items():
        put(name, k, v)
    store_options(name, function, options)
    result = compute(function, args, options)
    out, indices = result if function == "maxpool" else (result, None)
    put(name, "out", out)
    if indices is not None:
        put(name, "indices", indices)
    if not floating:
        return
    g = fixed_g(dtype) if fixed_g else randn(*out.shape).to(dtype)
    g = g.requires_grad_(second)
    put(name, "g", g)
    names = list(args)
    variables = [args[k] for k in names]
    if half and function == "conv":
        try:
            torch.autograd.grad(out, variables, g)
            raise AssertionError("PyTorch computed a half precision convolution gradient")
        except RuntimeError as e:
            assert "DNNL does not support bf16/f16 backward" in str(e), e
        up = {k: v.detach().float().requires_grad_() for k, v in args.items()}
        up_out = compute(function, up, options)
        first = [t.to(dtype) for t in torch.autograd.grad(up_out, [up[k] for k in names], g.float())]
    else:
        first = torch.autograd.grad(out, variables, g, create_graph=second)
    for k, t in zip(names, first):
        put(name, f"grad_{k}", t)
    if second:
        gg = [randn(*t.shape) for t in first]
        total = sum((t * s).sum() for t, s in zip(first, gg))
        for k, s in zip(names, gg):
            put(name, f"gg_{k}", s)
        grads = torch.autograd.grad(total, variables + [g], allow_unused=True)
        for k, t in zip(names + ["g"], grads):
            if t is not None:
                put(name, f"grad2_{k}", t)


for key, (_, _, dtypes, _, _, _) in CASES.items():
    for dtype_name in dtypes:
        generate(key, dtype_name)

OUT.parent.mkdir(exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT}")
