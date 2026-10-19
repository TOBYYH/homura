"""Generates tests/golden/m2_embed_norm.safetensors: F.embedding, F.layer_norm, F.rms_norm and
the deterministic parts of F.dropout (M2 step 5), with PyTorch's outputs and gradients.
tests/nn_embed_norm.rs and tests/nn_dropout.rs read the file.

Case "<function>_<dtype>.<name>", function one of emb, ln, rms, stores the inputs ("weight" and
the integer "indices" for emb; "x", and "weight" / "bias" when given, for ln and rms), the
options as 0-D tensors ("opt.padding_idx" when given; "opt.normalized_ndim" and "opt.eps",
which rms stores only when given), the upstream gradient "g", the output "out", and
"grad_<input>" for each floating point input, from torch.autograd.grad(out, inputs, g). The
inputs keep their stored dtype: a half precision ln case with "F32" parameters is the mixed
dtype path. Cases of SECOND (f64 only) also store "gg_<input>" and the second derivatives
"grad2_<variable>": the gradients of sum(grad_<input> * gg_<input>), with the first gradients
computed with create_graph and "g" as a variable too; a variable whose second derivative is
None has no entry.

Case "dropout_<dtype>.scale" stores the probabilities "p" (f64) and the value "noise" (the
dtype) that F.dropout gives the kept elements for each, 1 / (1 - p) in the dtype. Case
"dropout_<dtype>.values" stores "x", "g", "p" and the products "x_noise" = x * noise and
"g_noise" = g * noise with PyTorch's kernel: a kept element of F.dropout(x, p) is x_noise and its
gradient g_noise. Case "dropout_<dtype>.p1" stores "x", "g", "out" = F.dropout(x, 1) and
"grad_x".

Usage: python tools/golden/gen_m2_embed_norm.py (needs torch and safetensors).
"""

from pathlib import Path

import warnings

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_embed_norm.safetensors"

reference.check()
# rms_norm warns about a weight of another dtype.
warnings.simplefilter("ignore")
torch.manual_seed(0)
data = {}
DTYPES = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
INF, NAN = float("inf"), float("nan")


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def randn(*shape):
    return torch.randn(shape, dtype=torch.float64)


# Each case: function, inputs {name: f64 tensor or integer tensor}, options. Floating point
# inputs are converted to the case dtype, except those named in "keep", which keep their dtype.
CASES = {}


def case(function, name, inputs, keep=(), **options):
    CASES[f"{function}.{name}"] = (function, inputs, keep, options)


# F.embedding
case("emb", "1d", {"weight": randn(6, 3), "indices": torch.tensor([4, 0, 5, 2])})
case("emb", "2d_repeated", {"weight": randn(5, 4), "indices": torch.tensor([[1, 3, 1], [0, 1, 4]])})
case("emb", "0d", {"weight": randn(5, 2), "indices": torch.tensor(3)})
case("emb", "3d", {"weight": randn(4, 2), "indices": torch.randint(4, (2, 3, 2))})
case("emb", "padding", {"weight": randn(5, 3), "indices": torch.tensor([[2, 0, 2], [4, 2, 1]])},
     padding_idx=2)
case("emb", "padding_negative", {"weight": randn(5, 3), "indices": torch.tensor([4, 0, 4, 3])},
     padding_idx=-1)
case("emb", "padding_zero", {"weight": randn(4, 2), "indices": torch.tensor([0, 1, 0])},
     padding_idx=0)
case("emb", "int32", {"weight": randn(5, 3), "indices": torch.tensor([[3, 1], [1, 1]], dtype=torch.int32)},
     padding_idx=1)
case("emb", "empty", {"weight": randn(5, 3), "indices": torch.zeros(2, 0, dtype=torch.long)})
case("emb", "zero_dim_rows", {"weight": randn(3, 0), "indices": torch.tensor([2, 0])})

# F.layer_norm
case("ln", "2d", {"x": randn(3, 5), "weight": randn(5), "bias": randn(5)}, normalized_ndim=1)
case("ln", "3d", {"x": randn(2, 3, 4), "weight": randn(4), "bias": randn(4)}, normalized_ndim=1)
case("ln", "3d_two_dims", {"x": randn(2, 3, 4), "weight": randn(3, 4), "bias": randn(3, 4)},
     normalized_ndim=2)
case("ln", "plain", {"x": randn(4, 6)}, normalized_ndim=1)
case("ln", "weight_only", {"x": randn(3, 4), "weight": randn(4)}, normalized_ndim=1)
case("ln", "bias_only", {"x": randn(3, 4), "bias": randn(4)}, normalized_ndim=1)
case("ln", "eps", {"x": randn(3, 4) * 0.01, "weight": randn(4), "bias": randn(4)},
     normalized_ndim=1, eps=1e-3)
case("ln", "whole_input", {"x": randn(3, 4), "weight": randn(3, 4), "bias": randn(3, 4)},
     normalized_ndim=2)
case("ln", "1d", {"x": randn(5), "weight": randn(5), "bias": randn(5)}, normalized_ndim=1)
case("ln", "constant_rows", {"x": torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]),
                             "weight": randn(3), "bias": randn(3)}, normalized_ndim=1)
case("ln", "offset", {"x": randn(2, 8) + 1000.0, "weight": randn(8), "bias": randn(8)},
     normalized_ndim=1)
case("ln", "empty_batch", {"x": randn(0, 4), "weight": randn(4), "bias": randn(4)},
     normalized_ndim=1)
case("ln", "nonfinite", {"x": torch.tensor([[INF, 1.0, 2.0, 3.0], [NAN, 0.0, 1.0, 2.0],
                                            [0.5, 1.0, 2.0, 3.0]]),
                         "weight": randn(4), "bias": randn(4)}, normalized_ndim=1)
case("ln", "mixed", {"x": randn(3, 6), "weight": randn(6).float(), "bias": randn(6).float()},
     keep=("weight", "bias"), normalized_ndim=1)
case("ln", "mixed_weight_only", {"x": randn(2, 3, 4), "weight": randn(3, 4).float()},
     keep=("weight",), normalized_ndim=2)

# F.rms_norm
case("rms", "2d", {"x": randn(3, 5), "weight": randn(5)}, normalized_ndim=1, eps=1e-6)
case("rms", "3d_two_dims", {"x": randn(2, 3, 4), "weight": randn(3, 4)}, normalized_ndim=2,
     eps=1e-5)
case("rms", "default_eps", {"x": randn(4, 3), "weight": randn(3)}, normalized_ndim=1)
case("rms", "plain", {"x": randn(2, 6)}, normalized_ndim=1)
case("rms", "small", {"x": torch.tensor([[1e-4, -1e-4], [0.0, 0.0], [3.0, 4.0]])},
     normalized_ndim=1)
case("rms", "weight_f64", {"x": randn(3, 4), "weight": randn(4)}, keep=("weight",),
     normalized_ndim=1)
case("rms", "empty_batch", {"x": randn(0, 4), "weight": randn(4)}, normalized_ndim=1, eps=1e-6)

FUNCTIONS = ["emb", "ln", "rms"]
HALF = ["emb.2d_repeated", "emb.padding", "ln.2d", "ln.3d_two_dims", "ln.weight_only",
        "ln.mixed", "ln.mixed_weight_only", "rms.2d", "rms.default_eps", "rms.plain"]
# Not ln.bias_only: PyTorch's layer_norm_double_backward ignores the bias without a weight (see
# check_m2_embed_norm.py).
SECOND = ["emb.2d_repeated", "emb.padding", "ln.2d", "ln.3d_two_dims", "ln.plain",
          "ln.weight_only", "ln.whole_input", "rms.2d", "rms.3d_two_dims",
          "rms.plain"]
# The mixed cases exist only in half precision, the f64 weight only for an f32 input.
ONLY = {"ln.mixed": ["f16", "bf16"], "ln.mixed_weight_only": ["f16", "bf16"],
        "rms.weight_f64": ["f32"]}


def compute(function, args, options):
    if function == "emb":
        return F.embedding(args["indices"], args["weight"], padding_idx=options.get("padding_idx"))
    shape = list(args["x"].shape[args["x"].dim() - options["normalized_ndim"]:])
    if function == "ln":
        return F.layer_norm(args["x"], shape, args.get("weight"), args.get("bias"),
                            options.get("eps", 1e-5))
    assert function == "rms"
    return F.rms_norm(args["x"], shape, args.get("weight"), options.get("eps"))


def generate(key, dtype_name):
    function, inputs, keep, options = CASES[key]
    dtype = DTYPES[dtype_name]
    name = f"{function}_{dtype_name}.{key.split('.', 1)[1]}"
    args = {}
    for k, v in inputs.items():
        floating = v.is_floating_point()
        v = v.to(dtype) if floating and k not in keep else v.clone()
        v.requires_grad_(floating)
        args[k] = v
        put(name, k, v)
    for k, v in options.items():
        put(name, f"opt.{k}", torch.tensor(v, dtype=torch.float64 if k == "eps" else torch.int64))
    if function == "ln" and "eps" not in options:
        put(name, "opt.eps", torch.tensor(1e-5, dtype=torch.float64))
    out = compute(function, args, options)
    put(name, "out", out)
    second = dtype_name == "f64" and key in SECOND
    g = randn(*out.shape).to(out.dtype).requires_grad_(second)
    if key == "emb.padding":
        # The padding positions get no gradient, even an infinite or NaN one.
        rows = g.detach().view(-1, 3)
        rows[0] = torch.tensor([NAN, INF, 1.0])
        rows[4] = torch.tensor([-INF, 2.0, NAN])
        rows[3, 1] = INF
    if key == "ln.nonfinite":
        g.detach()[2, 0] = INF
    put(name, "g", g)
    names = [k for k, v in args.items() if v.requires_grad]
    variables = [args[k] for k in names]
    first = torch.autograd.grad(out, variables, g, create_graph=second)
    for k, t in zip(names, first):
        put(name, f"grad_{k}", t)
    if second:
        gg = [randn(*t.shape) for t in first]
        total = sum((t * s).sum() for t, s in zip(first, gg))
        for k, s in zip(names, gg):
            put(name, f"gg_{k}", s)
        second_grads = torch.autograd.grad(total, variables + [g], allow_unused=True)
        for k, t in zip(names + ["g"], second_grads):
            if t is not None:
                put(name, f"grad2_{k}", t)


for key in CASES:
    names = ONLY.get(key, ["f64", "f32"] + (["f16", "bf16"] if key in HALF else []))
    for dtype_name in names:
        generate(key, dtype_name)

# F.dropout
PROBABILITIES = [1e-7, 0.1, 0.2, 0.25, 0.3, 1 / 3, 0.5, 0.7, 0.9, 0.999]
for dtype_name, dtype in DTYPES.items():
    name = f"dropout_{dtype_name}"
    noise = []
    for p in PROBABILITIES:
        expected = torch.ones(1, dtype=dtype).div_(1 - p)
        out = F.dropout(torch.ones(20000, dtype=dtype), p)
        kept = out[out != 0].unique()
        assert torch.equal(kept, expected), (dtype, p, kept, expected)
        noise.append(expected)
    put(f"{name}.scale", "p", torch.tensor(PROBABILITIES, dtype=torch.float64))
    put(f"{name}.scale", "noise", torch.cat(noise))
    x = (randn(4, 25) * 3).to(dtype)
    g = randn(4, 25).to(dtype)
    p = 0.3
    full = torch.full_like(x, torch.ones(1, dtype=dtype).div_(1 - p).item())
    for k, v in [("x", x), ("g", g), ("p", torch.tensor(p, dtype=torch.float64)),
                 ("x_noise", x.mul(full)), ("g_noise", g.mul(full))]:
        put(f"{name}.values", k, v)
    x = torch.tensor([1.5, -2.0, INF, -INF, NAN, 0.0, -0.0], dtype=dtype, requires_grad=True)
    g = torch.tensor([1.0, INF, 2.0, NAN, -1.0, 3.0, 1.0], dtype=dtype)
    out = F.dropout(x, 1.0)
    (grad,) = torch.autograd.grad(out, x, g)
    for k, v in [("x", x), ("g", g), ("out", out), ("grad_x", grad)]:
        put(f"{name}.p1", k, v)

OUT.parent.mkdir(exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT}")
