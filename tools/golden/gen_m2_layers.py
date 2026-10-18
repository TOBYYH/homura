"""Generates tests/golden/m2_layers.safetensors: F.linear in each of its branches and a stack of
layers, with PyTorch's outputs and gradients. tests/nn_layers.rs reads the file.

Case "linear_<dtype>.<name>" stores the input "x", the weight "w", the bias "b" if there is
one, the upstream gradient "g", the output "out" and "grad_<input>" for each floating point
input, from torch.autograd.grad(out, inputs, g). An input stored as "x_t" is used transposed in
its last two dimensions, so that it isn't contiguous; a bias stored as "b_s" (shape [n, 2]) is
used as its column 0, a 1-D bias with stride 2. Integer cases have no gradients. The F64 cases
of SECOND also store "gg_<input>" and the second derivatives "grad2_<variable>": the gradients
of sum(grad_<input> * gg_<input>), with the first gradients computed with create_graph and "g"
as a variable too; a variable whose second derivative is None has no entry.

Case "layers.mlp_<dtype>" is a Sequential of Linear layers and every activation layer of step
3a. It stores the state dict as "param.<key>", the input "x", "g", "out", "grad_x" and
"grad.<key>" for each parameter. The weights are drawn once in f64; the generator checks that
the ReLU inputs keep a distance from 0, so the gradients don't depend on how ties at the kink
are broken.

Usage: python tools/golden/gen_m2_layers.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_layers.safetensors"

reference.check()
torch.manual_seed(0)
data = {}


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def randn(*shape):
    return torch.randn(shape, dtype=torch.float64)


# name: (input shape, weight shape, bias shape or None, input transposed in its last two dims,
# bias with stride 2)
LINEAR = {
    "2d_bias": ((5, 4), (3, 4), (3,), False, False),
    "2d_nobias": ((5, 4), (3, 4), None, False, False),
    "1d_bias": ((4,), (3, 4), (3,), False, False),
    "1d_nobias": ((4,), (3, 4), None, False, False),
    "3d_bias": ((2, 5, 4), (3, 4), (3,), False, False),
    "3d_nobias": ((2, 5, 4), (3, 4), None, False, False),
    "4d_bias": ((2, 3, 2, 4), (3, 4), (3,), False, False),
    "4d_bias_2d": ((2, 3, 2, 4), (3, 4), (2, 3), False, False),
    "2d_bias_2d": ((5, 4), (3, 4), (5, 3), False, False),
    "2d_t_bias": ((5, 4), (3, 4), (3,), True, False),
    "3d_t_bias": ((2, 5, 4), (3, 4), (3,), True, False),
    "3d_t_nobias": ((2, 5, 4), (3, 4), None, True, False),
    "weight_1d": ((5, 4), (4,), None, False, False),
    "weight_1d_bias_0d": ((4,), (4,), (), False, False),
    "empty_2d": ((0, 4), (3, 4), (3,), False, False),
    "empty_3d": ((2, 0, 4), (3, 4), (3,), False, False),
    "zero_features": ((2, 0), (3, 0), (3,), False, False),
    # Review round 47 (R56): only a contiguous bias with a single dimension that isn't 1 takes
    # the flattened addmm; other biases are added in place after the matmul.
    "3d_bias_2d": ((2, 5, 4), (3, 4), (5, 3), False, False),
    "3d_bias_3d": ((2, 5, 4), (3, 4), (2, 5, 3), False, False),
    "3d_bias_0d": ((2, 5, 4), (3, 4), (), False, False),
    "4d_bias_row": ((2, 3, 2, 4), (3, 4), (1, 3), False, False),
    "3d_bias_strided": ((2, 5, 4), (3, 4), (3,), False, True),
    "1d_bias_strided": ((4,), (3, 4), (3,), False, True),
}
HALF = ["2d_bias", "1d_nobias", "3d_bias", "3d_t_bias", "4d_bias_2d", "3d_bias_2d", "3d_bias_3d",
        "3d_bias_0d", "4d_bias_row", "3d_bias_strided", "1d_bias_strided"]
# One case per branch and bias layout for the second derivatives.
SECOND = ["2d_bias", "3d_bias", "3d_t_bias", "3d_bias_2d", "4d_bias_row", "3d_bias_strided",
          "2d_nobias"]
DTYPES = {"f32": torch.float32, "f64": torch.float64, "f16": torch.float16, "bf16": torch.bfloat16}


def linear_case(tag, name):
    x_shape, w_shape, b_shape, transposed, strided = LINEAR[name]
    dtype = DTYPES[tag]
    case = f"linear_{tag}.{name}"
    if transposed:
        x_stored = randn(*x_shape[:-2], x_shape[-1], x_shape[-2]).to(dtype)
        x = x_stored.transpose(-1, -2)
    else:
        x_stored = x = randn(*x_shape).to(dtype)
    w = randn(*w_shape).to(dtype)
    if strided:
        b_stored = randn(*b_shape, 2).to(dtype)
        b = b_stored[..., 0]
        assert not b.is_contiguous()
    else:
        b_stored = b = randn(*b_shape).to(dtype) if b_shape is not None else None
    inputs = {"x": x.detach().requires_grad_(), "w": w.detach().requires_grad_()}
    if b is not None:
        inputs["b"] = b.detach().requires_grad_()
    out = F.linear(inputs["x"], inputs["w"], inputs.get("b"))
    g = randn(*out.shape).to(dtype)
    second = tag == "f64" and name in SECOND
    if second:
        g.requires_grad_()
    grads = torch.autograd.grad(out, list(inputs.values()), g, create_graph=second)
    put(case, "x_t" if transposed else "x", x_stored)
    put(case, "w", w)
    if b is not None:
        put(case, "b_s" if strided else "b", b_stored)
    put(case, "g", g)
    put(case, "out", out)
    for k, gk in zip(inputs, grads):
        put(case, f"grad_{k}", gk)
    if second:
        gg = {k: randn(*t.shape).to(dtype) for k, t in inputs.items()}
        total = sum((gk * gg[k]).sum() for k, gk in zip(inputs, grads))
        variables = {**inputs, "g": g}
        second_grads = torch.autograd.grad(total, list(variables.values()), allow_unused=True)
        for k, t in gg.items():
            put(case, f"gg_{k}", t)
        for k, t in zip(variables, second_grads):
            if t is not None:
                put(case, f"grad2_{k}", t)


for tag in ["f32", "f64"]:
    for name in LINEAR:
        linear_case(tag, name)
for tag in ["f16", "bf16"]:
    for name in HALF:
        linear_case(tag, name)

# Integer inputs: forward only, in the addmm and the matmul + add_ branches.
for name, (x_shape, w_shape, b_shape) in {
    "2d_bias": ((5, 4), (3, 4), (3,)),
    "3d_t_bias": ((2, 4, 5), (3, 4), (3,)),
    "3d_nobias": ((2, 5, 4), (3, 4), None),
}.items():
    case = f"linear_i64.{name}"
    x_stored = torch.randint(-5, 6, x_shape)
    x = x_stored.transpose(-1, -2) if "_t_" in name else x_stored
    w = torch.randint(-5, 6, w_shape)
    b = torch.randint(-5, 6, b_shape) if b_shape is not None else None
    put(case, "x_t" if "_t_" in name else "x", x_stored)
    put(case, "w", w)
    if b is not None:
        put(case, "b", b)
    put(case, "out", F.linear(x, w, b))


def mlp():
    """Linear(4→6)–ReLU–Linear(6→5)–Tanh–Linear(5→5, no bias)–GELU(tanh)–Linear(5→4)–SiLU–
    Linear(4→4)–Sigmoid–GELU–Linear(4→3)–Softmax(-1), matching tests/nn_layers.rs."""
    return nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 5),
        nn.Tanh(),
        nn.Linear(5, 5, bias=False),
        nn.GELU(approximate="tanh"),
        nn.Linear(5, 4),
        nn.SiLU(),
        nn.Linear(4, 4),
        nn.Sigmoid(),
        nn.GELU(),
        nn.Linear(4, 3),
        nn.Softmax(-1),
    )


# Its own seed, so that adding cases above doesn't change the model.
torch.manual_seed(1)
model = mlp().double()
x = randn(7, 4)
with torch.no_grad():
    margin = model[0](x).abs().min().item()
assert margin > 0.02, f"a ReLU input is {margin} from the kink; pick another seed"
state = {k: v.detach().clone() for k, v in model.state_dict().items()}
for tag in ["f64", "f32"]:
    dtype = DTYPES[tag]
    case = f"layers.mlp_{tag}"
    m = mlp().to(dtype)
    m.load_state_dict({k: v.to(dtype) for k, v in state.items()})
    xi = x.to(dtype).requires_grad_()
    out = m(xi)
    g = randn(*out.shape).to(dtype)
    params = dict(m.named_parameters())
    grads = torch.autograd.grad(out, [xi, *params.values()], g)
    for k, v in m.state_dict().items():
        put(case, f"param.{k}", v)
    put(case, "x", xi)
    put(case, "g", g)
    put(case, "out", out)
    put(case, "grad_x", grads[0])
    for k, gk in zip(params, grads[1:]):
        put(case, f"grad.{k}", gk)

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
