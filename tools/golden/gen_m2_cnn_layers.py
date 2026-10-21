"""Generates tests/golden/m2_cnn_layers.safetensors: nn.BatchNorm2d layers called in sequences of
training and evaluation passes, and the CNN of M2 step 9 as an nn.Sequential, with PyTorch's
outputs, gradients and states. tests/nn_cnn_layers.rs reads the file.

Case "bn.<variant>_<dtype>" is a BatchNorm2d(3) with the options of VARIANTS that starts from the
state dict "init.<key>" (random weights and statistics, a count of 3) and is called on the inputs
"x<i>" in the modes of SEQUENCE. Call i stores the output "out<i>" and the state dict after it,
"state<i>.<key>"; a call with elements also stores the upstream gradient "g<i>" and the
gradients "grad_x<i>" and "grad.<parameter><i>", from torch.autograd.grad. The values are drawn
in f64 and rounded to the dtype.

Case "cnn_f64" is the CNN of M2 step 9 (TESTS order: Conv2d, BatchNorm2d, ReLU, MaxPool2d,
Conv2d, AdaptiveAvgPool2d, Flatten, Linear) with its initial state dict "param.<key>", called in
training mode on "x" with the upstream gradient "g": "out", "grad_x", "grad.<key>" for each
parameter and the state dict after it, "state.<key>"; then in evaluation mode on "x_eval":
"out_eval". The generator checks that the ReLU inputs keep a distance from 0 and that the
maximum of each pooling window keeps a distance from the other elements, so the gradients don't
depend on how ties are broken.

Usage: python tools/golden/gen_m2_cnn_layers.py (needs torch and safetensors).
"""

import copy
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_cnn_layers.safetensors"

data = {}


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def randn(*shape):
    return torch.randn(shape, dtype=torch.float64)


DTYPES = {"f64": torch.float64, "f32": torch.float32}
# name: keyword arguments of nn.BatchNorm2d
VARIANTS = {
    "momentum": {},
    "cumulative": {"momentum": None},
    "options": {"eps": 1e-3, "momentum": 0.3},
    "untracked": {"track_running_stats": False},
    "no_affine": {"affine": False},
    "no_bias": {"bias": False},
}
# (training, batch size) of each call
SEQUENCE = [(True, 4), (True, 5), (False, 3), (True, 0), (False, 2)]
C = 3


def batch_norm_case(tag, variant):
    dtype = DTYPES[tag]
    case = f"bn.{variant}_{tag}"
    layer = nn.BatchNorm2d(C, **VARIANTS[variant]).to(dtype)
    init = {}
    for key, t in layer.state_dict().items():
        if key == "weight":
            init[key] = (1 + 0.3 * randn(C)).to(dtype)
        elif key == "bias" or key == "running_mean":
            init[key] = (0.5 * randn(C)).to(dtype)
        elif key == "running_var":
            init[key] = (0.5 + torch.rand(C, dtype=torch.float64)).to(dtype)
        else:
            init[key] = torch.tensor(3)
    layer.load_state_dict(init)
    for key, t in init.items():
        put(case, f"init.{key}", t)
    params = dict(layer.named_parameters())
    for i, (training, n) in enumerate(SEQUENCE):
        layer.train(training)
        x = randn(n, C, 2, 3).to(dtype).requires_grad_()
        out = layer(x)
        put(case, f"x{i}", x)
        put(case, f"out{i}", out)
        if n > 0:
            g = randn(*out.shape).to(dtype)
            grads = torch.autograd.grad(out, [x, *params.values()], g)
            put(case, f"g{i}", g)
            put(case, f"grad_x{i}", grads[0])
            for k, gk in zip(params, grads[1:]):
                put(case, f"grad.{k}{i}", gk)
        for key, t in layer.state_dict().items():
            put(case, f"state{i}.{key}", t)


def cnn():
    """The CNN of M2 step 9, matching tests/nn_cnn_layers.rs."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )


def check_margins(model, x):
    """Asserts that the ReLU inputs and the pooling windows of model(x) have no near ties, on a
    copy of the model: a training pass updates the running statistics."""
    model = copy.deepcopy(model)
    with torch.no_grad():
        pre = model[1](model[0](x))
        margin = pre.abs().min().item()
        assert margin > 1e-6, f"a ReLU input is {margin} from the kink; pick another seed"
        windows = model[2](pre).unfold(2, 2, 2).unfold(3, 2, 2).flatten(-2)
        top = windows.topk(2, dim=-1).values
        positive = top[..., 0] > 0
        gap = (top[..., 0] - top[..., 1])[positive].min().item()
        assert gap > 1e-6, f"a pooling window has a gap of {gap}; pick another seed"
        # A window of zeros selects its first element in both libraries; with the margin above,
        # its ReLU inputs are all negative, so the gradient through it is zero either way.


def cnn_case():
    # Its own seed, so that adding cases above doesn't change the model.
    torch.manual_seed(1)
    case = "cnn_f64"
    model = cnn().double()
    for key, t in model.state_dict().items():
        put(case, f"param.{key}", t)
    x = randn(4, 3, 8, 8).requires_grad_()
    check_margins(model, x)
    out = model(x)
    g = randn(*out.shape)
    params = dict(model.named_parameters())
    grads = torch.autograd.grad(out, [x, *params.values()], g)
    put(case, "x", x)
    put(case, "g", g)
    put(case, "out", out)
    put(case, "grad_x", grads[0])
    for k, gk in zip(params, grads[1:]):
        put(case, f"grad.{k}", gk)
    for key, t in model.state_dict().items():
        put(case, f"state.{key}", t)
    model.eval()
    x_eval = randn(2, 3, 8, 8)
    check_margins(model, x_eval)
    put(case, "x_eval", x_eval)
    put(case, "out_eval", model(x_eval))


def main():
    reference.check()
    torch.manual_seed(0)
    for tag in DTYPES:
        for variant in VARIANTS:
            batch_norm_case(tag, variant)
    cnn_case()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(data, str(OUT), metadata=reference.metadata())
    print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
