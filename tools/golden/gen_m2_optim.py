"""Generates tests/golden/m2_optim.safetensors: several steps of PyTorch's SGD, Adam and AdamW on
small parameters, with the parameters and the optimizer state after every step.
tests/optim.rs reads the file.

Case "<optimizer>_<dtype>.<name>", optimizer one of sgd, adam, adamw, stores
- the options as 0-D F64 tensors "opt.<name>": lr, and momentum, dampening, weight_decay,
  nesterov, maximize (sgd) or beta1, beta2, eps, weight_decay, amsgrad, maximize (adam, adamw),
  with 0 / 1 for flags;
- the initial parameters "init.<p>" for p in a ([2, 16]), b ([32]) and c ([2]). The optimizer has
  two parameter groups: [a, c] with the options and [b] with twice the learning rate. Before
  step 2 the learning rate of the first group is halved through param_groups;
- for each step t in 0..STEPS the gradients "grad<t>.<p>" that are set before it (a missing
  entry means the gradient is None; c never has one, b has None at step 2 and zeros at step 3),
  and after it the parameters "after<t>.<p>" and the state "after<t>.<p>.<state>" of each
  parameter that has state: momentum_buffer (sgd), step, exp_avg, exp_avg_sq and, with amsgrad,
  max_exp_avg_sq (adam, adamw).

The values are drawn in F64 and rounded to the case's dtype, so every dtype starts from the
same data. CPU runs the single-tensor implementations (foreach and fused are off by default).

tests/golden/m2_optim_edges.safetensors has the cases of EDGE, one parameter "p" in one group
with the same entries: values whose intermediate products overflow or vanish when rounded
separately (R65 of review/M2.md), tiny and subnormal values, cancelling large values, existing
momentum and moments over several steps, and F32 / F64 lengths that end in PyTorch's scalar
loop. The case name is "<optimizer>_<dtype>.<name>".

Usage: python tools/golden/gen_m2_optim.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

import reference

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden"
OUT = GOLDEN / "m2_optim.safetensors"
OUT_EDGES = GOLDEN / "m2_optim_edges.safetensors"
STEPS = 5
# Gradient scales per step, so that amsgrad's maximum and the bias corrections matter.
SCALES = [1.0, 3.0, 0.5, 2.0, 0.2]
# Multiples of 32 elements, so that PyTorch's AVX2 kernels run their vectorized loop, which takes
# two vectors at a time (32 F16 / BF16 elements), for every dtype; its scalar loop for the
# remaining elements rounds F16 / BF16 scaled additions differently (see EDGE below).
SHAPES = {"a": (2, 16), "b": (32,), "c": (2,)}

reference.check()
torch.manual_seed(0)
data = {}
DTYPES = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}

SGD_CASES = {
    "plain": dict(lr=0.1),
    "momentum": dict(lr=0.1, momentum=0.9),
    "full": dict(lr=0.05, momentum=0.9, dampening=0.3, weight_decay=0.1),
    "nesterov": dict(lr=0.05, momentum=0.8, weight_decay=0.01, nesterov=True),
    "maximize": dict(lr=0.05, momentum=0.9, weight_decay=0.1, maximize=True),
}
ADAM_CASES = {
    "default": dict(lr=0.01),
    "full": dict(lr=0.02, betas=(0.8, 0.99), eps=1e-6, weight_decay=0.1, amsgrad=True),
    "maximize": dict(lr=0.01, weight_decay=0.05, maximize=True),
}
ADAMW_CASES = {
    "default": dict(lr=0.01),
    "full": dict(
        lr=0.02, betas=(0.85, 0.995), eps=1e-7, weight_decay=0.1, amsgrad=True, maximize=True
    ),
}


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def scalar(v):
    return torch.tensor(float(v), dtype=torch.float64)


def options(kind, opt):
    """The options of the optimizer `opt` (its defaults), as stored in the file."""
    d = opt.defaults
    if kind == "sgd":
        names = ["lr", "momentum", "dampening", "weight_decay", "nesterov", "maximize"]
        out = {n: d[n] for n in names}
    else:
        out = {n: d[n] for n in ["lr", "eps", "weight_decay", "amsgrad", "maximize"]}
        out["beta1"], out["beta2"] = d["betas"]
    return out


# The data of each run, in F64.
INIT = {p: torch.randn(s, dtype=torch.float64) for p, s in SHAPES.items()}
GRADS = []
for t in range(STEPS):
    g = {p: torch.randn(SHAPES[p], dtype=torch.float64) * SCALES[t] for p in ("a", "b")}
    if t == 2:
        del g["b"]
    elif t == 3:
        g["b"] = torch.zeros(SHAPES["b"], dtype=torch.float64)
    GRADS.append(g)


def run(kind, name, kwargs, dtype_name):
    dtype = DTYPES[dtype_name]
    case = f"{kind}_{dtype_name}.{name}"
    params = {p: torch.nn.Parameter(v.to(dtype)) for p, v in INIT.items()}
    cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind]
    lr = kwargs["lr"]
    groups = [
        {"params": [params["a"], params["c"]]},
        {"params": [params["b"]], "lr": 2 * lr},
    ]
    opt = cls(groups, **kwargs)
    for n, v in options(kind, opt).items():
        put(case, f"opt.{n}", scalar(v))
    for p, v in params.items():
        put(case, f"init.{p}", v)
    for t in range(STEPS):
        if t == 2:
            opt.param_groups[0]["lr"] *= 0.5
        for p, v in params.items():
            g = GRADS[t].get(p)
            v.grad = None if g is None else g.to(dtype)
            if g is not None:
                put(case, f"grad{t}.{p}", v.grad)
        opt.step()
        for p, v in params.items():
            put(case, f"after{t}.{p}", v)
            for s, value in opt.state.get(v, {}).items():
                assert value.grad_fn is None and not value.requires_grad
                put(case, f"after{t}.{p}.{s}", value)
    # c never has a gradient, so neither state nor an update, even with weight decay.
    assert params["c"] not in opt.state
    assert torch.equal(params["c"].detach(), INIT["c"].to(dtype))


for dtype_name in DTYPES:
    for kind, cases in (("sgd", SGD_CASES), ("adam", ADAM_CASES), ("adamw", ADAMW_CASES)):
        for name, kwargs in cases.items():
            run(kind, name, kwargs, dtype_name)

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT}")


def full(n, v):
    return torch.full((n,), v, dtype=torch.float64)


def tiny(n, scale):
    return torch.randn(n, dtype=torch.float64) * scale


# name -> (optimizer, dtype, options, initial value, gradients per step). Values are F64 and
# rounded to the dtype.
EDGE = {
    # R65: Adam's update -step_size * exp_avg / denom vanishes in F16 if the product is rounded
    # before the division.
    "adam_f16.small_update": ("adam", "f16", dict(lr=1e-3, eps=1e-4), full(32, 0.0), [full(32, 1e-5)]),
    "adamw_f16.small_update": ("adamw", "f16", dict(lr=1e-3, eps=1e-4), full(32, 0.0), [full(32, 1e-5)]),
    "adam_bf16.small_update": ("adam", "bf16", dict(lr=1e-3, eps=1e-6), full(32, 0.0), [full(32, 1e-9)]),
    # R65: lr * grad overflows although p - lr * grad is finite.
    "sgd_f16.large_update": ("sgd", "f16", dict(lr=60000.0), full(32, 60000.0), [full(32, 2.0)]),
    "sgd_bf16.large_update": ("sgd", "bf16", dict(lr=3e38), full(32, 3e38), [full(32, 2.0)]),
    "sgd_f32.large_update": ("sgd", "f32", dict(lr=3e38), full(32, 3e38), [full(32, 2.0)]),
    "sgd_f64.large_update": ("sgd", "f64", dict(lr=1e308), full(32, 1e308), [full(32, 2.0)]),
    # Weight decay and momentum with large values that cancel.
    "sgd_f32.large_momentum": (
        "sgd", "f32", dict(lr=1.0, momentum=0.9, weight_decay=4.0), full(37, 1e38),
        [full(37, -2e38), full(37, 1.5e38), full(37, -1e38)],
    ),
    "sgd_f16.large_momentum": (
        "sgd", "f16", dict(lr=2.0, momentum=0.9, nesterov=True), full(32, 60000.0),
        [full(32, 30000.0), full(32, -20000.0), full(32, -10000.0)],
    ),
    # Tiny and subnormal gradients and parameters, several steps of state.
    "sgd_f16.tiny": (
        "sgd", "f16", dict(lr=1e-3, momentum=0.9, dampening=0.1, weight_decay=0.5),
        tiny(32, 1e-4), [tiny(32, 1e-5) for _ in range(3)],
    ),
    "sgd_bf16.tiny": (
        "sgd", "bf16", dict(lr=1e-3, momentum=0.9, weight_decay=0.5), tiny(32, 1e-30),
        [tiny(32, 1e-33) for _ in range(3)],
    ),
    "adam_f16.tiny": (
        "adam", "f16", dict(lr=1e-3, eps=1e-7, weight_decay=0.1, amsgrad=True), tiny(32, 1e-3),
        [tiny(32, 1e-6 * (t + 1)) for t in range(3)],
    ),
    "adamw_bf16.tiny": (
        "adamw", "bf16", dict(lr=1e-2, eps=1e-30, weight_decay=0.1), tiny(32, 1e-20),
        [tiny(32, 1e-25) for _ in range(3)],
    ),
    "adam_f32.subnormal": (
        "adam", "f32", dict(lr=1e-3, eps=1e-40, amsgrad=True), tiny(37, 1e-38),
        [tiny(37, 1e-40) for _ in range(3)],
    ),
    "adamw_f64.subnormal": (
        "adamw", "f64", dict(lr=1e-3, eps=1e-310, maximize=True), tiny(37, 1e-300),
        [tiny(37, 1e-310) for _ in range(3)],
    ),
    # Lengths that end in the scalar loop (F32: 8 per vector, F64: 4), with every update.
    "sgd_f32.tail": (
        "sgd", "f32", dict(lr=0.1, momentum=0.9, dampening=0.2, weight_decay=0.1), tiny(13, 1.0),
        [tiny(13, 1.0) for _ in range(3)],
    ),
    "adamw_f64.tail": (
        "adamw", "f64", dict(lr=0.1, weight_decay=0.1, amsgrad=True), tiny(7, 1.0),
        [tiny(7, 1.0) for _ in range(3)],
    ),
    "adam_f32.tail": (
        "adam", "f32", dict(lr=0.1, betas=(0.3, 0.6), weight_decay=0.1), tiny(5, 1.0),
        [tiny(5, 1.0) for _ in range(3)],
    ),
}

edges = {}
for case, (kind, dtype_name, kwargs, init, grads) in EDGE.items():
    dtype = DTYPES[dtype_name]
    assert case.startswith(f"{kind}_{dtype_name}.")
    cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind]
    p = torch.nn.Parameter(init.to(dtype))
    opt = cls([p], **kwargs)

    def store(name, t):
        key = f"{case}/{name}"
        assert key not in edges, key
        edges[key] = t.detach().contiguous().clone()

    for n, v in options(kind, opt).items():
        store(f"opt.{n}", scalar(v))
    store("init.p", p)
    for t, g in enumerate(grads):
        p.grad = g.to(dtype)
        store(f"grad{t}.p", p.grad)
        opt.step()
        store(f"after{t}.p", p)
        for s, value in opt.state.get(p, {}).items():
            store(f"after{t}.p.{s}", value)

# The reference values of R65 that separate rounding gets wrong.
f = lambda case: edges[f"{case}/after0.p"].double()
assert torch.all(f("adam_f16.small_update") == -0.00010132789611816406)
assert torch.all(f("adamw_f16.small_update") == -0.00010132789611816406)
assert torch.all(f("adam_bf16.small_update") != 0)
assert torch.all(f("sgd_f16.large_update") == -60000.0)
assert torch.all(torch.isfinite(f("sgd_bf16.large_update")) & (f("sgd_bf16.large_update") < 0))
assert torch.all(torch.isfinite(f("sgd_f32.large_update")) & (f("sgd_f32.large_update") < 0))
assert torch.all(f("sgd_f64.large_update") == -1e308)
for case in EDGE:
    for k, v in edges.items():
        if k.startswith(case + "/after") and v.is_floating_point():
            assert torch.all(torch.isfinite(v)), k
save_file(edges, str(OUT_EDGES), metadata=reference.metadata())
print(f"wrote {len(edges)} tensors to {OUT_EDGES}")
