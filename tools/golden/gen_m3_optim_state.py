"""Generates tests/golden/m3_optim_state.pt: PyTorch's SGD, Adam and AdamW trained for six steps,
with their state dicts, so that tests/optim_state_dict.rs can check that homura resumes training
from PyTorch's state dicts bit for bit (M3 step 6). tools/golden/check_m3_optim_state.py checks the
other direction with the same cases.

The file holds a dict of cases "<name>_<dtype>", each a dict with
- "kind": "sgd", "adam" or "adamw";
- "init": the parameters a ([2, 16]), b ([32]) and c ([2]) before training. The optimizer has two
  parameter groups, [a, c] and [b]; b's learning rate is twice a's;
- "grads": for each of the six steps a dict of the gradients set before it; a missing parameter
  has the gradient None. c never has one (it never gets state), b has None at step 2 and zeros
  at step 5;
- "options": for each step a dict of options set in every group before it (empty for most), to
  switch amsgrad or momentum off and on again;
- "optimizer0": the optimizer's state dict before the first step, without state;
- "saved_at": the number of steps after which "params" and "optimizer" are saved;
- "params", "optimizer": the parameters and the optimizer's state dict after "saved_at" steps;
- "after": for each step after "saved_at", the parameters ("params") and the optimizer's state
  ("state") after it.

The values are drawn in F64 and rounded to the case's dtype. Case "adam_default_f64_f64" is made
with the default dtype F64, so its step counts are F64; in the others they are F32.

Usage: python tools/golden/gen_m3_optim_state.py (needs torch).
"""

import copy
from pathlib import Path

import torch

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m3_optim_state.pt"
STEPS = 6
SHAPES = {"a": (2, 16), "b": (32,), "c": (2,)}
SCALES = [1.0, 3.0, 0.5, 2.0, 0.2, 1.5]
DTYPES = {"f32": torch.float32, "f64": torch.float64}

# name: (kind, constructor options, saved_at, options per step, dtypes, default dtype)
CASES = {
    "sgd": ("sgd", dict(lr=0.1, momentum=0.9, dampening=0.1, weight_decay=0.01), 3, {}),
    "sgd_nesterov": (
        "sgd",
        dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01, maximize=True),
        3,
        {},
    ),
    # The momentum buffer is kept while the momentum is 0, and used again after.
    "sgd_momentum_off": (
        "sgd",
        dict(lr=0.1, momentum=0.9),
        3,
        {2: dict(momentum=0.0), 3: dict(momentum=0.0), 4: dict(momentum=0.9)},
    ),
    "adam": ("adam", dict(lr=0.01, amsgrad=True, weight_decay=0.01), 3, {}),
    "adamw": ("adamw", dict(lr=0.01, betas=(0.8, 0.99), weight_decay=0.05), 3, {}),
    # max_exp_avg_sq is kept while amsgrad is off, and used again after.
    "adam_amsgrad_switch": (
        "adam",
        dict(lr=0.01, amsgrad=True),
        3,
        {2: dict(amsgrad=False), 3: dict(amsgrad=False), 4: dict(amsgrad=True)},
    ),
    # Saved before the first step, with no state.
    "adam_fresh": ("adam", dict(lr=0.01), 0, {}),
}

reference.check()


def draw(shape, dtype, generator):
    return torch.randn(shape, dtype=torch.float64, generator=generator).to(dtype)


def make(kind, params, options):
    groups = [{"params": [params["a"], params["c"]]}, {"params": [params["b"]], "lr": 2 * options["lr"]}]
    cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind]
    return cls(groups, **options)


def snapshot(params):
    return {k: p.detach().clone() for k, p in params.items()}


def case(kind, options, saved_at, switches, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    init = {k: draw(s, dtype, generator) for k, s in SHAPES.items()}
    grads = []
    for t in range(STEPS):
        g = {}
        g["a"] = draw(SHAPES["a"], dtype, generator) * SCALES[t]
        if t == 4:
            g["b"] = torch.zeros(SHAPES["b"], dtype=dtype)
        elif t != 1:
            g["b"] = draw(SHAPES["b"], dtype, generator) * SCALES[t]
        grads.append(g)
    params = {k: torch.nn.Parameter(v.clone()) for k, v in init.items()}
    opt = make(kind, params, options)
    out = {
        "kind": kind,
        "init": init,
        "grads": grads,
        "options": [switches.get(t, {}) for t in range(STEPS)],
        "optimizer0": copy.deepcopy(opt.state_dict()),
        "saved_at": saved_at,
        "after": [],
    }
    for t in range(STEPS):
        if t == saved_at:
            out["params"] = snapshot(params)
            out["optimizer"] = copy.deepcopy(opt.state_dict())
        for group in opt.param_groups:
            group.update(switches.get(t, {}))
        for k, p in params.items():
            p.grad = grads[t].get(k)
            if p.grad is not None:
                p.grad = p.grad.clone()
        opt.step()
        if t >= saved_at:
            state = copy.deepcopy(opt.state_dict()["state"])
            out["after"].append({"params": snapshot(params), "state": state})
    return out


def main():
    cases = {}
    seed = 0
    for name, (kind, options, saved_at, switches) in CASES.items():
        for dname, dtype in DTYPES.items():
            seed += 1
            cases[f"{name}_{dname}"] = case(kind, options, saved_at, switches, dtype, seed)
    torch.set_default_dtype(torch.float64)
    try:
        cases["adam_default_f64_f64"] = case("adam", dict(lr=0.01), 3, {}, torch.float64, 100)
    finally:
        torch.set_default_dtype(torch.float32)
    step = cases["adam_default_f64_f64"]["optimizer"]["state"][0]["step"]
    assert step.dtype == torch.float64, step.dtype
    assert cases["adam_f64"]["optimizer"]["state"][0]["step"].dtype == torch.float32
    # c never has a gradient, so it never has state; b has state from step 1.
    for name, c in cases.items():
        if c["saved_at"] > 0:
            assert set(c["optimizer"]["state"]) == {0, 2}, name
    switch = cases["adam_amsgrad_switch_f32"]["optimizer"]
    assert switch["param_groups"][0]["amsgrad"] is False
    assert "max_exp_avg_sq" in switch["state"][0]
    off = cases["sgd_momentum_off_f32"]["optimizer"]
    assert off["param_groups"][0]["momentum"] == 0.0 and "momentum_buffer" in off["state"][0]
    torch.save(cases, OUT)
    loaded = torch.load(OUT, weights_only=True)
    assert loaded.keys() == cases.keys()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cases)} cases)")


main()
