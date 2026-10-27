"""Checks M3 step 6 against PyTorch: the reference behavior that src/optim documents, and that
PyTorch resumes training from the optimizer state dicts homura saves, bit for bit.

Run the Rust tests first; they write target/tmp/m3_optim_state/<case>.pt, one per case of
tests/golden/m3_optim_state.pt (from gen_m3_optim_state.py), each a dict with homura's "params"
and "optimizer" state dict after the case's "saved_at" steps, and "after": the parameters and
state after each later step of homura's training:

    cargo test --test optim_state_dict --test optim_default_dtype

For each case, this loads homura's file with torch.load (weights_only=True, and mmap=True),
checks that the state dict has the layout of PyTorch's own (key order, option values, state
names) and the values of PyTorch's state dict at the same step, then loads it into a PyTorch
optimizer of the same groups, trains the remaining steps with the case's gradients and options,
and compares the parameters and state with homura's after every step.

Usage: python tools/golden/check_m3_optim_state.py (needs torch).
"""

import io
from pathlib import Path

import torch

import reference

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "m3_optim_state.pt"
WRITTEN = ROOT / "target" / "tmp" / "m3_optim_state"
PARAMS = ["a", "b", "c"]
CLASSES = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def bits(t):
    t = t.detach().contiguous()
    return t.view({torch.float32: torch.int32, torch.float64: torch.int64}[t.dtype])


def same(a, b, what):
    assert torch.is_tensor(a) and torch.is_tensor(b), what
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} {b.shape}"
    assert a.device == b.device, f"{what}: device"
    assert torch.equal(bits(a), bits(b)), f"{what}: values differ"


def same_option(a, b, what):
    if isinstance(b, tuple):
        assert isinstance(a, tuple) and len(a) == len(b), what
        for x, y in zip(a, b):
            same_option(x, y, what)
    elif isinstance(b, bool) or b is None:
        assert a is b, f"{what}: {a!r} is not {b!r}"
    else:
        # PyTorch keeps the ints it was given; homura saves floats.
        assert type(a) is float and not isinstance(a, bool), f"{what}: {a!r}"
        assert a == float(b), f"{what}: {a!r} != {b!r}"


def same_state(a, b, what):
    assert type(a) is dict, f"{what}: {type(a)}"
    assert list(a) == list(b), f"{what}: parameters {list(a)} {list(b)}"
    for n in a:
        assert type(n) is int, what
        assert list(a[n]) == list(b[n]), f"{what}: names of {n}"
        for name in a[n]:
            same(a[n][name], b[n][name], f"{what}: state[{n}][{name}]")


def same_state_dict(a, b, what):
    assert type(a) is dict and list(a) == ["state", "param_groups"], what
    assert type(a["param_groups"]) is list and len(a["param_groups"]) == len(b["param_groups"])
    for g, (x, y) in enumerate(zip(a["param_groups"], b["param_groups"])):
        assert type(x) is dict and list(x) == list(y), f"{what}: keys of group {g}: {list(x)}"
        assert x["params"] == y["params"], what
        for k in x:
            if k != "params":
                same_option(x[k], y[k], f"{what}: group {g} {k}")
    same_state(a["state"], b["state"], what)


def make(kind, params):
    groups = [{"params": [params["a"], params["c"]]}, {"params": [params["b"]]}]
    # The options are what the state dict loads.
    return CLASSES[kind](groups, lr=0.0)


def check_case(name, case):
    path = WRITTEN / f"{name}.pt"
    written = torch.load(path, weights_only=True)
    mapped = torch.load(path, weights_only=True, mmap=True)
    assert written["kind"] == case["kind"]
    saved_at = case["saved_at"]
    for k in PARAMS:
        same(written["params"][k], case["params"][k], f"{name}: params {k}")
    same_state_dict(written["optimizer"], case["optimizer"], f"{name}: optimizer")
    same_state_dict(mapped["optimizer"], case["optimizer"], f"{name}: optimizer (mmap)")

    params = {k: torch.nn.Parameter(written["params"][k].clone()) for k in PARAMS}
    opt = make(case["kind"], params)
    opt.load_state_dict(written["optimizer"])
    for i, t in enumerate(range(saved_at, len(case["grads"]))):
        for group in opt.param_groups:
            group.update(case["options"][t])
        for k, p in params.items():
            g = case["grads"][t].get(k)
            p.grad = None if g is None else g.clone()
        opt.step()
        expected = written["after"][i]
        for k in PARAMS:
            same(params[k].detach(), expected["params"][k], f"{name}: step {t} {k}")
        same_state(opt.state_dict()["state"], expected["state"], f"{name}: step {t}")
    return len(case["grads"]) - saved_at


def check_reference():
    """What src/optim documents about PyTorch's state dicts."""
    a = torch.nn.Parameter(torch.randn(2, 3))
    a.grad = torch.randn(2, 3)
    adam = torch.optim.Adam([a], lr=0.1)
    adam.step()
    sd = adam.state_dict()
    # Loading another optimizer's state dict succeeds; it fails in step, which needs options
    # that the state dict doesn't have.
    sgd = torch.optim.SGD([a], lr=0.1)
    sgd.load_state_dict(sd)
    assert "betas" in sgd.param_groups[0] and "momentum" not in sgd.param_groups[0]
    try:
        sgd.step()
    except KeyError as e:
        assert e.args == ("momentum",), e
    else:
        raise AssertionError("SGD with Adam's options stepped")
    # The state is shared with the state dict when the dtype is the parameter's, and step is
    # never converted.
    other = torch.optim.Adam([a], lr=0.1)
    other.load_state_dict(sd)
    assert other.state[a]["exp_avg"].data_ptr() == sd["state"][0]["exp_avg"].data_ptr()
    a64 = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float64))
    other = torch.optim.Adam([a64], lr=0.1)
    other.load_state_dict(sd)
    assert other.state[a64]["exp_avg"].dtype == torch.float64
    assert other.state[a64]["step"].dtype == torch.float32
    # A state tensor of another shape is accepted until step.
    bad = {"state": {0: dict(sd["state"][0], exp_avg=torch.zeros(3, 2))}, "param_groups": sd["param_groups"]}
    torch.optim.Adam([a], lr=0.1).load_state_dict(bad)
    # The messages for groups that don't match.
    for groups, message in [
        ([{"params": [a]}, {"params": [a64]}], "different number of parameter groups"),
        ([{"params": [a, a64]}], "doesn't match the size of optimizer's group"),
    ]:
        try:
            torch.optim.Adam(groups, lr=0.1).load_state_dict(sd)
        except ValueError as e:
            assert message in str(e), e
        else:
            raise AssertionError(message)
    # Options missing from older state dicts default, a step saved as a Python number becomes a
    # tensor of the default scalar dtype, and an empty entry or a None buffer is no state.
    old = {
        "state": {0: {"step": 3, "exp_avg": torch.zeros(2, 3), "exp_avg_sq": torch.zeros(2, 3)}},
        "param_groups": [{"lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0, "params": [0]}],
    }
    other = torch.optim.Adam([a], lr=0.5)
    other.load_state_dict(old)
    group = other.param_groups[0]
    assert (group["amsgrad"], group["maximize"], group["capturable"], group["decoupled_weight_decay"]) == (False,) * 4
    assert group["foreach"] is None and group["fused"] is None
    assert other.state[a]["step"].dtype == torch.float32 and other.state[a]["step"].item() == 3
    other.load_state_dict({"state": {0: {}}, "param_groups": sd["param_groups"]})
    other.step()
    assert other.state[a]["step"].item() == 1
    sgd = torch.optim.SGD([a], lr=0.1, momentum=0.9)
    sgd.load_state_dict({"state": {0: {"momentum_buffer": None}}, "param_groups": sgd.state_dict()["param_groups"]})
    sgd.step()
    assert torch.equal(sgd.state[a]["momentum_buffer"], a.grad)
    # Optimizers of named parameters save their names, first in each group.
    named = torch.optim.Adam(torch.nn.Linear(2, 1).named_parameters(), lr=0.1).state_dict()
    keys = list(named["param_groups"][0])
    assert keys[0] == "param_names" and keys[-1] == "params", keys
    assert named["param_groups"][0]["param_names"] == ["weight", "bias"]
    # The state dict as torch.save writes it reads back with weights_only.
    buffer = io.BytesIO()
    torch.save(sd, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert type(loaded["param_groups"][0]["betas"]) is tuple


def main():
    reference.check()
    check_reference()
    cases = torch.load(GOLDEN, weights_only=True)
    steps = 0
    for name, case in cases.items():
        if name == "adam_default_f64_f64":
            torch.set_default_dtype(torch.float64)
        try:
            steps += check_case(name, case)
        finally:
            torch.set_default_dtype(torch.float32)
    print(f"{len(cases)} / {len(cases)} cases resumed in PyTorch bit for bit ({steps} steps)")


main()
