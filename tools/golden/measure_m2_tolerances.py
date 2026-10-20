"""Measures the error budgets behind the floating point tolerances of the M2 golden tests and
prints a report (M2 step 8b: convolutions; pooling has no allowance to measure).

For every floating point convolution case of tests/golden/m2_conv_pool.safetensors it reports:

- PyTorch's own drift: its F32 / half precision results (half precision gradients: the derived
  ones that the fixture stores) against PyTorch evaluated in F64 on the same rounded inputs;
- homura's errors against the stored results, as tests/nn_conv.rs measures them (the tool runs
  that test with HOMURA_TOLERANCE_LOG), and the fraction of the test's allowance they use.

Errors are the largest absolute error, the largest error relative to the expected value, and the
largest error relative to the magnitude bound conv2d(|x|, |w|) + |b| (and the gradients of that
for |g|), which shows how close to zero results behave. Random cases and special ones
(cancellation) are reported separately. tests/nn_pool.rs compares max_pool2d and the local
adaptive_avg_pool2d kernels bit for bit, and the [1, 1] mean path of adaptive_avg_pool2d within
twice the default tolerance; the tool runs those golden tests too and fails if they do.

Usage: python tools/golden/measure_m2_tolerances.py (needs torch, safetensors and cargo).
"""

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import reference

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "m2_conv_pool.safetensors"
SPECIAL = {"cancellation"}

reference.check()
# "same" padding warns about even kernels.
warnings.simplefilter("ignore")


def cases(data, prefix):
    names = sorted({k.split("/")[0] for k in data if k.startswith(prefix)})
    return {n: {k.split("/", 1)[1]: v for k, v in data.items() if k.startswith(n + "/")} for n in names}


def conv(c, x, w, b):
    if "opt.transpose" in c:
        x = x.transpose(-2, -1)
    if "opt.transpose_w" in c:
        w = w.transpose(-2, -1)
    padding = "valid" if "opt.valid" in c else "same" if "opt.same" in c else c["opt.padding"].tolist()
    return F.conv2d(x, w, b, stride=c["opt.stride"].tolist(), padding=padding,
                    dilation=c["opt.dilation"].tolist(), groups=int(c["opt.groups"]))


def errors(actual, expected, bound):
    """The largest absolute, relative and bound-relative errors of the finite elements."""
    a, e, m = actual.double(), expected.double(), bound.double()
    finite = e.isfinite() & a.isfinite()
    err = (a - e).abs()[finite]
    if err.numel() == 0:
        return 0.0, 0.0, 0.0
    e, m = e[finite], m[finite]
    rel = (err[e != 0] / e[e != 0].abs()).max().item() if (e != 0).any() else 0.0
    norm = (err[m > 0] / m[m > 0]).max().item() if (m > 0).any() else 0.0
    return err.max().item(), rel, norm


def reference_drift(data):
    """{(case, tensor): (abs, rel, norm)} of PyTorch's low precision results against F64."""
    out = {}
    for name, c in cases(data, "conv_").items():
        dtype = c["x"].dtype
        if not dtype.is_floating_point or dtype == torch.float64:
            continue
        keys = [k for k in ("x", "w", "b") if k in c]
        exact = {k: c[k].double().requires_grad_() for k in keys}
        absolute = {k: c[k].double().abs().requires_grad_() for k in keys}
        y = conv(c, exact["x"], exact["w"], exact.get("b"))
        bound = conv(c, absolute["x"], absolute["w"], absolute.get("b"))
        g = c["g"].double()
        out[(name, "out")] = errors(c["out"], y, bound)
        exact_grads = torch.autograd.grad(y, [exact[k] for k in keys], g)
        bound_grads = torch.autograd.grad(bound, [absolute[k] for k in keys], g.abs())
        for k, t, m in zip(keys, exact_grads, bound_grads):
            out[(name, f"grad_{k}")] = errors(c[f"grad_{k}"], t, m)
    return out


def homura_errors():
    """{(case, tensor): (abs, rel, norm, used)} from tests/nn_conv.rs; exits if it or the pooling
    golden tests fail."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "tolerances.jsonl"
        env = dict(os.environ, HOMURA_TOLERANCE_LOG=str(log))
        run = lambda *args: subprocess.run(["cargo", "test", "--release", "-q", *args], cwd=ROOT, env=env,
                                           capture_output=True, text=True)
        conv_run = run("--test", "nn_conv", "conv2d_matches_pytorch")
        if conv_run.returncode != 0:
            raise SystemExit(f"tests/nn_conv.rs failed:\n{conv_run.stdout}\n{conv_run.stderr}")
        pool_run = run("--test", "nn_pool", "matches_pytorch")
        if pool_run.returncode != 0:
            raise SystemExit(f"tests/nn_pool.rs failed:\n{pool_run.stdout}\n{pool_run.stderr}")
        rows = [json.loads(line) for line in log.read_text().splitlines()]
    return {(r["case"], r["tensor"]): (r["abs"], r["rel"], r["norm"], r["used"]) for r in rows}


def report(title, rows, columns):
    print(f"\n{title}")
    print(f"  {'dtype':5} {'group':8} {'tensor':7} " + " ".join(f"{c:>10}" for c in columns) + "  worst case")
    groups = {}
    for (name, tensor), values in rows.items():
        dtype = name.split("_")[1].split(".")[0]
        kind = "special" if name.split(".", 1)[1] in SPECIAL else "random"
        groups.setdefault((dtype, kind, "out" if tensor == "out" else "grad"), []).append((name, tensor, values))
    for (dtype, kind, tensor), entries in sorted(groups.items()):
        worst = [max(v[i] for _, _, v in entries) for i in range(len(columns))]
        name, t, _ = max(entries, key=lambda e: e[2][-1])
        print(f"  {dtype:5} {kind:8} {tensor:7} " + " ".join(f"{w:10.3e}" for w in worst) + f"  {name}/{t}")


def main():
    data = load_file(str(GOLDEN))
    drift = reference_drift(data)
    report("PyTorch: low precision against F64 (largest errors)", drift, ["abs", "rel", "norm"])
    ours = homura_errors()
    report("homura against PyTorch (largest errors and fraction of the test's allowance)", ours,
           ["abs", "rel", "norm", "used"])
    print("\npooling golden tests pass: max_pool2d and the local adaptive_avg_pool2d kernels bit for bit, "
          "the [1, 1] mean path within twice the default tolerance")


if __name__ == "__main__":
    main()
