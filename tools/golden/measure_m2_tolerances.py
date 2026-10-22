"""Measures the error budgets behind the floating point tolerances of the M2 golden tests and
prints a report: convolutions (M2 step 8b), batch norm (step 8c), the MLP training of step 4b and
the CNN training of step 9b; pooling has no allowance to measure. The report is the basis of the
F32 tolerances of the training tests.

For every floating point convolution case of tests/golden/m2_conv_pool.safetensors and batch
norm case of tests/golden/m2_batch_norm.safetensors it reports:

- PyTorch's own drift: its F32 / half precision results (half precision convolution gradients:
  the derived ones that the fixture stores) against PyTorch evaluated in F64 on the same rounded
  inputs;
- homura's errors against the stored results, as tests/nn_conv.rs and tests/nn_batch_norm.rs
  measure them (the tool runs those tests with HOMURA_TOLERANCE_LOG), and the fraction of the
  test's allowance they use.

Errors are the largest absolute error, the largest error relative to the expected value, and the
largest error relative to a magnitude bound, which shows how close to zero results behave: for a
convolution conv2d(|x|, |w|) + |b| (and the gradients of that for |g|), for a batch norm the
formula with |x| + |mean| for the centered input and absolute values elsewhere (see
tests/nn_batch_norm.rs). Random cases and special ones (cancellation, offsets, constant
channels, eps 0) are reported separately.

For the MLP of tests/golden/m2_training.safetensors and the CNN of
tests/golden/m2_cnn_training.safetensors it reports PyTorch's F32 run against its F64 run
(losses, gradients, parameters, optimizer state and, for the CNN, the batch norm outputs and
buffers of all ten steps and the evaluation pass; the magnitude is the largest of each tensor)
and homura's F32 and F64 errors against PyTorch's, from tests/m2_training.rs and
tests/m2_cnn_training.rs. The largest error of the CNN's batch norm outputs ("pre") is what the
margin of gen_m2_cnn_training.py is compared with.

tests/nn_pool.rs compares max_pool2d and the local adaptive_avg_pool2d kernels bit for bit, and
the [1, 1] mean path of adaptive_avg_pool2d within twice the default tolerance; the tool runs
those golden tests too and fails if they do.

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
BATCH_NORM = ROOT / "tests" / "golden" / "m2_batch_norm.safetensors"
TRAINING = ROOT / "tests" / "golden" / "m2_training.safetensors"
CNN_TRAINING = ROOT / "tests" / "golden" / "m2_cnn_training.safetensors"
SPECIAL = {"cancellation", "train_offset", "train_constant", "eval_eps_zero"}

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


def channel(t, dims):
    return t.double().view([1, -1] + [1] * (dims - 2))


def batch_norm_drift(data):
    """{(case, tensor): (abs, rel, norm)} of PyTorch's low precision batch norms against F64."""
    out = {}
    for name, c in cases(data, "bn_").items():
        if c["x"].dtype == torch.float64:
            continue
        keys = [k for k in ("x", "weight", "bias") if k in c]
        exact = {k: c[k].double().requires_grad_() for k in keys}
        running = [c[k].double() if k in c else None for k in ("running_mean", "running_var")]
        training = bool(c["opt.training"])
        options = dict(training=training, momentum=float(c["opt.momentum"]), eps=float(c["opt.eps"]))
        for _ in range(int(c["opt.steps"])):
            y = F.batch_norm(exact["x"], *running, exact.get("weight"), exact.get("bias"), **options)
        grads = torch.autograd.grad(y, [exact[k] for k in keys], c["g"].double())
        for k, (bound, t) in zip(["out"] + [f"grad_{k}" for k in keys],
                                 zip(batch_norm_bounds(c), [y, *grads])):
            out[(name, k)] = errors(c[k], t, bound)
    return out


def batch_norm_bounds(c):
    """The magnitude bounds of tests/nn_batch_norm.rs: of the output and of the gradients of the
    floating point inputs."""
    x = c["x"].double()
    dims = x.dim()
    axes = [0] + list(range(2, dims))
    n = x.numel() // x.shape[1] if x.shape[1] else 0
    if bool(c["opt.training"]):
        mean, variance = x.mean(axes, keepdim=True), x.var(axes, correction=0, keepdim=True)
    else:
        mean, variance = channel(c["running_mean"], dims), channel(c["running_var"], dims)
    invstd = 1 / (variance + float(c["opt.eps"])).sqrt()
    w = channel(c["weight"], dims).abs() if "weight" in c else 1
    big_x = (x.abs() + mean.abs()) * invstd
    out = big_x * w + (channel(c["bias"], dims).abs() if "bias" in c else 0)
    g = c["g"].double().abs()
    total, dotp = g.sum(axes, keepdim=True), (g * big_x).sum(axes, keepdim=True)
    if bool(c["opt.training"]):
        grad_x = (g + total / max(n, 1) + big_x * dotp / max(n, 1)) * invstd
    else:
        grad_x = g * invstd
    result = [out, grad_x * w]
    if "weight" in c:
        result.append(dotp.reshape(-1))
    if "bias" in c:
        result.append(total.reshape(-1))
    return result


def training_drift(data):
    """{(case, tensor): (abs, rel, norm)} of PyTorch's F32 training runs against its F64 runs;
    the magnitude is the largest of each tensor."""
    out = {}
    for kind in ("sgd", "adamw"):
        low, high = cases(data, f"{kind}_f32")[f"{kind}_f32"], cases(data, f"{kind}_f64")[f"{kind}_f64"]
        for k, t in low.items():
            if k.startswith(("opt.", "indices")):
                continue
            e = high[k]
            out[(f"{kind}_f32", k)] = errors(t, e, torch.full_like(e, e.double().abs().max().item()))
    return out


def homura_errors():
    """{test: {(case, tensor): (abs, rel, norm, used)}} from tests/nn_conv.rs,
    tests/nn_batch_norm.rs, tests/m2_training.rs and tests/m2_cnn_training.rs (the training tests
    share their case names); exits if they or the pooling golden tests fail."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for test, name in [("nn_conv", "conv2d_matches_pytorch"), ("nn_batch_norm", "batch_norm_matches_pytorch"),
                           ("m2_training", "mlp_training_matches_pytorch"),
                           ("m2_cnn_training", "cnn_training_matches_pytorch"), ("nn_pool", "matches_pytorch")]:
            log = Path(tmp) / f"{test}.jsonl"
            env = dict(os.environ, HOMURA_TOLERANCE_LOG=str(log))
            result = subprocess.run(["cargo", "test", "--release", "-q", "--test", test, name], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise SystemExit(f"tests/{test}.rs failed:\n{result.stdout}\n{result.stderr}")
            rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            out[test] = {(r["case"], r["tensor"]): (r["abs"], r["rel"], r["norm"], r["used"]) for r in rows}
    return out


def group(name, tensor):
    """(dtype, random or special, out or grad or the training run's tensor kind) of a row."""
    family, rest = name.split("_", 1)
    if family in ("sgd", "adamw"):
        parts = tensor.split(".")
        kind = parts[0].removesuffix("_eval").rstrip("0123456789")
        if kind == "after" and len(parts) == 4:
            kind = parts[3]
        elif kind == "after" and ("running" in tensor or "num_batches" in tensor):
            kind = "buffer"
        return rest, family, kind
    dtype, case = rest.split(".", 1)
    kind = "out" if tensor == "out" else "running" if tensor.startswith("running") else "grad"
    return dtype, "special" if case in SPECIAL else "random", kind


def report(title, rows, columns):
    print(f"\n{title}")
    print(f"  {'dtype':5} {'group':8} {'tensor':7} " + " ".join(f"{c:>10}" for c in columns) + "  worst case")
    groups = {}
    for (name, tensor), values in rows.items():
        groups.setdefault(group(name, tensor), []).append((name, tensor, values))
    for (dtype, kind, tensor), entries in sorted(groups.items()):
        worst = [max(v[i] for _, _, v in entries) for i in range(len(columns))]
        name, t, _ = max(entries, key=lambda e: e[2][-1])
        print(f"  {dtype:5} {kind:8} {tensor:7} " + " ".join(f"{w:10.3e}" for w in worst) + f"  {name}/{t}")


def main():
    conv = load_file(str(GOLDEN))
    batch_norm = load_file(str(BATCH_NORM))
    training = load_file(str(TRAINING))
    ours = homura_errors()
    columns = ["abs", "rel", "norm", "used"]
    pick = lambda prefix: {k: v for rows in ours.values() for k, v in rows.items() if k[0].startswith(prefix)}
    ours_of = lambda test: ours[test]
    report("conv2d: PyTorch's low precision against F64 (largest errors)", reference_drift(conv),
           ["abs", "rel", "norm"])
    report("conv2d: homura against PyTorch (largest errors and fraction of the test's allowance)",
           pick("conv_"), columns)
    report("batch_norm: PyTorch's low precision against F64", batch_norm_drift(batch_norm), ["abs", "rel", "norm"])
    report("batch_norm: homura against PyTorch", pick("bn_"), columns)
    report("MLP training: PyTorch's F32 run against its F64 run (norm: relative to each tensor's largest "
           "magnitude)", training_drift(training), ["abs", "rel", "norm"])
    report("MLP training: homura against PyTorch", ours_of("m2_training"), columns)
    report("CNN training: PyTorch's F32 run against its F64 run", training_drift(load_file(str(CNN_TRAINING))),
           ["abs", "rel", "norm"])
    report("CNN training: homura against PyTorch", ours_of("m2_cnn_training"), columns)
    print("\npooling golden tests pass: max_pool2d and the local adaptive_avg_pool2d kernels bit for bit, "
          "the [1, 1] mean path within twice the default tolerance")


if __name__ == "__main__":
    main()
