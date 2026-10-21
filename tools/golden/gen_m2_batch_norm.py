"""Generates tests/golden/m2_batch_norm.safetensors: F.batch_norm (M2 step 8c), with PyTorch's
outputs, gradients and updated running statistics. tests/nn_batch_norm.rs reads the file.

Case "bn_<dtype>.<name>" stores the input "x", the optional "weight", "bias", "running_mean"
and "running_var" (their values before the call), the options as 0-D tensors ("opt.training",
I64 0 or 1; "opt.momentum" and "opt.eps", F64; "opt.steps", I64, the number of calls with the
same running statistics), the upstream gradient "g", the output "out" of the last call, the
gradient "grad_<input>" of each floating point input of that call, from
torch.autograd.grad(out, inputs, g), and in training with running statistics their values after
the calls, "running_mean_out" and "running_var_out". The inputs keep their stored dtype: a half
precision case with F32 parameters and statistics is the mixed dtype path.

Cases of SECOND (f64 only) also store "gg_<input>" and the second derivatives
"grad2_<variable>": the gradients of sum(grad_<input> * gg_<input>), with the first gradients
computed with create_graph and "g" as a variable too; a variable whose second derivative is
None has no entry: the bias, and in evaluation mode the input, whose first gradients don't
depend on any variable.

PyTorch's CPU kernel for inputs with one element per channel and sample ([N, C]) sums in buffers
per thread, so their F32 results depend on the number of threads (generated with 16).

Usage: python tools/golden/gen_m2_batch_norm.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_batch_norm.safetensors"

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


def put(case, name, t):
    key = f"{case}/{name}"
    assert key not in data, key
    data[key] = t.detach().contiguous().clone()


def randn(*shape):
    return torch.randn(shape, dtype=torch.float64)


def params(c, weight=True, bias=True, stats=True):
    """Per-channel inputs: weights about ±1, running variances in [0.5, 1.5]."""
    out = {}
    if weight:
        out["weight"] = randn(c)
    if bias:
        out["bias"] = randn(c)
    if stats:
        out["running_mean"] = randn(c) * 0.5
        out["running_var"] = torch.rand(c, dtype=torch.float64) + 0.5
    return out


# Each case: inputs {name: f64 tensor}, options, the dtypes, whether F64 second derivatives are
# stored. Inputs named in "keep" keep their dtype (F32 parameters of a half precision input).
CASES = {}
ALL = ["f64", "f32", "f16", "bf16"]
FULL = ["f64", "f32"]
HALF = ["f16", "bf16"]


def case(name, x, inputs, dtypes=FULL, second=False, keep=(), training=True, momentum=0.1,
         eps=1e-5, steps=1):
    inputs = {"x": x, **inputs}
    options = {"training": training, "momentum": momentum, "eps": eps, "steps": steps}
    CASES[name] = (inputs, options, dtypes, second, keep)


case("train_2d", randn(8, 3), params(3), ALL, second=True)
case("train_3d", randn(4, 3, 5), params(3), ALL)
case("train_4d", randn(2, 4, 3, 3), params(4), ALL, second=True)
case("train_5d", randn(2, 3, 2, 2, 3), params(3))
case("train_one_sample", randn(1, 3, 4), params(3))
case("train_no_stats", randn(4, 3, 2), params(3, stats=False), second=True)
case("train_plain", randn(5, 2, 3), {}, ALL, second=True)
case("train_weight_only", randn(4, 3, 2), params(3, bias=False), second=True)
# Not second: PyTorch's batchnorm_double_backward fails without a weight but with a bias (see
# check_m2_batch_norm.py).
case("train_bias_only", randn(4, 3, 2), params(3, weight=False))
case("train_momentum_0", randn(4, 3, 2), params(3), momentum=0.0)
case("train_momentum_1", randn(4, 3, 2), params(3), momentum=1.0)
case("train_momentum_2", randn(4, 3, 2), params(3), momentum=2.0)
case("train_eps", randn(4, 3, 2) * 0.01, params(3), eps=1e-3)
case("train_twice", randn(3, 2, 4), params(2), steps=2, momentum=0.3)
case("train_offset", randn(4, 2, 8) + 1000.0, params(2))
case("train_constant", torch.tensor([[[2.0, 2.0], [1.0, -1.0]], [[2.0, 2.0], [3.0, 0.5]]]),
     params(2), second=True)
case("train_empty_batch", randn(0, 3, 2), params(3), second=True)
case("train_nonfinite", torch.tensor([[INF, 1.0, 0.5], [2.0, NAN, 1.5], [3.0, 2.0, -1.0],
                                      [0.0, 1.0, 2.0]]), params(3))
case("eval_2d", randn(6, 3), params(3), ALL, second=True, training=False)
case("eval_4d", randn(2, 3, 2, 3), params(3), ALL, second=True, training=False)
case("eval_plain", randn(4, 2, 3), params(2, weight=False, bias=False), second=True,
     training=False)
case("eval_weight_only", randn(4, 3), params(3, bias=False), training=False)
case("eval_eps_zero", randn(3, 2, 2), params(2), training=False, eps=0.0)
case("eval_empty_batch", randn(0, 2, 3), params(2), training=False)
mixed = {k: v.float() for k, v in params(3).items()}
case("mixed", randn(4, 3, 5), mixed, HALF, keep=tuple(mixed))
mixed = {k: v.float() for k, v in params(3).items()}
case("mixed_eval", randn(4, 3, 5), mixed, HALF, keep=tuple(mixed), training=False)
mixed = {k: v.float() for k, v in params(2, bias=False, stats=False).items()}
case("mixed_no_stats", randn(3, 2, 6), mixed, HALF, keep=tuple(mixed))


def generate(key, dtype_name):
    inputs, options, dtypes, second, keep = CASES[key]
    dtype = DTYPES[dtype_name]
    second = second and dtype_name == "f64"
    name = f"bn_{dtype_name}.{key}"
    args = {}
    for k, v in inputs.items():
        v = v.clone() if k in keep else v.to(dtype)
        if k not in ("running_mean", "running_var"):
            v.requires_grad_()
        args[k] = v
        put(name, k, v)
    put(name, "opt.training", torch.tensor(int(options["training"])))
    put(name, "opt.momentum", torch.tensor(options["momentum"], dtype=torch.float64))
    put(name, "opt.eps", torch.tensor(options["eps"], dtype=torch.float64))
    put(name, "opt.steps", torch.tensor(options["steps"]))
    for _ in range(options["steps"]):
        out = F.batch_norm(args["x"], args.get("running_mean"), args.get("running_var"),
                           args.get("weight"), args.get("bias"), training=options["training"],
                           momentum=options["momentum"], eps=options["eps"])
    put(name, "out", out)
    if options["training"] and "running_mean" in args:
        put(name, "running_mean_out", args["running_mean"])
        put(name, "running_var_out", args["running_var"])
    g = randn(*out.shape).to(out.dtype).requires_grad_(second)
    if key == "train_nonfinite":
        g.detach()[3, 2] = INF
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


def main():
    torch.set_num_threads(16)
    for key, (_, _, dtypes, _, _) in CASES.items():
        for dtype_name in dtypes:
            generate(key, dtype_name)
    OUT.parent.mkdir(exist_ok=True)
    save_file(data, str(OUT), metadata=reference.metadata())
    print(f"wrote {len(data)} tensors to {OUT}")


if __name__ == "__main__":
    main()
