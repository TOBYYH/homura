"""Generates tests/golden/m2_loss.safetensors: the losses of M2 step 3b and F.logsigmoid, with
PyTorch's outputs and gradients. tests/nn_loss.rs reads the file.

Case "<function>_<dtype>.<name>", function one of mse, nll, ce, bce, logsigmoid, stores the
input "x", the "target" (bce, ce, mse, nll), "weight" and "pos_weight" when given, the options as
0-D tensors "opt.reduction" (0 none, 1 mean, 2 sum), "opt.ignore_index" (nll, ce) and
"opt.label_smoothing" (ce), the upstream gradient "g", the output "out", and "grad_<input>" for
each input that requires grad, from torch.autograd.grad(out, inputs, g); the inputs that
require grad are exactly those with a "grad_" entry. A ce case whose target has the shape of x
has probability targets. Cases of SECOND also store "gg_<input>" and the second derivatives
"grad2_<variable>": the gradients of sum(grad_<input> * gg_<input>), with the first gradients
computed with create_graph and "g" as a variable too; a variable whose second derivative is
None has no entry.

Usage: python tools/golden/gen_m2_loss.py (needs torch and safetensors).
"""

from pathlib import Path

import warnings

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_loss.safetensors"

reference.check()
# mse_loss warns about the broadcast cases.
warnings.simplefilter("ignore")
torch.manual_seed(0)
data = {}
REDUCTIONS = {"none": 0, "mean": 1, "sum": 2}
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


def positive(*shape):
    return torch.rand(shape, dtype=torch.float64) + 0.5


def probabilities(*shape):
    return randn(*shape).softmax(1 if len(shape) > 1 else 0)


def classes(*shape, n, ignore=()):
    """Random classes below n, with the given flat positions set to -100."""
    t = torch.randint(n, shape)
    for i in ignore:
        t.view(-1)[i] = -100
    return t


# Each case: function, inputs {name: f64 tensor or integer tensor}, options, differentiable
# inputs. Floating point inputs are converted to the case dtype, except names listed in "keep".
CASES = {}


def case(function, name, inputs, grads, reduction="mean", **options):
    CASES[f"{function}.{name}"] = (function, inputs, grads, reduction, options)


# mse_loss
case("mse", "mean", {"x": randn(3, 4), "target": randn(3, 4)}, ["x", "target"])
case("mse", "sum", {"x": randn(3, 4), "target": randn(3, 4)}, ["x", "target"], "sum")
case("mse", "none", {"x": randn(3, 4), "target": randn(3, 4)}, ["x", "target"], "none")
case("mse", "broadcast", {"x": randn(3, 1), "target": randn(1, 4)}, ["x", "target"])
case("mse", "broadcast_none", {"x": randn(4), "target": randn(2, 1)}, ["x", "target"], "none")
case("mse", "target_constant", {"x": randn(5), "target": randn(5)}, ["x"])
case("mse", "zero_dim", {"x": randn(), "target": randn()}, ["x", "target"], "none")
case("mse", "empty", {"x": randn(0, 3), "target": randn(0, 3)}, ["x", "target"])
case("mse", "empty_sum", {"x": randn(0, 3), "target": randn(0, 3)}, ["x", "target"], "sum")
case("mse", "integer_target", {"x": randn(4), "target": torch.tensor([1, -2, 0, 3])}, ["x"])
# PyTorch's gradient scales by 2 / n before multiplying by the difference: finite here although
# the squares overflow.
case("mse", "large", {"x": torch.tensor([300.0, -300.0, 1e200]),
                      "target": torch.tensor([-300.0, 300.0, -1e200])}, ["x", "target"])

# nll_loss on log-probabilities
case("nll", "mean", {"x": randn(5, 4).log_softmax(1), "target": classes(5, n=4, ignore=[2])},
     ["x"])
case("nll", "sum", {"x": randn(5, 4).log_softmax(1), "target": classes(5, n=4, ignore=[2])},
     ["x"], "sum")
case("nll", "none", {"x": randn(5, 4).log_softmax(1), "target": classes(5, n=4, ignore=[2])},
     ["x"], "none")
case("nll", "weight", {"x": randn(5, 4).log_softmax(1), "target": classes(5, n=4, ignore=[1]),
                       "weight": positive(4)}, ["x"])
case("nll", "weight_sum", {"x": randn(5, 4).log_softmax(1), "target": classes(5, n=4),
                           "weight": positive(4)}, ["x"], "sum")
case("nll", "weight_none", {"x": randn(5, 4).log_softmax(1),
                            "target": classes(5, n=4, ignore=[0]), "weight": positive(4)},
     ["x"], "none")
case("nll", "ignore_class", {"x": randn(6, 3).log_softmax(1),
                             "target": torch.tensor([0, 1, 2, 1, 1, 0])}, ["x"], ignore_index=1)
case("nll", "ignore_beyond_classes", {"x": randn(3, 3).log_softmax(1),
                                      "target": torch.tensor([0, 7, 2])}, ["x"], ignore_index=7)
case("nll", "byte_target", {"x": randn(4, 3).log_softmax(1),
                            "target": torch.tensor([2, 0, 1, 1], dtype=torch.uint8),
                            "weight": positive(3)}, ["x"])
case("nll", "single", {"x": randn(4).log_softmax(0), "target": torch.tensor(2)}, ["x"])
# A [1] target is accepted forward, but PyTorch's backward rejects it (check_m2_loss.py).
case("nll", "single_none", {"x": randn(4).log_softmax(0), "target": torch.tensor(1),
                            "weight": positive(4)}, ["x"], "none")
case("nll", "single_ignored", {"x": randn(4).log_softmax(0), "target": torch.tensor(-100)},
     ["x"], "sum")
case("nll", "3d", {"x": randn(2, 4, 3).log_softmax(1),
                   "target": classes(2, 3, n=4, ignore=[4]), "weight": positive(4)}, ["x"])
case("nll", "3d_none", {"x": randn(2, 4, 3).log_softmax(1), "target": classes(2, 3, n=4)},
     ["x"], "none")
case("nll", "4d", {"x": randn(2, 3, 2, 2).log_softmax(1),
                   "target": classes(2, 2, 2, n=3, ignore=[3]), "weight": positive(3)}, ["x"])
case("nll", "4d_none", {"x": randn(2, 3, 2, 2).log_softmax(1), "target": classes(2, 2, 2, n=3)},
     ["x"], "none")
case("nll", "5d_sum", {"x": randn(2, 3, 2, 1, 2).log_softmax(1),
                       "target": classes(2, 2, 1, 2, n=3)}, ["x"], "sum")
case("nll", "all_ignored", {"x": randn(3, 4).log_softmax(1), "target": torch.full((3,), -100)},
     ["x"])
case("nll", "all_ignored_weight", {"x": randn(3, 4).log_softmax(1),
                                   "target": torch.full((3,), -100), "weight": positive(4)},
     ["x"])
case("nll", "empty", {"x": randn(0, 4), "target": torch.zeros(0, dtype=torch.long)}, ["x"])
case("nll", "empty_none", {"x": randn(0, 4), "target": torch.zeros(0, dtype=torch.long)},
     ["x"], "none")
# An ignored position is never read: -inf there doesn't turn the loss or the gradient into NaN.
case("nll", "infinite_ignored", {"x": torch.tensor([[-INF, 0.0], [0.5, -1.0]]),
                                 "target": torch.tensor([1, 0])}, ["x"], "sum", ignore_index=0)

# cross_entropy with class targets
case("ce", "mean", {"x": randn(5, 4), "target": classes(5, n=4)}, ["x"])
case("ce", "weight_ignore", {"x": randn(5, 4), "target": classes(5, n=4, ignore=[3]),
                             "weight": positive(4)}, ["x"])
case("ce", "none", {"x": randn(5, 4), "target": classes(5, n=4, ignore=[0])}, ["x"], "none")
case("ce", "smoothing", {"x": randn(5, 4), "target": classes(5, n=4)}, ["x"],
     label_smoothing=0.1)
case("ce", "smoothing_weight_ignore", {"x": randn(6, 4), "target": classes(6, n=4, ignore=[2]),
                                       "weight": positive(4)}, ["x"], label_smoothing=0.2)
case("ce", "smoothing_sum", {"x": randn(5, 4), "target": classes(5, n=4, ignore=[1]),
                             "weight": positive(4)}, ["x"], "sum", label_smoothing=0.3)
case("ce", "smoothing_none", {"x": randn(5, 4), "target": classes(5, n=4, ignore=[4])}, ["x"],
     "none", label_smoothing=0.25)
case("ce", "smoothing_full", {"x": randn(3, 4), "target": classes(3, n=4)}, ["x"],
     label_smoothing=1.0)
case("ce", "smoothing_byte", {"x": randn(4, 3),
                              "target": torch.tensor([2, 0, 1, 1], dtype=torch.uint8)}, ["x"],
     label_smoothing=0.1)
case("ce", "smoothing_all_ignored", {"x": randn(3, 4), "target": torch.full((3,), -100),
                                     "weight": positive(4)}, ["x"], label_smoothing=0.2)
case("ce", "single", {"x": randn(4), "target": torch.tensor(1)}, ["x"])
case("ce", "single_smoothing", {"x": randn(4), "target": torch.tensor(3), "weight": positive(4)},
     ["x"], "none", label_smoothing=0.2)
case("ce", "3d_smoothing", {"x": randn(2, 4, 3), "target": classes(2, 3, n=4, ignore=[1]),
                            "weight": positive(4)}, ["x"], label_smoothing=0.1)
case("ce", "4d", {"x": randn(2, 3, 2, 2), "target": classes(2, 2, 2, n=3)}, ["x"])
case("ce", "empty", {"x": randn(0, 4), "target": torch.zeros(0, dtype=torch.long)}, ["x"])
case("ce", "negative_smoothing", {"x": randn(3, 4), "target": classes(3, n=4)}, ["x"],
     label_smoothing=-0.5)

# cross_entropy with probability targets: the weight and the target get gradients too
case("ce", "prob", {"x": randn(5, 4), "target": probabilities(5, 4)}, ["x", "target"])
case("ce", "prob_weight", {"x": randn(5, 4), "target": probabilities(5, 4),
                           "weight": positive(4)}, ["x", "target", "weight"])
case("ce", "prob_weight_sum", {"x": randn(5, 4), "target": probabilities(5, 4),
                               "weight": positive(4)}, ["x", "target", "weight"], "sum")
case("ce", "prob_weight_none", {"x": randn(5, 4), "target": probabilities(5, 4),
                                "weight": positive(4)}, ["x", "target", "weight"], "none")
case("ce", "prob_smoothing", {"x": randn(5, 4), "target": probabilities(5, 4)},
     ["x", "target"], label_smoothing=0.2)
case("ce", "prob_single", {"x": randn(4), "target": probabilities(4), "weight": positive(4)},
     ["x", "target", "weight"], "none")
case("ce", "prob_3d", {"x": randn(2, 4, 3), "target": probabilities(2, 4, 3),
                       "weight": positive(4)}, ["x", "target", "weight"])
case("ce", "prob_empty", {"x": randn(0, 4), "target": randn(0, 4), "weight": positive(4)},
     ["x", "target", "weight"])
case("ce", "prob_constant_target", {"x": randn(3, 4), "target": probabilities(3, 4)}, ["x"])

# binary_cross_entropy_with_logits
case("bce", "mean", {"x": randn(3, 4), "target": torch.rand(3, 4, dtype=torch.float64)},
     ["x", "target"])
case("bce", "sum", {"x": randn(3, 4), "target": torch.rand(3, 4, dtype=torch.float64)},
     ["x", "target"], "sum")
case("bce", "none", {"x": randn(3, 4), "target": torch.rand(3, 4, dtype=torch.float64)},
     ["x", "target"], "none")
case("bce", "weight", {"x": randn(3, 4), "target": torch.rand(3, 4, dtype=torch.float64),
                       "weight": positive(4)}, ["x", "target"])
case("bce", "weight_column", {"x": randn(3, 4), "target": torch.rand(3, 4, dtype=torch.float64),
                              "weight": positive(3, 1)}, ["x", "target"], "none")
case("bce", "pos_weight", {"x": randn(3, 4), "target": torch.rand(3, 4, dtype=torch.float64),
                           "pos_weight": positive(4)}, ["x", "target"])
case("bce", "pos_weight_weight", {"x": randn(3, 4),
                                  "target": torch.rand(3, 4, dtype=torch.float64),
                                  "weight": positive(3, 4), "pos_weight": positive(3, 1)},
     ["x", "target"], "sum")
case("bce", "pos_weight_none", {"x": randn(2, 3), "target": torch.rand(2, 3, dtype=torch.float64),
                                "pos_weight": positive(3)}, ["x", "target"], "none")
case("bce", "large_logits", {"x": torch.tensor([40.0, -40.0, 800.0, -800.0, 0.0, 15.0]),
                             "target": torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])},
     ["x", "target"], "none")
case("bce", "large_logits_pos_weight", {"x": torch.tensor([40.0, -40.0, 800.0, -800.0, 0.0]),
                                        "target": torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5]),
                                        "pos_weight": torch.tensor([2.0, 3.0, 0.5, 1.5, 4.0])},
     ["x", "target"], "none")
case("bce", "zero_dim", {"x": randn(), "target": torch.tensor(0.25)}, ["x", "target"])
case("bce", "empty", {"x": randn(0, 2), "target": randn(0, 2)}, ["x", "target"])
case("bce", "target_constant", {"x": randn(4), "target": torch.tensor([0.0, 1.0, 1.0, 0.0])},
     ["x"])

# F.logsigmoid
case("logsigmoid", "values", {"x": torch.tensor([0.0, 1e-30, -1e-30, 0.5, -0.5, 3.0, -3.0, 40.0,
                                                  -40.0, 800.0, -800.0, INF, -INF, NAN])},
     ["x"], "none")
case("logsigmoid", "random", {"x": randn(3, 4) * 4}, ["x"], "none")

FUNCTIONS = ["mse", "nll", "ce", "bce", "logsigmoid"]
HALF = ["mse.mean", "mse.large", "nll.weight", "nll.3d", "ce.smoothing_weight_ignore",
        "ce.prob_weight", "bce.pos_weight_weight", "bce.large_logits", "logsigmoid.values",
        "logsigmoid.random"]
SECOND = ["mse.broadcast", "mse.none", "nll.weight", "nll.4d", "ce.weight_ignore",
          "ce.smoothing_weight_ignore", "ce.prob_weight", "ce.prob_smoothing", "bce.pos_weight_weight",
          "bce.weight", "bce.none", "logsigmoid.random"]


def compute(function, args, reduction, options):
    if function == "mse":
        return F.mse_loss(args["x"], args["target"], reduction=reduction)
    if function == "nll":
        return F.nll_loss(args["x"], args["target"], weight=args.get("weight"),
                          reduction=reduction, **options)
    if function == "ce":
        return F.cross_entropy(args["x"], args["target"], weight=args.get("weight"),
                               reduction=reduction, **options)
    if function == "bce":
        return F.binary_cross_entropy_with_logits(args["x"], args["target"],
                                                  weight=args.get("weight"),
                                                  pos_weight=args.get("pos_weight"),
                                                  reduction=reduction)
    assert function == "logsigmoid"
    return F.logsigmoid(args["x"])


def generate(key, dtype_name):
    function, inputs, grads, reduction, options = CASES[key]
    dtype = DTYPES[dtype_name]
    name = f"{function}_{dtype_name}.{key.split('.', 1)[1]}"
    args = {}
    for k, v in inputs.items():
        v = v.to(dtype) if v.is_floating_point() else v.clone()
        v.requires_grad_(k in grads)
        args[k] = v
        put(name, k, v)
    put(name, "opt.reduction", torch.tensor(REDUCTIONS[reduction]))
    if function in ("nll", "ce"):
        put(name, "opt.ignore_index", torch.tensor(options.get("ignore_index", -100)))
    if function == "ce":
        put(name, "opt.label_smoothing",
            torch.tensor(options.get("label_smoothing", 0.0), dtype=torch.float64))
    out = compute(function, args, reduction, options)
    put(name, "out", out)
    second = dtype_name == "f64" and key in SECOND
    g = randn(*out.shape).to(out.dtype).requires_grad_(second)
    put(name, "g", g)
    variables = [args[k] for k in grads]
    first = torch.autograd.grad(out, variables, g, create_graph=second)
    for k, t in zip(grads, first):
        put(name, f"grad_{k}", t)
    if second:
        gg = [randn(*t.shape) for t in first]
        total = sum((t * s).sum() for t, s in zip(first, gg))
        for k, s in zip(grads, gg):
            put(name, f"gg_{k}", s)
        second_grads = torch.autograd.grad(total, variables + [g], allow_unused=True)
        for k, t in zip(grads + ["g"], second_grads):
            if t is not None:
                put(name, f"grad2_{k}", t)


for key in CASES:
    for dtype_name in ["f64", "f32"] + (["f16", "bf16"] if key in HALF else []):
        generate(key, dtype_name)

OUT.parent.mkdir(exist_ok=True)
save_file(data, str(OUT), metadata=reference.metadata())
print(f"wrote {len(data)} tensors to {OUT}")
