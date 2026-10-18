"""Generates tests/golden/m2_training.safetensors: ten steps of training a small MLP with SGD and
AdamW, with the loss, the gradients, the parameters and the optimizer state of every step.
tests/m2_training.rs reads the file; tools/golden/check_m2_training.py checks it against a fresh
run.

The model is Linear(8→32)–ReLU–Linear(32→4) with the attributes fc1, act and fc2, trained on
cross_entropy (default options) with batches of 16. Case "data" stores, in F64,
- the initial state dict "init.<key>" (PyTorch's default initialization, drawn in F32);
- the inputs "x<t>" ([16, 8]) and the targets "target<t>" ([16], I64) of each step t in
  0..STEPS. The targets are the argmax of a fixed random linear map of the inputs, so that the
  model can learn them.

Case "<optimizer>_<dtype>", optimizer one of sgd, adamw and dtype f64 or f32, trains a model
that starts from the data rounded to the dtype. It stores
- the options as 0-D F64 tensors "opt.<name>": lr, momentum and weight_decay (sgd, one group of
  all parameters), or lr, weight_decay of the group of weights and bias_weight_decay of the
  group of biases (adamw, default betas and eps);
- for each step t, which runs zero_grad, the forward pass, backward and step: the loss
  "loss<t>", the gradients "grad<t>.<key>", and after the step the parameters "after<t>.<key>"
  and their state "after<t>.<key>.<state>": momentum_buffer (sgd), or step, exp_avg and
  exp_avg_sq (adamw).

The generator checks that every input of the ReLU keeps a distance of MARGIN from 0 in every
run, so that the gradients don't depend on how the kink is handled.

Usage: python tools/golden/gen_m2_training.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_training.safetensors"
STEPS = 10
SEED = 890
# The smallest distance of a ReLU input from 0. F32 runs differ from F64 runs by less than 2e-7
# (check_m2_training.py prints it); 3000 seeds have no run above 1e-3.
MARGIN = 5e-4
DTYPES = {"f64": torch.float64, "f32": torch.float32}
OPTIONS = {
    "sgd": dict(lr=0.1, momentum=0.9, weight_decay=0.01),
    "adamw": dict(lr=0.01, weight_decay=0.1, bias_weight_decay=0.0),
}


class Mlp(nn.Module):
    """Matches `Mlp` of tests/m2_training.rs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def make_data():
    """The initial state dict, the inputs and the targets, in F64 (targets in I64)."""
    torch.manual_seed(SEED)
    init = {k: v.detach().double() for k, v in Mlp().state_dict().items()}
    xs = [torch.randn(16, 8, dtype=torch.float64) for _ in range(STEPS)]
    teacher = torch.randn(4, 8, dtype=torch.float64)
    targets = [(x @ teacher.T).argmax(1) for x in xs]
    return init, xs, targets


def train(kind, dtype, init, xs, targets):
    """Trains a model of `dtype` with the optimizer `kind`. Returns the smallest distance of a
    ReLU input from 0 and the tensors of the case."""
    out = {}

    def put(name, t):
        # Parameters and state are updated in place: copy them now.
        assert name not in out, name
        out[name] = t.detach().contiguous().clone()

    model = Mlp().to(dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in init.items()})
    o = OPTIONS[kind]
    if kind == "sgd":
        opt = torch.optim.SGD(model.parameters(), **o)
    else:
        weights = [model.fc1.weight, model.fc2.weight]
        biases = [model.fc1.bias, model.fc2.bias]
        opt = torch.optim.AdamW(
            [{"params": weights}, {"params": biases, "weight_decay": o["bias_weight_decay"]}],
            lr=o["lr"],
            weight_decay=o["weight_decay"],
        )
    for n, v in o.items():
        put(f"opt.{n}", torch.tensor(float(v), dtype=torch.float64))
    params = dict(model.named_parameters())
    assert list(params) == list(init)
    margin = float("inf")
    for t, (x, target) in enumerate(zip(xs, targets)):
        opt.zero_grad()
        assert all(p.grad is None for p in params.values())
        hidden = model.fc1(x.to(dtype))
        margin = min(margin, hidden.detach().abs().min().item())
        loss = F.cross_entropy(model.fc2(model.act(hidden)), target)
        loss.backward()
        put(f"loss{t}", loss)
        for k, p in params.items():
            put(f"grad{t}.{k}", p.grad)
        opt.step()
        for k, p in params.items():
            put(f"after{t}.{k}", p)
            for s, v in opt.state[p].items():
                assert v.grad_fn is None and not v.requires_grad
                put(f"after{t}.{k}.{s}", v)
    return margin, out


def generate():
    """All tensors of the file, keyed "<case>/<name>", and the margin of each case."""
    init, xs, targets = make_data()
    data = {f"data/init.{k}": v.contiguous() for k, v in init.items()}
    for t, (x, target) in enumerate(zip(xs, targets)):
        data[f"data/x{t}"] = x
        data[f"data/target{t}"] = target
    margins = {}
    for kind in OPTIONS:
        for tag, dtype in DTYPES.items():
            case = f"{kind}_{tag}"
            margins[case], tensors = train(kind, dtype, init, xs, targets)
            data.update({f"{case}/{k}": v for k, v in tensors.items()})
    return data, margins


if __name__ == "__main__":
    reference.check()
    data, margins = generate()
    for case, margin in margins.items():
        assert margin >= MARGIN, f"{case}: a ReLU input is {margin} from 0; pick another seed"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(data, str(OUT), metadata=reference.metadata())
    print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")
