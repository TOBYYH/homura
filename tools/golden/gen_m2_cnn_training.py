"""Generates tests/golden/m2_cnn_training.safetensors: ten steps of training the small CNN of M2
step 9 with SGD and AdamW, with the loss, the gradients, the parameters, the optimizer state and
the batch norm buffers of every step, and an evaluation pass after the last one.
tests/m2_cnn_training.rs reads the file; tools/golden/check_m2_cnn_training.py checks it against
a fresh run.

The model is the nn.Sequential Conv2d(3→8, 3, padding=1)–BatchNorm2d(8)–ReLU–MaxPool2d(2)–
Conv2d(8→16, 3, padding=1)–AdaptiveAvgPool2d(1)–Flatten–Linear(16→4), trained on cross_entropy
(default options) with batches of 4 inputs of 3×8×8. Case "data" stores, in F64,
- the initial state dict "init.<key>" (PyTorch's default initialization, drawn in F32; the batch
  norm count in I64);
- the inputs "x<t>" ([4, 3, 8, 8]) and the targets "target<t>" ([4], I64) of each step t in
  0..STEPS, and the input "x_eval" ([4, 3, 8, 8]) of the evaluation pass. The targets are the
  classes of a teacher, a CNN of the same architecture with other random weights in evaluation
  mode, whose logits are centered on their mean over the training inputs (otherwise one class
  wins for every input), so that the model can learn them.

Case "<optimizer>_<dtype>", optimizer one of sgd, adamw and dtype f64 or f32, trains a model
that starts from the data rounded to the dtype, in training mode, with the bias of the first
convolution frozen (requires_grad=False; FROZEN): the batch norm after it subtracts the mean of
each channel, so the gradient of that bias is zero and its computed value only rounding noise,
which Adam would scale up to updates. The optimizers still hold it, and skip it as it has no
gradient. It stores
- the options as 0-D F64 tensors "opt.<name>": lr, momentum and weight_decay (sgd, one group of
  all parameters), or lr, weight_decay of the group of convolution and linear weights and
  other_weight_decay of the group of the other parameters (adamw, default betas and eps);
- for each step t, which runs zero_grad, the forward pass, backward and step: the batch norm
  output "pre<t>" (the ReLU input), the indices of the max pooling "indices<t>" (I64), the loss
  "loss<t>", the gradients "grad<t>.<key>" (none for the frozen bias), and after the step the parameters "after<t>.<key>",
  their state "after<t>.<key>.<state>" (momentum_buffer for sgd; step, exp_avg and exp_avg_sq
  for adamw) and the buffers "after<t>.<key>" of the batch norm;
- after the last step, in evaluation mode: "pre_eval", "indices_eval" and the output "out_eval"
  for "x_eval".

With batches of 4, the loss of a step depends more on its batch than on the training; the
model learns the targets slowly with these learning rates: LEARNED checks that the mean loss on
the ten training batches falls. The generator checks every run, all steps and the evaluation
pass, for near ties: each ReLU
input is at least MARGIN from 0, and in each pooling window whose maximum is positive the
maximum is at least MARGIN above the other elements. So results within MARGIN / 2 of PyTorch's
select the same elements. (With the ReLU margin, a window whose maximum is 0 has only negative
inputs, and its gradient is zero whichever element is selected.) The runs use one thread, so
that the sums of PyTorch's convolutions are taken in a fixed order.

Usage: python tools/golden/gen_m2_cnn_training.py [first seed] (needs torch and safetensors).
Without an argument it writes the file for SEED; with one it searches for the first seed from
there whose runs keep the margins and learn, and prints it.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_cnn_training.safetensors"
STEPS = 10
SEED = 0
# The smallest distance of a ReLU input from 0, and of a pooling maximum from the rest of its
# window. homura's F32 batch norm outputs are within MARGIN / 10 of PyTorch's over the whole
# trajectory (tests/m2_cnn_training.rs; measure_m2_tolerances.py reports it).
MARGIN = 2e-5
DTYPES = {"f64": torch.float64, "f32": torch.float32}
OPTIONS = {
    "sgd": dict(lr=0.01, momentum=0.9, weight_decay=0.01),
    "adamw": dict(lr=0.002, weight_decay=0.1, other_weight_decay=0.0),
}
WEIGHTS = ["0.weight", "4.weight", "7.weight"]
FROZEN = "0.bias"
# The least fall of the mean loss on the training batches.
LEARNED = 0.01


def cnn():
    """Matches `cnn` of tests/m2_cnn_training.rs."""
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


def make_data(seed):
    """The initial state dict, the inputs, the targets and the evaluation input, in F64 (the
    targets and the count in I64)."""
    torch.manual_seed(seed)
    init = {k: v.detach().double() if v.is_floating_point() else v.detach().clone()
            for k, v in cnn().state_dict().items()}
    xs = [torch.randn(4, 3, 8, 8, dtype=torch.float64) for _ in range(STEPS)]
    teacher = cnn().double().eval()
    with torch.no_grad():
        logits = [teacher(x) for x in xs]
        center = torch.cat(logits).mean(0)
    targets = [(t - center).argmax(1) for t in logits]
    x_eval = torch.randn(4, 3, 8, 8, dtype=torch.float64)
    return init, xs, targets, x_eval


def forward(model, x):
    """The output of model for x, layer by layer like nn.Sequential, with the batch norm output
    and the pooling indices, and the margins of this pass."""
    pre = model[1](model[0](x))
    relu = model[2](pre)
    pooled, indices = F.max_pool2d(relu, 2, return_indices=True)
    assert torch.equal(pooled, model[3](relu))
    out = pooled
    for layer in list(model)[4:]:
        out = layer(out)
    with torch.no_grad():
        relu_margin = pre.abs().min().item()
        windows = relu.unfold(2, 2, 2).unfold(3, 2, 2).flatten(-2)
        top = windows.topk(2, dim=-1).values
        positive = top[..., 0] > 0
        gaps = (top[..., 0] - top[..., 1])[positive]
        pool_margin = gaps.min().item() if gaps.numel() else float("inf")
    return out, pre, indices, min(relu_margin, pool_margin)


def train(kind, dtype, init, xs, targets, x_eval):
    """Trains a model of `dtype` with the optimizer `kind`. Returns the smallest margin of the
    run and the tensors of the case."""
    out = {}

    def put(name, t):
        # Parameters, buffers and state are updated in place: copy them now.
        assert name not in out, name
        out[name] = t.detach().contiguous().clone()

    model = cnn().to(dtype)
    model.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in init.items()})
    params = dict(model.named_parameters())
    params[FROZEN].requires_grad_(False)
    o = OPTIONS[kind]
    if kind == "sgd":
        opt = torch.optim.SGD(model.parameters(), **o)
    else:
        weights = [p for k, p in params.items() if k in WEIGHTS]
        others = [p for k, p in params.items() if k not in WEIGHTS]
        opt = torch.optim.AdamW(
            [{"params": weights}, {"params": others, "weight_decay": o["other_weight_decay"]}],
            lr=o["lr"],
            weight_decay=o["weight_decay"],
        )
    for n, v in o.items():
        put(f"opt.{n}", torch.tensor(float(v), dtype=torch.float64))
    margin = float("inf")
    for t, (x, target) in enumerate(zip(xs, targets)):
        opt.zero_grad()
        assert all(p.grad is None for p in params.values())
        logits, pre, indices, m = forward(model, x.to(dtype))
        margin = min(margin, m)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        put(f"pre{t}", pre)
        put(f"indices{t}", indices)
        put(f"loss{t}", loss)
        assert params[FROZEN].grad is None
        for k, p in params.items():
            if k != FROZEN:
                put(f"grad{t}.{k}", p.grad)
        opt.step()
        for k, p in params.items():
            put(f"after{t}.{k}", p)
            for s, v in opt.state[p].items():
                assert v.grad_fn is None and not v.requires_grad
                put(f"after{t}.{k}.{s}", v)
        for k, b in model.named_buffers():
            put(f"after{t}.{k}", b)
    model.eval()
    with torch.no_grad():
        logits, pre, indices, m = forward(model, x_eval.to(dtype))
    margin = min(margin, m)
    put("pre_eval", pre)
    put("indices_eval", indices)
    put("out_eval", logits)
    return margin, out


def state_after(data, case, t):
    """The state dict of the model of `case` after step t."""
    keys = [k[len("data/init."):] for k in data if k.startswith("data/init.")]
    return {k: data[f"{case}/after{t}.{k}"] for k in keys}


def mean_loss(state, dtype, xs, targets):
    """The mean loss on the training batches of a new model with `state`, in training mode (with
    the statistics of each batch)."""
    model = cnn().to(dtype)
    model.load_state_dict(state)
    with torch.no_grad():
        return sum(F.cross_entropy(model(x.to(dtype)), t).item() for x, t in zip(xs, targets)) / len(xs)


def learned(data):
    """{case: the fall of the mean loss on the training batches over the ten steps}."""
    xs = [data[f"data/x{t}"] for t in range(STEPS)]
    targets = [data[f"data/target{t}"] for t in range(STEPS)]
    init = {k[len("data/init."):]: v for k, v in data.items() if k.startswith("data/init.")}
    out = {}
    for kind in OPTIONS:
        for tag, dtype in DTYPES.items():
            start = {k: v.to(dtype) if v.is_floating_point() else v for k, v in init.items()}
            case = f"{kind}_{tag}"
            out[case] = (mean_loss(start, dtype, xs, targets)
                         - mean_loss(state_after(data, case, STEPS - 1), dtype, xs, targets))
    return out


def acceptable(data, margins):
    return min(margins.values()) >= MARGIN and min(learned(data).values()) >= LEARNED


def generate(seed=SEED):
    """All tensors of the file, keyed "<case>/<name>", and the margin of each case. PyTorch runs
    with one thread meanwhile."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        init, xs, targets, x_eval = make_data(seed)
        data = {f"data/init.{k}": v.contiguous() for k, v in init.items()}
        for t, (x, target) in enumerate(zip(xs, targets)):
            data[f"data/x{t}"] = x
            data[f"data/target{t}"] = target
        data["data/x_eval"] = x_eval
        margins = {}
        for kind in OPTIONS:
            for tag, dtype in DTYPES.items():
                case = f"{kind}_{tag}"
                margins[case], tensors = train(kind, dtype, init, xs, targets, x_eval)
                data.update({f"{case}/{k}": v for k, v in tensors.items()})
        return data, margins
    finally:
        torch.set_num_threads(threads)


def main():
    reference.check()
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        while not acceptable(*generate(seed)):
            seed += 1
        print(f"seed {seed} keeps the margin {MARGIN}")
        return
    data, margins = generate()
    for case, margin in margins.items():
        assert margin >= MARGIN, f"{case}: a margin is {margin}; pick another seed"
    for case, fall in learned(data).items():
        assert fall >= LEARNED, f"{case}: the mean loss falls by {fall}; pick another seed"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(data, str(OUT), metadata=reference.metadata())
    print(f"wrote {len(data)} tensors to {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
