"""Checks tests/golden/m2_training.safetensors (M2 step 4b) against a fresh run of
tools/golden/gen_m2_training.py: the same tensors bit for bit, the ReLU margin of every run and
a falling loss. Prints how far each F32 run is from the F64 run of the same optimizer, the scale
that the F32 tolerances of tests/m2_training.rs are compared with.

Usage: python tools/golden/check_m2_training.py (needs torch and safetensors).
"""

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import gen_m2_training as gen
import reference

reference.check()
data, margins = gen.generate()

# The committed file is what the generator writes now, with the reference build's metadata.
stored = load_file(str(gen.OUT))
assert sorted(stored) == sorted(data), sorted(set(stored) ^ set(data))
BITS = {torch.float64: torch.int64, torch.float32: torch.int32, torch.int64: torch.int64}
for k, v in data.items():
    s = stored[k]
    assert s.dtype == v.dtype and s.shape == v.shape, k
    assert s.view(BITS[s.dtype]).equal(v.view(BITS[v.dtype])), k
    if v.is_floating_point():
        assert s.isfinite().all(), k
with safe_open(str(gen.OUT), "pt") as f:
    assert f.metadata() == reference.metadata()

for case, margin in margins.items():
    assert margin >= gen.MARGIN, (case, margin)
    print(f"{case}: smallest |ReLU input| {margin:.3e}")

# Every step changes each parameter and state tensor, not necessarily each element: the stored
# values are copies, not the tensors that later steps update in place.
for case in margins:
    for t in range(1, gen.STEPS):
        for k, v in data.items():
            if k.startswith(f"{case}/after{t}.") and not k.endswith(".step"):
                before = data[k.replace(f"/after{t}.", f"/after{t - 1}.")]
                assert not torch.equal(v, before), k

# The model learns the targets: the loss of the last steps is below the first one.
for case in margins:
    losses = [data[f"{case}/loss{t}"].item() for t in range(gen.STEPS)]
    assert max(losses[-2:]) < losses[0] - 0.2, (case, losses)
    print(f"{case}: loss " + " ".join(f"{x:.4f}" for x in losses))

# Distance of each F32 run from the F64 run, per kind of tensor: the largest absolute
# difference, and the largest magnitude of the F64 values.
for kind in gen.OPTIONS:
    worst = {}
    prefix64, prefix32 = f"{kind}_f64/", f"{kind}_f32/"
    for k, v64 in data.items():
        if not k.startswith(prefix64) or k.startswith(prefix64 + "opt."):
            continue
        name = k[len(prefix64):]
        v32 = data[prefix32 + name]
        kind_of = name.split(".")[0].rstrip("0123456789")
        if name.count(".") == 3:
            kind_of += "." + name.split(".")[-1]
        diff = (v32.double() - v64.double()).abs().max().item()
        mag = v64.double().abs().max().item()
        d, m = worst.get(kind_of, (0.0, 0.0))
        worst[kind_of] = (max(d, diff), max(m, mag))
    for kind_of, (d, m) in sorted(worst.items()):
        print(f"{kind} f32 - f64 {kind_of:>18}: max |diff| {d:.3e}, max |f64| {m:.3e}")

print("ok")
