"""Checks tests/golden/m2_cnn_training.safetensors (M2 step 9b) against a fresh run of
tools/golden/gen_m2_cnn_training.py: the same tensors bit for bit, the margins of every run, the
fall of the mean loss, and that every step changes what it should. Prints the margins and how far
each F32 run is from the F64 run of the same optimizer, the scale that the F32 tolerances of
tests/m2_cnn_training.rs are compared with.

Usage: python tools/golden/check_m2_cnn_training.py (needs torch and safetensors).
"""

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import gen_m2_cnn_training as gen
import reference

reference.check()
threads = torch.get_num_threads()
data, margins = gen.generate()
# The generator restores the number of threads it changes.
assert torch.get_num_threads() == threads

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
    print(f"{case}: smallest margin {margin:.3e} (MARGIN {gen.MARGIN:.0e})")
for case, fall in gen.learned(data).items():
    assert fall >= gen.LEARNED, (case, fall)
    print(f"{case}: mean loss on the training batches falls by {fall:.4f}")

# Every step changes each parameter and state tensor that is trained, not necessarily each
# element: the stored values are copies, not the tensors that later steps update in place. The
# frozen bias keeps its initial value and gets neither gradients nor state; the count rises by
# one per step.
for case in margins:
    dtype = gen.DTYPES[case.split("_")[1]]
    init = data[f"data/init.{gen.FROZEN}"].to(dtype)
    for t in range(gen.STEPS):
        assert torch.equal(data[f"{case}/after{t}.{gen.FROZEN}"], init), (case, t)
        assert f"{case}/grad{t}.{gen.FROZEN}" not in data
        assert not any(k.startswith(f"{case}/after{t}.{gen.FROZEN}.") for k in data)
        assert data[f"{case}/after{t}.1.num_batches_tracked"].item() == t + 1
        if t == 0:
            continue
        for k, v in data.items():
            if (k.startswith(f"{case}/after{t}.") and not k.endswith((".step", "num_batches_tracked"))
                    and k != f"{case}/after{t}.{gen.FROZEN}"):
                before = data[k.replace(f"/after{t}.", f"/after{t - 1}.")]
                assert not torch.equal(v, before), k

# The F32 runs select the same ReLU inputs and pooling elements as the F64 runs.
for kind in gen.OPTIONS:
    for t in [*map(str, range(gen.STEPS)), "_eval"]:
        low, high = data[f"{kind}_f32/pre{t}"], data[f"{kind}_f64/pre{t}"]
        assert torch.equal(low > 0, high > 0), (kind, t)
        assert torch.equal(data[f"{kind}_f32/indices{t}"], data[f"{kind}_f64/indices{t}"]), (kind, t)

# Distance of each F32 run from the F64 run, per kind of tensor: the largest absolute
# difference, and the largest magnitude of the F64 values.
for kind in gen.OPTIONS:
    worst = {}
    prefix64, prefix32 = f"{kind}_f64/", f"{kind}_f32/"
    for k, v64 in data.items():
        if (not k.startswith(prefix64) or k.startswith((prefix64 + "opt.", prefix64 + "indices"))
                or k.endswith("num_batches_tracked")):
            continue
        name = k[len(prefix64):]
        v32 = data[prefix32 + name]
        parts = name.split(".")
        kind_of = parts[0].removesuffix("_eval").rstrip("0123456789")
        if len(parts) == 4:
            kind_of += "." + parts[3]
        diff = (v32.double() - v64.double()).abs().max().item()
        mag = v64.double().abs().max().item()
        d, m = worst.get(kind_of, (0.0, 0.0))
        worst[kind_of] = (max(d, diff), max(m, mag))
    for kind_of, (d, m) in sorted(worst.items()):
        print(f"{kind} f32 - f64 {kind_of:>22}: max |diff| {d:.3e}, max |f64| {m:.3e}")

print("ok")
