"""Generates tests/golden/m3_safetensors/: safetensors files written by Python, which
tests/io_safetensors.rs reads (M3 step 2a).

Every tensor holds values of a formula that the Rust test computes too: the flat index i maps to
k = (i * 37 + 5) % 251; BOOL is i % 3 == 0, U8 is k, other integers are k - 125 and floats
(k - 125) / 8, all exact in every dtype.

- dtypes.safetensors: `save_file` with `__metadata__` (format "pt" and the reference build), one
  [2, 3] tensor per dtype named after it ("bool", "u8", ..., "f64"), a zero-dimensional "scalar"
  (F32), and the empty "empty_i64" ([0, 3]) and "empty_bool" ([0]).
- unaligned.safetensors: written by hand with a compact header whose length is not a multiple of
  8 and no `__metadata__`: "x" (F32, [2]) and "y" (I16, [3]). Python reads it.
- sharded/: a checkpoint in the Hugging Face layout, as `save_pretrained` writes it: shards
  model-00001-of-00002.safetensors ("a.weight" F32 [4, 4], "a.bias" F32 [4]) and
  model-00002-of-00002.safetensors ("b.weight" F16 [2, 4], "b.count" I64 []), both with
  `__metadata__` {"format": "pt"}, and model.safetensors.index.json (`metadata.total_size` and a
  `weight_map` sorted by key, indented as `json.dump(..., indent=2, sort_keys=True)` writes it).

Usage: python tools/golden/gen_m3_safetensors.py (needs torch and safetensors).
"""

import json
import struct
from pathlib import Path

import torch
from safetensors.torch import load, load_file, save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m3_safetensors"

reference.check()

DTYPES = {
    "bool": torch.bool, "u8": torch.uint8, "i8": torch.int8, "i16": torch.int16,
    "i32": torch.int32, "i64": torch.int64, "f16": torch.float16, "bf16": torch.bfloat16,
    "f32": torch.float32, "f64": torch.float64,
}


def values(shape, dtype):
    """The formula's values for a tensor of `shape` and `dtype`."""
    i = torch.arange(int(torch.Size(shape).numel()), dtype=torch.int64)
    k = (i * 37 + 5) % 251
    if dtype == torch.bool:
        v = i % 3 == 0
    elif dtype == torch.uint8:
        v = k.to(dtype)
    elif dtype.is_floating_point:
        v = ((k - 125).to(torch.float64) / 8).to(dtype)
        assert torch.equal(v.to(torch.float64), (k - 125).to(torch.float64) / 8)
    else:
        v = (k - 125).to(dtype)
    return v.reshape(shape)


def dtypes():
    tensors = {name: values([2, 3], dtype) for name, dtype in DTYPES.items()}
    tensors["scalar"] = values([], torch.float32)
    tensors["empty_i64"] = values([0, 3], torch.int64)
    tensors["empty_bool"] = values([0], torch.bool)
    path = OUT / "dtypes.safetensors"
    save_file(tensors, path, metadata={"format": "pt", **reference.metadata()})
    back = load_file(path)
    assert back.keys() == tensors.keys()
    for name, t in tensors.items():
        assert back[name].dtype == t.dtype and torch.equal(back[name], t), name


def unaligned():
    header = {
        "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "y": {"dtype": "I16", "shape": [3], "data_offsets": [8, 14]},
    }
    text = json.dumps(header, separators=(",", ":")).encode()
    assert len(text) % 8 != 0
    x, y = values([2], torch.float32), values([3], torch.int16)
    data = x.numpy().astype("<f4").tobytes() + y.numpy().astype("<i2").tobytes()
    raw = struct.pack("<Q", len(text)) + text + data
    (OUT / "unaligned.safetensors").write_bytes(raw)
    back = load(raw)
    assert torch.equal(back["x"], x) and torch.equal(back["y"], y)


def sharded():
    shards = [
        {"a.weight": values([4, 4], torch.float32), "a.bias": values([4], torch.float32)},
        {"b.weight": values([2, 4], torch.float16), "b.count": values([], torch.int64)},
    ]
    directory = OUT / "sharded"
    directory.mkdir(exist_ok=True)
    weight_map, total_size = {}, 0
    for n, shard in enumerate(shards, 1):
        file = f"model-{n:05}-of-{len(shards):05}.safetensors"
        save_file(shard, directory / file, metadata={"format": "pt"})
        for name, t in shard.items():
            weight_map[name] = file
            total_size += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(directory / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    # Loading as transformers does: every shard the index names, merged.
    with open(directory / "model.safetensors.index.json") as f:
        names = sorted(set(json.load(f)["weight_map"].values()))
    merged = {}
    for file in names:
        merged.update(load_file(directory / file))
    assert merged.keys() == {k for shard in shards for k in shard}


OUT.mkdir(exist_ok=True)
dtypes()
unaligned()
sharded()
print(f"wrote {OUT}")
