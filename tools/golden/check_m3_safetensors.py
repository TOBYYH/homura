"""Check that Python reads the safetensors files homura writes (M3 step 2a); writes no fixtures.

`homura_files_for_python` in tests/io_safetensors.rs writes target/tmp/m3_safetensors/ (run
`cargo test --test io_safetensors` first):

- metadata.safetensors, from `io::save_safetensors_with_metadata`: one [2, 3] tensor per dtype
  named after it, "transposed" (F32 [3, 2], the transpose of the formula's [2, 3], saved
  contiguous), "scalar" (F64 []) and "empty" (I16 [0, 2]), in that order, with `__metadata__`
  {"format": "pt", "writer": "homura"}. The values follow the formula of gen_m3_safetensors.py.
- sharded/, from `io::save_sharded` with a 40-byte limit and the same metadata: Python reads every
  shard with that metadata, the index has `metadata.total_size` equal to the data size and a
  `weight_map` in state dict order, and loading the shards it names as transformers does gives
  the same tensors.
- to_sharded/ and to_single/: the same tensors saved as shards over an earlier single file of
  zeros, and as a single file over earlier shards of zeros. Discovering the checkpoint as
  transformers 5.20.2 does in a local directory (R106: `model.safetensors`, then
  `model.safetensors.index.json`; `modeling_utils._get_resolved_checkpoint_files`) gives the new
  tensors, and no file of the earlier checkpoint remains.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import reference

reference.check()
DIR = Path(__file__).resolve().parents[2] / "target" / "tmp" / "m3_safetensors"
if not DIR.is_dir():
    raise SystemExit(f"{DIR} is missing: run `cargo test --test io_safetensors` first")

DTYPES = {
    "bool": torch.bool, "u8": torch.uint8, "i8": torch.int8, "i16": torch.int16,
    "i32": torch.int32, "i64": torch.int64, "f16": torch.float16, "bf16": torch.bfloat16,
    "f32": torch.float32, "f64": torch.float64,
}
METADATA = {"format": "pt", "writer": "homura"}


def values(shape, dtype):
    """The formula of gen_m3_safetensors.py."""
    i = torch.arange(int(torch.Size(shape).numel()), dtype=torch.int64)
    k = (i * 37 + 5) % 251
    if dtype == torch.bool:
        v = i % 3 == 0
    elif dtype == torch.uint8:
        v = k.to(dtype)
    elif dtype.is_floating_point:
        v = ((k - 125).to(torch.float64) / 8).to(dtype)
    else:
        v = (k - 125).to(dtype)
    return v.reshape(shape)


EXPECTED = {name: values([2, 3], dtype) for name, dtype in DTYPES.items()}
EXPECTED["transposed"] = values([2, 3], torch.float32).t().contiguous()
EXPECTED["scalar"] = values([], torch.float64)
EXPECTED["empty"] = values([0, 2], torch.int16)
passed = 0


def check(tensors):
    assert tensors.keys() == EXPECTED.keys(), tensors.keys()
    for name, t in EXPECTED.items():
        got = tensors[name]
        assert got.dtype == t.dtype and got.shape == t.shape, (name, got.dtype, got.shape)
        assert torch.equal(got, t), name


def single():
    path = DIR / "metadata.safetensors"
    check(load_file(path))
    with safe_open(path, "pt") as f:
        assert f.metadata() == METADATA, f.metadata()
    # The header keeps homura's order and starts the data 8-byte aligned.
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    assert n % 8 == 0
    header = json.loads(raw[8:8 + n])
    assert list(header) == ["__metadata__", *EXPECTED], list(header)


def sharded():
    directory = DIR / "sharded"
    with open(directory / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    assert list(weight_map) == list(EXPECTED), list(weight_map)
    total = sum(t.numel() * t.element_size() for t in EXPECTED.values())
    assert index["metadata"] == {"total_size": total}, index["metadata"]
    files = sorted(set(weight_map.values()))
    assert len(files) > 1 and all(f.startswith("model-") for f in files), files
    merged = {}
    for file in files:
        shard = load_file(directory / file)
        assert all(weight_map[k] == file for k in shard), file
        with safe_open(directory / file, "pt") as f:
            assert f.metadata() == METADATA
        size = sum(t.numel() * t.element_size() for t in shard.values())
        assert size <= 40 or len(shard) == 1, (file, size)
        merged.update(shard)
    check(merged)


def discover(directory):
    """Loads the checkpoint in `directory` in transformers' local discovery order (R106)."""
    if (directory / "model.safetensors").is_file():
        return load_file(directory / "model.safetensors")
    with open(directory / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    merged = {}
    for file in sorted(set(weight_map.values())):
        merged.update(load_file(directory / file))
    return merged


def layout_switch():
    to_sharded = DIR / "to_sharded"
    check(discover(to_sharded))
    assert not (to_sharded / "model.safetensors").exists()
    to_single = DIR / "to_single"
    check(discover(to_single))
    names = sorted(p.name for p in to_single.iterdir())
    assert names == ["model.safetensors"], names


for f in [single, sharded, layout_switch]:
    f()
    passed += 1
print(f"check_m3_safetensors: {passed} blocks passed")
