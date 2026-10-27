"""Check that PyTorch and transformers read the .pt files homura writes (M3 step 5); writes no
fixtures.

Run the Rust tests first; they write the files under target/tmp/:

- `cargo test --lib io::pt` (`fixture_files_for_python`): m3_pt_written_fixtures/<name>.pt, each
  reference file of tests/golden/m3_pt/ that homura reads, written again, and <name>_zip64.pt,
  the same with ZIP64 records everywhere;
- `cargo test --test io_pt_write` (`homura_files_for_python`): m3_pt_written/layouts.pt,
  checkpoint.pt and model.pt;
- `cargo test -p homura-models --test resnet_pt_write`: m3_pt_written/resnet/<repo>/, the mini
  repositories of homura-models/tests/golden/repos saved as config.json + pytorch_model.bin, and,
  with `-- --ignored` in a release build and the cache of fetch_hf_repos.py,
  m3_pt_written/resnet-18/ (checked when present; set HF_HUB_CACHE to target/hf-cache).

Every file is read with torch.load(weights_only=True), (weights_only=False) and (mmap=True), and:

- the fixtures equal the reference files as torch.load reads those: the same structure, dict and
  OrderedDict types, _metadata, tensor dtypes, sizes, strides, offsets, requires_grad, storage
  sizes and bytes, object identities and storage sharing (a Parameter of the reference is a plain
  tensor in homura's file);
- layouts.pt, checkpoint.pt and model.pt hold what the Rust test built (model.pt and the model
  of checkpoint.pt also load strictly into the same PyTorch modules, with their _metadata);
- every file's data.pkl is byte for byte what torch.save(pickle_protocol=2) writes for the object
  it holds with a pickler that memoizes only tensors, as homura's does, and its storage records
  have the same bytes; the only globals are OrderedDict, _rebuild_tensor_v2 and storage classes;
- every record is 64-byte aligned, without data descriptors, and Python's zipfile tests the
  archive; the _zip64 files have ZIP64 end records and zipfile reads them;
- the ResNet state dicts load strictly into transformers' ResNetForImageClassification built
  from the same config, equal the original repository's weights bit for bit, have its _metadata,
  give its logits exactly, and from_pretrained reads the saved repository without missing or
  unexpected keys.
"""

import collections
import io
import math
import os
import pickle
import pickletools
import struct
import zipfile
from pathlib import Path

import torch

import reference

reference.check_transformers()
from transformers import ResNetConfig, ResNetForImageClassification  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "target" / "tmp"
FIXTURES = TMP / "m3_pt_written_fixtures"
WRITTEN = TMP / "m3_pt_written"
GOLDEN = ROOT / "tests" / "golden" / "m3_pt"
REPOS = ROOT / "homura-models" / "tests" / "golden" / "repos"
TENSOR_FILES = ["dtypes", "big_endian", "layouts", "layouts_p4", "layouts_p5", "layouts_repacked",
                "bool_bytes", "optimizer", "state_dict", "checkpoint"]
VALUE_FILES = ["values_p2", "build_once"]
for d, how in [(FIXTURES, "cargo test --lib io::pt"), (WRITTEN, "cargo test --test io_pt_write"),
               (WRITTEN / "resnet", "cargo test -p homura-models --test resnet_pt_write")]:
    if not d.is_dir():
        raise SystemExit(f"{d} is missing: run `{how}` first")

checks = 0


def loads(path):
    """The object of a .pt file, read the three ways."""
    return [torch.load(path, weights_only=True), torch.load(path, weights_only=False),
            torch.load(path, weights_only=True, mmap=True)]


# Comparing objects.

def tensors_in(v, out):
    if isinstance(v, torch.Tensor):
        out.append(v)
    elif isinstance(v, (list, tuple)):
        for x in v:
            tensors_in(x, out)
    elif isinstance(v, dict):
        for x in v.values():
            tensors_in(x, out)
    return out


def float_bits(f):
    return struct.pack(">d", f)


def same(a, b, at="obj"):
    """Asserts that a (from homura's file) is b (the expectation), recursively."""
    if isinstance(b, torch.Tensor):
        assert type(a) is torch.Tensor, f"{at}: {type(a)}"
        assert a.dtype == b.dtype, f"{at}: {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{at}: {a.shape} != {b.shape}"
        assert a.stride() == b.stride(), f"{at}: {a.stride()} != {b.stride()}"
        assert a.storage_offset() == b.storage_offset(), at
        assert a.requires_grad == b.requires_grad, at
        sa, sb = a.untyped_storage(), b.untyped_storage()
        assert sa.nbytes() == sb.nbytes(), at
        if a.dtype == torch.bool:
            # The reference file can have BOOL bytes other than 0 and 1, which homura reads as 1.
            assert torch.equal(a, b), at
        else:
            assert bytes(sa) == bytes(sb), at
        return
    assert type(a) is type(b), f"{at}: {type(a)} != {type(b)}"
    if isinstance(b, float):
        assert float_bits(a) == float_bits(b), f"{at}: {a!r} != {b!r}"
    elif isinstance(b, (list, tuple)):
        assert len(a) == len(b), at
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{at}[{i}]")
    elif isinstance(b, dict):
        assert list(a.keys()) == list(b.keys()), f"{at}: {list(a)} != {list(b)}"
        ma, mb = getattr(a, "_metadata", None), getattr(b, "_metadata", None)
        assert (ma is None) == (mb is None), f"{at}: _metadata {ma} vs {mb}"
        if mb is not None:
            # homura keeps only the versions (fields other than "version" are dropped, as are
            # entries without one).
            versions = lambda m: [(k, v["version"]) for k, v in m.items() if "version" in v]  # noqa: E731
            assert versions(ma) == versions(mb), f"{at}: {ma} != {mb}"
        for k in b:
            same(a[k], b[k], f"{at}[{k!r}]")
    else:
        assert a == b, f"{at}: {a!r} != {b!r}"


def same_sharing(a, b, what):
    """The same pattern of object identity and storage sharing among the tensors."""
    def pattern(v):
        ts = tensors_in(v, [])
        ids, storages = {}, {}
        return [(ids.setdefault(id(t), i), storages.setdefault(t.untyped_storage().data_ptr(), i))
                for i, t in enumerate(ts)]
    assert pattern(a) == pattern(b), f"{what}: {pattern(a)} != {pattern(b)}"


# The bytes of the archive.

class TensorMemoPickler(pickle._Pickler):
    """Python's pickler, memoizing only tensors (homura's `Value` is a tree)."""

    def memoize(self, obj):
        if isinstance(obj, torch.Tensor):
            super().memoize(obj)


class TensorMemoPickle:
    Pickler = TensorMemoPickler
    Unpickler = pickle.Unpickler
    load = pickle.load


ALLOWED_GLOBALS = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2"} | {
    f"torch {name}Storage" for name in
    ["Float", "Double", "Half", "BFloat16", "Long", "Int", "Short", "Char", "Byte", "Bool"]}


def records(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    return {i.filename: z.read(i.filename) for i in z.infolist()}


def check_archive(path, obj, zip64=False):
    """The layout of homura's archive, and its pickle and records against torch.save's."""
    data = path.read_bytes()
    z = zipfile.ZipFile(path)
    assert z.testzip() is None, path
    names = z.namelist()
    keys = sorted((n for n in names if n.startswith("archive/data/")),
                  key=lambda n: int(n.rsplit("/", 1)[1]))
    assert names == (["archive/data.pkl", "archive/.format_version", "archive/.storage_alignment",
                      "archive/byteorder"] + keys
                     + ["archive/version", "archive/.data/serialization_id"]), names
    for info in z.infolist():
        assert info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 0x08, info
        name_len, extra_len = struct.unpack("<HH", data[info.header_offset + 26:][:4])
        start = info.header_offset + 30 + name_len + extra_len
        assert start % 64 == 0, (path, info.filename)
    sid = z.read("archive/.data/serialization_id")
    assert len(sid) == 40 and sid.isdigit(), sid
    assert (b"PK\x06\x06" in data[-200:]) == zip64, path
    ops = pickletools.genops(z.read("archive/data.pkl"))
    globals_ = {arg for op, arg, _ in ops if op.name == "GLOBAL"}
    assert globals_ <= ALLOWED_GLOBALS, globals_
    # torch.save of what torch.load read, with a pickler like homura's.
    buf = io.BytesIO()
    torch.save(obj, buf, pickle_module=TensorMemoPickle, pickle_protocol=2)
    theirs = records(buf.getvalue())
    ours = {n: z.read(n) for n in names}
    for name in ["archive/data.pkl", "archive/version", "archive/byteorder",
                 "archive/.format_version", "archive/.storage_alignment"] + keys:
        assert ours[name] == theirs[name], (path, name)
    assert sorted(ours) == sorted(theirs), (sorted(ours), sorted(theirs))


def check_file(path, expected, zip64=False):
    """Reads `path` three ways and compares it with `expected`; checks its archive."""
    global checks
    for obj in loads(path):
        same(obj, expected)
        same_sharing(obj, expected, path)
    check_archive(path, torch.load(path, weights_only=True), zip64)
    checks += 1


# 1. The reference files, written again.

for name in TENSOR_FILES + VALUE_FILES:
    ref = torch.load(GOLDEN / f"{name}.pt", weights_only=False)
    # Parameters are written as plain tensors (homura has no Parameter class in its values).
    def plain(v):
        if isinstance(v, torch.nn.Parameter):
            t = v.data.detach()
            t.requires_grad_(v.requires_grad)
            return t
        return v
    def unparam(v, seen):
        if isinstance(v, torch.Tensor):
            return seen.setdefault(id(v), plain(v))
        if isinstance(v, list):
            return [unparam(x, seen) for x in v]
        if isinstance(v, tuple):
            return tuple(unparam(x, seen) for x in v)
        if isinstance(v, dict):
            out = type(v)((k, unparam(x, seen)) for k, x in v.items())
            if hasattr(v, "_metadata"):
                out._metadata = v._metadata
            return out
        return v
    expected = unparam(ref, {})
    check_file(FIXTURES / f"{name}.pt", expected)
    check_file(FIXTURES / f"{name}_zip64.pt", expected, zip64=True)
print(f"reference files written again: {2 * len(TENSOR_FILES + VALUE_FILES)} files")


# 2. The files of tests/io_pt_write.rs.

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


DTYPES = {"bool": torch.bool, "u8": torch.uint8, "i8": torch.int8, "i16": torch.int16,
          "i32": torch.int32, "i64": torch.int64, "f16": torch.float16, "bf16": torch.bfloat16,
          "f32": torch.float32, "f64": torch.float64}

layouts = collections.OrderedDict((name, values([2, 3], d)) for name, d in DTYPES.items())
base = values([3, 4], torch.float32)
layouts["base"] = base
layouts["transposed"] = base.t()
layouts["slice"] = base[1:, 2:]
layouts["expanded"] = base[0, :1].expand(3, 4)
layouts["scalar"] = values([], torch.float64)
layouts["empty"] = values([0, 2], torch.int16)
layouts["grad"] = values([2], torch.float32).requires_grad_(True)
check_file(WRITTEN / "layouts.pt", layouts)
for obj in loads(WRITTEN / "layouts.pt"):
    assert not hasattr(obj, "_metadata")
    ptr = obj["base"].untyped_storage().data_ptr()
    assert all(obj[k].untyped_storage().data_ptr() == ptr
               for k in ["transposed", "slice", "expanded"])

linear = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.BatchNorm2d(4))
with torch.no_grad():
    linear[0].weight.copy_(values([4, 3], torch.float32))
    linear[0].bias.copy_(values([4], torch.float32))
model_sd = linear.state_dict()
shared = values([2], torch.float64)
checkpoint = {
    "model": model_sd,
    "state": {0: {"step": values([], torch.float32)}, 1: None},
    "ints": [0, 1, 255, 256, 65535, 65536, -1, -2**31, 2**31 - 1, 2**31, 2**63 - 1, -2**63],
    "floats": (0.5, -0.0, float("nan"), math.inf, -math.inf),
    "strings": ["", "名前", "a\nb"],
    "flags": (True, False, None),
    "ordered": collections.OrderedDict([("z", 1), ("a", 2)]),
    "list_1000": list(range(1000)),
    "list_1001": list(range(1001)),
    "dict_1001": {i: 2 * i for i in range(1001)},
    "same": [shared, shared],
    "nested": (([],),),
}
check_file(WRITTEN / "checkpoint.pt", checkpoint)
for obj in loads(WRITTEN / "checkpoint.pt"):
    assert obj["same"][0] is obj["same"][1]
    fresh = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.BatchNorm2d(4))
    fresh.load_state_dict(obj["model"], strict=True)

model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2),
                            torch.nn.Linear(1, 2))
with torch.no_grad():
    for k, t in model.state_dict().items():
        t.copy_(values(list(t.shape), t.dtype) if t.dtype.is_floating_point else torch.full_like(t, 5))
check_file(WRITTEN / "model.pt", model.state_dict())
for obj in loads(WRITTEN / "model.pt"):
    model.load_state_dict(obj, strict=True)
print("files of tests/io_pt_write.rs: 3 files")


# 3. ResNet repositories.

def resnet_input(config):
    n = 2 * config.num_channels * 32 * 32
    i = torch.arange(n, dtype=torch.int64)
    return (((i * 7919 + 13) % 1021).to(torch.float32) / 1024 - 0.5).reshape(
        2, config.num_channels, 32, 32)


def check_resnet(saved, original):
    global checks
    ref, info = ResNetForImageClassification.from_pretrained(original, output_loading_info=True)
    assert all(not v for v in info.values()), info
    ref.eval()
    config = ResNetConfig.from_pretrained(saved)
    ref_sd = ref.state_dict()
    expected = collections.OrderedDict((k, v.detach()) for k, v in ref_sd.items())
    expected._metadata = ref_sd._metadata
    for sd in loads(saved / "pytorch_model.bin"):
        same(sd, expected)
        model = ResNetForImageClassification(config).eval()
        model.load_state_dict(sd, strict=True)
        x = resnet_input(config)
        with torch.no_grad():
            assert torch.equal(model(x).logits, ref(x).logits), saved
    check_archive(saved / "pytorch_model.bin", torch.load(saved / "pytorch_model.bin"))
    loaded, info = ResNetForImageClassification.from_pretrained(saved, use_safetensors=False,
                                                                output_loading_info=True)
    assert all(not v for v in info.values()), info
    checks += 1


mini = sorted(p.name for p in (WRITTEN / "resnet").iterdir())
assert mini == ["basic", "basic_no_labels", "bottleneck", "bottleneck_downsample"], mini
for name in mini:
    check_resnet(WRITTEN / "resnet" / name, REPOS / name)
print(f"mini ResNet repositories: {len(mini)}")
if (WRITTEN / "resnet-18").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    check_resnet(WRITTEN / "resnet-18", "microsoft/resnet-18")
    print("microsoft/resnet-18: checked")
else:
    print("microsoft/resnet-18: not written (run the ignored test), skipped")
print(f"ok: {checks} files")
