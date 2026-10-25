"""Generates tests/golden/m3_pt/: PyTorch .pt files that the unit tests of src/io/{zip,pt,pickle}.rs
and tests/io_pt.rs read (M3 steps 3a and 3b), with the objects the reference reads from them.

Each `<name>.pt` that holds a supported object has `<name>.expected`: one line with the canonical
form of what `torch.load(weights_only=True)` returns (or, for pickle protocols that the reference
can't read that way, the object that was saved), which the Rust tests compare their `Value` with:

    None N, True T, False F, int i<decimal>, float f<16 hex digits of the IEEE bits>,
    str s<hex of the UTF-8 bytes>, list [a,b], tuple (a,b), dict {k:v,k:v},
    OrderedDict O{...}, with versions O<path=version;path=version>{...} (paths as str),
    tensor t.

A file whose object holds tensors also has `<name>.safetensors`: tensor `i` (in the order the
canonical form lists them, depth first) under the key `i`, with the values the reference reads
(contiguous; BOOL as 0 and 1), and in `__metadata__` under the same key a JSON object with the
tensor's `offset` and `stride`, `storage_nbytes`, `requires_grad`, `object` (the first tensor
that is the same Python object) and `storage` (the first tensor with the same storage).

- values_p2.pt, values_p4.pt, values_p5.pt: a checkpoint of plain Python values (every kind
  above, int keys, an OrderedDict with `_metadata`, a list referenced twice) saved with pickle
  protocols 2 (the default), 4 and 5. The reference's weights-only unpickler can't read 4 and 5.
- state_dict.pt: `Sequential(Conv2d, BatchNorm2d, Linear).state_dict()`, versions
  {"": 1, "0": 1, "1": 2, "2": 1}.
- empty_state_dict.pt: `Module().state_dict()`, which has no tensors, only the version {"": 1}.
- checkpoint.pt: {"model": that state dict, "optimizer": an SGD state dict after one step,
  "epoch": 3, ...}.
- repacked.pt: values_p2.pt rewritten by `zipfile` with the archive directory renamed: no data
  descriptors, data not aligned, CRCs set. zip64.pt: the same with ZIP64 local headers
  (`force_zip64=True`).
- crc0.pt: values_p2's object saved with CRC computation off, so every CRC is 0.
- deep.pt, wide.pt: lists of lists whose shared references expand to 2**30 and 64**8 leaves.
  cycle_list.pt, cycle_dict.pt: a list and a dict that contain themselves. No .expected: the
  tests expect errors.
- build_*.pt: hand-written pickles of an empty OrderedDict whose BUILD opcodes bind `_metadata`
  in different orders (R114), stored in plain zip archives. The reference applies each BUILD
  as `inst.__dict__.update(state)`: rebinding or extending the state dict afterwards changes
  nothing, an empty state keeps the attributes, and changing the bound metadata dict is seen.
- torchscript.pt (`torch.jit.save`) and legacy.pt (`_use_new_zipfile_serialization=False`):
  formats that are rejected.

Tensors (step 3b):

- dtypes.pt: an OrderedDict with a tensor of each of the ten dtypes, with extreme values, NaN,
  infinities and negative zero.
- layouts.pt: a dict of tensors with the layouts the files can hold: zero-dimensional, empty
  (also at an offset), views with offsets and strides, transposed, expanded (stride 0) and
  overlapping views of one storage, a tensor and a Parameter saved twice (the same object), a
  Parameter with and one without requires_grad, a tensor that requires grad, integer and byte
  views. layouts_p4.pt, layouts_p5.pt: the same with pickle protocols 4 and 5.
  layouts_repacked.pt: layouts.pt rewritten by `zipfile`: no data descriptors, data not aligned.
- big_endian.pt: dtypes.pt with the `byteorder` record `big` and the bytes of each element
  swapped (repacked by `zipfile`); the reference reads the same values.
- bool_bytes.pt: BOOL tensors whose storage holds the bytes 2, 255, 0 and 1, which PyTorch writes
  and reads as True, True, False and True (with the right CRCs).
- optimizer.pt: an Adam state dict after one step (zero-dimensional `step` tensors).
- module.pt: a whole `nn.Linear` pickled, which the reference's weights-only unpickler rejects.
- sharded/: `pytorch_model-0000i-of-00002.bin` shards of state_dict.pt's tensors (plain dicts,
  as Hugging Face writes them) and their `pytorch_model.bin.index.json`.

`torch.save` writes a random `.data/serialization_id`, so a file is only rewritten when another
record changes: regenerating leaves the files as they are.

Usage: python tools/golden/gen_m3_pt.py (needs torch).
"""

import collections
import io
import json
import math
import pickle
import struct
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save as save_safetensors

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m3_pt"

reference.check()


def canonical(obj):
    """The canonical form described above."""
    if obj is None:
        return "N"
    if obj is True:
        return "T"
    if obj is False:
        return "F"
    if isinstance(obj, int):
        return f"i{obj}"
    if isinstance(obj, float):
        return "f" + struct.pack(">d", obj).hex()
    if isinstance(obj, str):
        return "s" + obj.encode().hex()
    if isinstance(obj, torch.Tensor):
        return "t"
    if isinstance(obj, list):
        return "[" + ",".join(canonical(x) for x in obj) + "]"
    if isinstance(obj, tuple):
        return "(" + ",".join(canonical(x) for x in obj) + ")"
    if isinstance(obj, dict):
        items = ",".join(f"{canonical(k)}:{canonical(v)}" for k, v in obj.items())
        if type(obj) is dict:
            return "{" + items + "}"
        assert type(obj) is collections.OrderedDict
        metadata = getattr(obj, "_metadata", None)
        versions = ""
        if metadata is not None:
            versions = "<" + ";".join(
                f"{canonical(path)}={entry['version']}"
                for path, entry in metadata.items()
                if "version" in entry
            ) + ">"
        return "O" + versions + "{" + items + "}"
    raise TypeError(type(obj))


def save(obj, **kwargs):
    buf = io.BytesIO()
    torch.save(obj, buf, **kwargs)
    return buf.getvalue()


def load(data, weights_only=True):
    return torch.load(io.BytesIO(data), weights_only=weights_only)


def tensors_in(obj):
    """The tensors of `obj`, in the order of its canonical form."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for x in obj for t in tensors_in(x)]
    if isinstance(obj, dict):
        return [t for x in obj.values() for t in tensors_in(x)]
    return []


def plain(t):
    """The values of `t` as a new contiguous tensor; BOOL with the bytes 0 and 1."""
    t = t.detach()
    if t.dtype == torch.bool:
        out = torch.zeros(t.shape, dtype=torch.uint8).masked_fill_(t, 1).bool()
        assert out.tolist() == t.tolist()
        assert set(out.view(torch.uint8).flatten().tolist()) <= {0, 1}
        return out
    return t.clone().contiguous()


def expected_tensors(obj):
    """The `<name>.safetensors` described above."""
    tensors = tensors_in(obj)
    data, metadata = {}, {}
    for i, t in enumerate(tensors):
        storage = t.untyped_storage()
        same_storage = next(
            j for j, u in enumerate(tensors)
            if u is t or (storage.nbytes() > 0
                          and u.untyped_storage().data_ptr() == storage.data_ptr()))
        data[str(i)] = plain(t)
        metadata[str(i)] = json.dumps({
            "offset": t.storage_offset(), "stride": list(t.stride()),
            "storage_nbytes": storage.nbytes(), "requires_grad": t.requires_grad,
            "object": next(j for j, u in enumerate(tensors) if u is t),
            "storage": same_storage,
        })
    return save_safetensors(data, metadata) if tensors else None


def records(data):
    """The records of a zip archive but the random serialization id, or None for other data."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    return [(n, z.read(n)) for n in z.namelist() if not n.endswith("/.data/serialization_id")]


def same_values(a, b):
    """Whether two objects hold the same values, tensors compared bit for bit."""
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and a.stride() == b.stride() and a.storage_offset() == b.storage_offset()
                and a.requires_grad == b.requires_grad
                and plain(a).view(-1).view(torch.uint8).equal(plain(b).view(-1).view(torch.uint8)))
    if isinstance(a, dict):
        return (type(a) is type(b) and list(a) == list(b)
                and all(same_values(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(same_values(x, y) for x, y in zip(a, b)))
    return canonical(a) == canonical(b)


def same_legacy(old, new):
    """Whether two files in the format before PyTorch 1.6 hold the same object (their storage keys
    differ between processes)."""
    if old[:2] != b"\x80\x02" or new[:2] != b"\x80\x02":
        return False
    return same_values(torch.load(io.BytesIO(old), weights_only=False),
                       torch.load(io.BytesIO(new), weights_only=False))


def safetensors_content(data):
    """The header (as parsed JSON) and data of a safetensors file: the writer orders the header's
    entries differently from run to run."""
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    body = data[8 + n:]
    tensors = {k: (v["dtype"], v["shape"], body[v["data_offsets"][0]:v["data_offsets"][1]])
               for k, v in header.items() if k != "__metadata__"}
    return header.get("__metadata__"), tensors


def same_torchscript(old, new):
    """Whether two TorchScript archives have the same records (the code that `torch.jit.save`
    writes changes from run to run; torchscript.pt is only read to be rejected)."""
    old, new = records(old), records(new)
    names = lambda r: [n for n, _ in r]
    return (old is not None and new is not None and names(old) == names(new)
            and any(n.endswith("/constants.pkl") for n in names(old)))


def write_file(path, data):
    """Writes `data` to `path` unless the file holds it, up to the serialization id (in the format
    before PyTorch 1.6, the storage keys; in TorchScript archives, the code)."""
    if path.exists():
        old = path.read_bytes()
        if old == data or (records(old) is not None and records(old) == records(data)) \
                or same_legacy(old, data) or same_torchscript(old, data) \
                or (path.suffix == ".safetensors"
                    and safetensors_content(old) == safetensors_content(data)):
            return
    path.write_bytes(data)


def write(name, data, expected=None):
    write_file(OUT / f"{name}.pt", data)
    if expected is not None:
        write_file(OUT / f"{name}.expected", (canonical(expected) + "\n").encode())
        tensors = expected_tensors(expected)
        if tensors is not None:
            write_file(OUT / f"{name}.safetensors", tensors)


def values():
    shared = [1, 2]
    ordered = collections.OrderedDict([("b", 1), ("a", 2)])
    with_metadata = collections.OrderedDict([("w", 0.5)])
    with_metadata._metadata = collections.OrderedDict(
        [("", {"version": 1}), ("bn", {"version": 2, "other": "ignored"}), ("none", {})]
    )
    return {
        "none": None,
        "bools": [True, False],
        "ints": [0, 1, 255, 256, 65535, 65536, -1, -129, 2**31 - 1, -(2**31), 2**31,
                 2**40, 2**63 - 1, -(2**63)],
        "floats": [0.0, -0.0, 1.5, -2.25, 1e-300, 1e300, math.inf, -math.inf, math.nan],
        "strs": ["", "a", "é\U0001f600", "x" * 300, "line\nbreak"],
        "tuples": [(), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)],
        "nested": {"list": [[1], {"k": (None,)}], "empty": [[], {}]},
        "int_keys": {0: "zero", -5: "minus five", 2**40: "big"},
        "ordered": ordered,
        "with_metadata": with_metadata,
        "shared": [shared, shared],
    }


def model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2),
                               torch.nn.Linear(1, 2))


def repack(data, dirname, force_zip64=False):
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for name in src.namelist():
            new = dirname + name[name.index("/"):]
            with z.open(zipfile.ZipInfo(new), "w", force_zip64=force_zip64) as f:
                f.write(src.read(name))
    return out.getvalue()


def structure(data):
    """Asserts the layout torch.save writes (M3.3): stored, data 64-byte aligned, bit 3 and
    16-byte data descriptors with a signature on every entry but empty ones, which have neither."""
    z = zipfile.ZipFile(io.BytesIO(data))
    for info in z.infolist():
        assert info.compress_type == zipfile.ZIP_STORED
        if info.file_size == 0 and not info.flag_bits & 0x08:
            continue
        assert info.flag_bits & 0x08
        offset = info.header_offset
        name_len, extra_len = struct.unpack("<HH", data[offset + 26:offset + 30])
        start = offset + 30 + name_len + extra_len
        assert start % 64 == 0 or info.file_size == 0
        assert struct.unpack("<I", data[start + info.file_size:start + info.file_size + 4])[0] \
            == 0x08074B50


OUT.mkdir(exist_ok=True)

obj = values()
p2 = save(obj)
structure(p2)
back = load(p2)
assert canonical(back) == canonical(obj)
assert back["shared"][0] is back["shared"][1]
write("values_p2", p2, obj)
for protocol in (4, 5):
    data = save(obj, pickle_protocol=protocol)
    try:
        load(data)
        raise AssertionError("the weights-only unpickler reads protocol 4 now")
    except Exception as e:
        assert "Unsupported operand" in str(e), e
    assert canonical(load(data, weights_only=False)) == canonical(obj)
    write(f"values_p{protocol}", data, obj)

state_dict = model().state_dict()
sd = save(state_dict)
structure(sd)
back = load(sd)
assert dict(back._metadata) == {"": {"version": 1}, "0": {"version": 1}, "1": {"version": 2},
                                "2": {"version": 1}}
write("state_dict", sd, back)
# A state dict without tensors: an empty module's, with the root's version.
data = save(torch.nn.Module().state_dict())
back = load(data)
assert dict(back._metadata) == {"": {"version": 1}} and len(back) == 0
write("empty_state_dict", data, back)

net = model()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
net(torch.ones(2, 1, 1, 1)).sum().backward()
optimizer.step()
checkpoint = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 3,
              "loss": 0.25, "name": "run", "betas": (0.9, 0.999), "note": None}
data = save(checkpoint)
back = load(data)
assert back["model"]._metadata["1"] == {"version": 2}
write("checkpoint", data, back)

for name, zip64 in (("repacked", False), ("zip64", True)):
    data = repack(p2, "model_v1", force_zip64=zip64)
    z = zipfile.ZipFile(io.BytesIO(data))
    assert all(not info.flag_bits & 0x08 and info.CRC != 0 for info in z.infolist())
    back = load(data)
    assert canonical(back) == canonical(obj)
    write(name, data, back)

torch.serialization.set_crc32_options(False)
data = save(obj)
torch.serialization.set_crc32_options(True)
assert all(info.CRC == 0 for info in zipfile.ZipFile(io.BytesIO(data)).infolist())
back = load(data)
assert canonical(back) == canonical(obj)
write("crc0", data, back)

deep = [0]
for _ in range(30):
    deep = [deep, deep]
write("deep", save(deep))
wide = [0]
for _ in range(8):
    wide = [wide] * 64
write("wide", save(wide))
cycle = []
cycle.append(cycle)
data = save(cycle)
assert (lambda c: c[0] is c)(load(data))
write("cycle_list", data)
cycle = {}
cycle["self"] = cycle
data = save(cycle)
assert (lambda c: c["self"] is c)(load(data))
write("cycle_dict", data)

def pickle_str(s):
    b = s.encode()
    return b"X" + struct.pack("<I", len(b)) + b


def pickle_metadata(version):
    """`{"": {"version": version}}`."""
    return (b"}" + pickle_str("") + b"}" + pickle_str("version") + b"K" + bytes([version])
            + b"ss")


# OrderedDict() in memo 2, then BUILD with the state {"_metadata": <version 1>} in memo 0.
od = b"\x80\x02ccollections\nOrderedDict\n)Rq\x02"
first = b"}q\x00" + pickle_str("_metadata") + pickle_metadata(1) + b"sb"
builds = {
    "build_once": (od + first + b".", 1),
    # The state dict's _metadata is rebound after BUILD.
    "build_state_rebound": (od + first + b"h\x00" + pickle_str("_metadata") + pickle_metadata(2)
                            + b"sh\x02.", 1),
    # An attribute that homura doesn't support is added to the state dict after BUILD.
    "build_state_extended": (od + first + b"h\x00" + pickle_str("other") + b"K\x01sh\x02.", 1),
    # A second BUILD with an empty state.
    "build_empty_state": (od + first + b"}b.", 1),
    # A second BUILD with another _metadata.
    "build_replaced": (od + first + b"}" + pickle_str("_metadata") + pickle_metadata(2)
                       + b"sb.", 2),
    # The bound metadata dict (memo 1) is changed after BUILD.
    "build_metadata_changed": (od + b"}" + pickle_str("_metadata") + pickle_metadata(1)
                               + b"q\x01sbh\x01" + pickle_str("") + b"}" + pickle_str("version")
                               + b"K\x02ssh\x02.", 2),
}
for name, (pickled, version) in builds.items():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for entry, data in (("archive/data.pkl", pickled), ("archive/version", b"3\n")):
            z.writestr(zipfile.ZipInfo(entry, date_time=(2026, 1, 1, 0, 0, 0)), data)
    back = load(buf.getvalue())
    assert type(back) is collections.OrderedDict and len(back) == 0
    assert dict(back.__dict__) == {"_metadata": {"": {"version": version}}}, (name, back.__dict__)
    write(name, buf.getvalue(), back)

buf = io.BytesIO()
torch.jit.save(torch.jit.script(torch.nn.Linear(1, 1)), buf)
z = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
assert any(n.endswith("/constants.pkl") for n in z.namelist())
write("torchscript", buf.getvalue())
data = save(collections.OrderedDict(w=torch.ones(2)), _use_new_zipfile_serialization=False)
assert data[:8] == bytes([0x80, 0x02, 0x8A, 0x0A, 0x6C, 0xFC, 0x9C, 0x46])
write("legacy", data)


# Tensors (step 3b).


def dtypes():
    nan, inf = math.nan, math.inf
    return collections.OrderedDict([
        ("f32", torch.tensor([0.0, -0.0, 1.5, inf, -inf, nan, 1e-45, 3.4e38])),
        ("f64", torch.tensor([0.0, -0.0, 1e-300, 1e300, nan, -inf], dtype=torch.float64)),
        ("f16", torch.tensor([0.0, -0.0, 65504.0, 6e-8, inf, nan], dtype=torch.float16)),
        ("bf16", torch.tensor([0.0, 1.0, -3e38, 1e-40, -inf, nan], dtype=torch.bfloat16)),
        ("i64", torch.tensor([0, -1, 2**63 - 1, -(2**63), 1 << 40])),
        ("i32", torch.tensor([0, -1, 2**31 - 1, -(2**31)], dtype=torch.int32)),
        ("i16", torch.tensor([0, -1, 2**15 - 1, -(2**15)], dtype=torch.int16)),
        ("i8", torch.tensor([0, -1, 127, -128], dtype=torch.int8)),
        ("u8", torch.tensor([0, 1, 128, 255], dtype=torch.uint8)),
        ("bool", torch.tensor([True, False, True])),
    ])


def layouts():
    torch.manual_seed(0)
    base = torch.arange(24, dtype=torch.float32)
    grid = base.view(4, 6)
    shared = torch.ones(3)
    param = torch.nn.Parameter(torch.randn(3))
    ints = torch.arange(6).view(2, 3)
    return {
        "base": base,
        "scalar": torch.tensor(2.5),
        "empty": torch.zeros(0, 3),
        "empty_at_end": base[24:],
        "strided": base[1::3],
        "transposed": grid.t(),
        "column": grid[:, 2],
        "expanded": torch.arange(3.0).expand(2, 3),
        "overlapping": base.as_strided((3, 4), (2, 1), 1),
        "same": [shared, shared],
        "param": param,
        "param_again": param,
        "frozen": torch.nn.Parameter(torch.ones(2), requires_grad=False),
        "requires_grad": torch.ones(2, requires_grad=True),
        "ints": ints[:, 1:],
        "ints_base": ints,
        "bytes": torch.arange(8, dtype=torch.uint8)[2:6],
    }


def storage_dtypes(data):
    """The dtype of each storage record of a `.pt` file, from its pickle's persistent ids."""
    z = zipfile.ZipFile(io.BytesIO(data))
    name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
    found = {}

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            if module == "torch" and name.endswith("Storage"):
                return getattr(torch, name)
            assert module == "torch._utils", (module, name)
            return lambda *args: None

        def persistent_load(self, pid):
            found[pid[2]] = pid[1].dtype
            return None

    Unpickler(io.BytesIO(z.read(name))).load()
    return found


def to_big_endian(data):
    """`data` with the byteorder record `big` and each element's bytes swapped."""
    dtype_of = storage_dtypes(data)
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for name in src.namelist():
            record = src.read(name)
            rel = name[name.index("/") + 1:]
            if rel == "byteorder":
                record = b"big"
            elif rel.startswith("data/"):
                size = dtype_of[rel[len("data/"):]].itemsize
                record = b"".join(record[i:i + size][::-1] for i in range(0, len(record), size))
            z.writestr(zipfile.ZipInfo(name), record)
    return out.getvalue()


obj = dtypes()
data = save(obj)
structure(data)
back = load(data)
assert same_values(back, obj)
write("dtypes", data, back)
big = to_big_endian(data)
assert zipfile.ZipFile(io.BytesIO(big)).read("archive/byteorder") == b"big"
assert big != data and same_values(load(big), back)
write("big_endian", big, load(big))

obj = layouts()
data = save(obj)
structure(data)
back = load(data)
assert same_values(back, obj)
assert back["same"][0] is back["same"][1] and back["param"] is back["param_again"]
assert type(back["param"]) is torch.nn.Parameter and back["param"].requires_grad
assert back["expanded"].stride() == (0, 1) and back["empty_at_end"].storage_offset() == 24
write("layouts", data, back)
for protocol in (4, 5):
    data_p = save(obj, pickle_protocol=protocol)
    back_p = load(data_p, weights_only=False)
    assert same_values(back_p, back)
    write(f"layouts_p{protocol}", data_p, back_p)
repacked = repack(data, "layouts_repacked")
z = zipfile.ZipFile(io.BytesIO(repacked))
assert any(not info.flag_bits & 0x08 for info in z.infolist())
starts = [info.header_offset + 30 + len(info.filename) for info in z.infolist()
          if "/data/" in info.filename and info.file_size > 0]
assert any(start % 4 != 0 for start in starts), starts
assert same_values(load(repacked), back)
write("layouts_repacked", repacked, load(repacked))

raw = torch.tensor([2, 255, 0, 1], dtype=torch.uint8)
obj = {"flags": raw.view(torch.bool), "tail": raw[1:].view(torch.bool)}
data = save(obj)
z = zipfile.ZipFile(io.BytesIO(data))
assert z.read("archive/data/0") == b"\x02\xff\x00\x01" and z.testzip() is None
back = load(data)
assert back["flags"].tolist() == [True, True, False, True]
assert back["tail"].tolist() == [True, False, True] and back["tail"].storage_offset() == 1
write("bool_bytes", data, back)

net = model()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
net(torch.ones(2, 1, 1, 1)).sum().backward()
optimizer.step()
data = save(optimizer.state_dict())
back = load(data)
assert back["state"][0]["step"].shape == () and back["state"][0]["step"].dtype == torch.float32
write("optimizer", data, back)

torch.manual_seed(0)
data = save(torch.nn.Linear(1, 1))
try:
    load(data)
    raise AssertionError("the weights-only unpickler reads modules now")
except pickle.UnpicklingError as e:
    assert "torch.nn.modules.linear.Linear" in str(e), e
write("module", data)

(OUT / "sharded").mkdir(exist_ok=True)
state_dict = model().state_dict()
assert same_values(load((OUT / "state_dict.pt").read_bytes()), state_dict)
shards = [[k for k in state_dict if not k.startswith("2.")],
          [k for k in state_dict if k.startswith("2.")]]
weight_map = {}
for i, keys in enumerate(shards):
    name = f"pytorch_model-{i + 1:05}-of-{len(shards):05}.bin"
    data = save({k: state_dict[k] for k in keys})
    back = load(data)
    assert type(back) is dict and list(back) == keys
    write_file(OUT / "sharded" / name, data)
    weight_map.update((k, name) for k in keys)
total = sum(t.numel() * t.element_size() for t in state_dict.values())
index = {"metadata": {"total_size": total}, "weight_map": weight_map}
write_file(OUT / "sharded" / "pytorch_model.bin.index.json",
           (json.dumps(index, indent=2) + "\n").encode())

print(f"wrote {OUT}")
