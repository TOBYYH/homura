"""Generates tests/golden/m3_pt/: PyTorch .pt files that the unit tests of src/io/{zip,pt,pickle}.rs
read (M3 step 3a), with the objects the reference reads from them.

Each `<name>.pt` that holds a supported object has `<name>.expected`: one line with the canonical
form of what `torch.load(weights_only=True)` returns (or, for pickle protocols that the reference
can't read that way, the object that was saved), which the Rust tests compare their `Value` with:

    None N, True T, False F, int i<decimal>, float f<16 hex digits of the IEEE bits>,
    str s<hex of the UTF-8 bytes>, list [a,b], tuple (a,b), dict {k:v,k:v},
    OrderedDict O{...}, with versions O<path=version;path=version>{...} (paths as str),
    tensor t (step 3a doesn't rebuild tensors).

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

`torch.save` writes a random `.data/serialization_id`, so regenerating changes the files' bytes
but not what they hold.

Usage: python tools/golden/gen_m3_pt.py (needs torch).
"""

import collections
import io
import math
import struct
import zipfile
from pathlib import Path

import torch

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


def write(name, data, expected=None):
    (OUT / f"{name}.pt").write_bytes(data)
    if expected is not None:
        (OUT / f"{name}.expected").write_text(canonical(expected) + "\n")


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
    """Asserts the layout torch.save writes (M3.3): stored, bit 3 on every entry, data 64-byte
    aligned, 16-byte data descriptors with a signature."""
    z = zipfile.ZipFile(io.BytesIO(data))
    for info in z.infolist():
        assert info.compress_type == zipfile.ZIP_STORED and info.flag_bits & 0x08
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

print(f"wrote {OUT}")
