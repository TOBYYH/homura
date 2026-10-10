"""Check the reference semantics behind tests/m0_regressions.rs with local PyTorch.

Usage: python tools/golden/check_m0_edges.py (needs CPU torch and safetensors).
Does not rewrite committed golden fixtures. Homura-specific safety and concurrency contracts
are tested in Rust; this script checks only reference-library behaviour.
"""

import struct
import warnings

import torch
from safetensors.torch import load


def rejects(operation):
    try:
        operation()
    except (RuntimeError, IndexError):
        return
    raise AssertionError("reference accepted invalid arguments")


start = 2**53
assert torch.arange(start, start + 3).tolist() == [start, start + 1, start + 2]
assert torch.randint(start + 1, start + 2, (8,)).tolist() == [start + 1] * 8
assert len(torch.randint(-(2**63), 2**63 - 1, (16,))) == 16
assert len(torch.empty(0, 3).chunk(4, 0)) == 4
x = torch.arange(6).view(2, 3)
x.repeat(1, 1).fill_(99)
assert x.flatten().tolist() == list(range(6))
assert torch.tensor([300.0, -1.0]).to(torch.uint8).tolist() == [44, 255]
assert torch.signbit(torch.full((1,), -0.0)).item()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    assert torch.empty(0).var(correction=-1).isnan().item()
    assert torch.empty(0).std(correction=-1).isnan().item()
rejects(lambda: torch.tensor(1.0).flatten(10, 10))
rejects(lambda: torch.empty(2).normal_(0.0, -1.0))
rejects(lambda: torch.empty(2).uniform_(2.0, 1.0))
rejects(lambda: torch.zeros(2, 4).addbmm(torch.ones(1, 2, 3), torch.ones(2, 3, 4)))
rejects(lambda: torch.zeros(2, 4, dtype=torch.int64).addmm(torch.ones(2, 3), torch.ones(3, 4), beta=0))
a = torch.ones(1, 2, 3, dtype=torch.int32)
b = torch.ones(1, 3, 4, dtype=torch.int32)
assert torch.ones(2, 4, dtype=torch.int32).addbmm(a, b).dtype == torch.int32
# Verify the half-type upper endpoint, independently of homura's random sequence.
torch.manual_seed(42)
for dtype in (torch.float16, torch.bfloat16):
    values = torch.rand(100_000, dtype=dtype)
    assert bool(((values >= 0) & (values < 1)).all())

bad_files = {
    "overlap": ('{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b"\0"),
    "hole": ('{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}', b"\0\0"),
    "trailing payload": ('{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b"\0\0"),
    "invalid metadata": ('{"__metadata__":{"version":1}}', b""),
    "invalid number": ('{"a":{"dtype":"U8","shape":[01],"data_offsets":[0,1]}}', b"\0"),
}
for name, (header, payload) in bad_files.items():
    encoded = header.encode()
    blob = struct.pack("<Q", len(encoded)) + encoded + payload
    try:
        load(blob)
    except Exception:
        continue
    raise AssertionError(f"safetensors accepted {name}")
print(f"M0 edge reference checks passed (torch {torch.__version__})")
