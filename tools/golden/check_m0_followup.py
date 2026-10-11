"""Verify local reference behaviour behind the M0 follow-up review; writes no fixtures."""

import json
import struct
import warnings

import torch
from safetensors.torch import load


def rejects(operation):
    try:
        operation()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("reference accepted invalid arguments")


matrix = torch.ones((1, 1), dtype=torch.int32)
vector = matrix.flatten()
batch = matrix.reshape(1, 1, 1)
for beta, alpha in [(1.0, 0.5), (0.5, 1.0)]:
    outputs = [
        matrix.addmm(matrix, matrix, beta=beta, alpha=alpha),
        vector.addmv(matrix, vector, beta=beta, alpha=alpha),
        batch.baddbmm(batch, batch, beta=beta, alpha=alpha),
        matrix.addbmm(batch, batch, beta=beta, alpha=alpha),
    ]
    for output in outputs:
        assert output.dtype == torch.int32
        assert output.flatten().tolist() == [1]
    # addr has stricter coefficient validation than the matrix-multiply composites.
    rejects(lambda: matrix.addr(vector, vector, beta=beta, alpha=alpha))

rejects(lambda: torch.empty(8, dtype=torch.float64).uniform_(-1e308, 1e308))
rejects(lambda: torch.empty(8, dtype=torch.float16).uniform_(65520.0, 65521.0))
rejects(lambda: torch.randint(70000, 70001, (8,), dtype=torch.float16))
rejects(lambda: torch.randint(16777219, 16777220, (8,), dtype=torch.float32))

header = r'{"\u+001":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
rejects(lambda: json.loads(header))
blob = struct.pack("<Q", len(header)) + header.encode() + bytes([7])
try:
    load(blob)
except Exception:
    pass
else:
    raise AssertionError("safetensors accepted invalid Unicode escape")

# Coefficients are range-checked for integral inputs (addr also for floating inputs), and
# integral addmm truncates floating coefficients.
rejects(lambda: matrix.addmm(matrix, matrix, alpha=2**40))
rejects(lambda: matrix.addmm(matrix, matrix, beta=float("nan")))
rejects(lambda: matrix.addr(vector, vector, alpha=True))
half = torch.ones((1, 1), dtype=torch.float16)
assert half.addmm(half, half, alpha=1e5).dtype == torch.float16
rejects(lambda: half.addr(half.flatten(), half.flatten(), alpha=1e5))
assert matrix.addmm(matrix, matrix, alpha=-2.7).tolist() == [[-1]]
assert matrix.addr(vector, vector, beta=2, alpha=3).tolist() == [[5]]
byte = torch.ones((1, 1), dtype=torch.uint8)
assert byte.addmm(byte, byte, alpha=-1).tolist() == [[0]]
rejects(lambda: byte.addmm(byte, byte, alpha=-1.0))
flag = torch.tensor([True])
assert flag.unsqueeze(0).addr(flag, flag, beta=True, alpha=2).tolist() == [[True]]

rejects(lambda: torch.empty(1, dtype=torch.float16).uniform_(-60000.0, 60000.0))
# Outside the exactly representable range PyTorch only warns, and may sample outside [from, to);
# Homura rejects these intervals instead.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    assert torch.randint(2049, 2050, (4,), dtype=torch.float16).tolist() == [2048.0] * 4

# Homura's defined cast policy gives i32::MAX here, so the broad compatibility claim is false.
assert torch.tensor([-2147483649.0], dtype=torch.float64).to(torch.int32).tolist() == [-(2**31)]
# The narrowed claim: results agree when the truncated value fits in the target type, or in i32
# for narrower targets.
values = [300.7, -1.5, 70000.0, -70000.0, 2147483647.0, -2147483648.0]
for dtype, bits in [(torch.uint8, 8), (torch.int8, 8), (torch.int16, 16)]:
    for value in values:
        wrapped = int(value) % 2**bits
        if dtype.is_signed and wrapped >= 2 ** (bits - 1):
            wrapped -= 2**bits
        assert torch.tensor([value], dtype=torch.float64).to(dtype).tolist() == [wrapped]
print(f"M0 follow-up reference checks passed (torch {torch.__version__})")
