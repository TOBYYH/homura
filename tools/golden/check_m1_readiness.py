"""Check reference semantics for remaining M0 composite edges; does not write fixtures."""

import torch

for dtype, alpha, expected in [
    (torch.uint8, 255.5, 255),
    (torch.int8, 127.5, 127),
    (torch.int16, 32767.5, 32767),
    (torch.int32, 2147483647.5, 2147483647),
]:
    matrix = torch.ones(1, 1, dtype=dtype)
    vector = matrix.flatten()
    batch = matrix.reshape(1, 1, 1)
    outputs = [
        matrix.addmm(matrix, matrix, beta=0, alpha=alpha),
        vector.addmv(matrix, vector, beta=0, alpha=alpha),
        matrix.addbmm(batch, batch, beta=0, alpha=alpha),
        batch.baddbmm(batch, batch, beta=0, alpha=alpha),
    ]
    for output in outputs:
        assert output.dtype == dtype
        assert output.flatten().tolist() == [expected]

matrix = torch.ones(1, 1, dtype=torch.int64)
try:
    matrix.addmm(matrix, matrix, beta=0, alpha=float(2**63))
except RuntimeError:
    pass
else:
    raise AssertionError("2^63 coefficient was accepted")

for dtype, lower, upper in [
    (torch.uint8, 0.0, 256.0),
    (torch.int8, -128.0, 128.0),
    (torch.int16, -32768.0, 32768.0),
    (torch.int32, -2147483648.0, 2147483648.0),
    (torch.int64, -float(2**63), float(2**63)),
]:
    matrix = torch.ones(1, 1, dtype=dtype)
    below = lower - 2048 if dtype == torch.int64 else lower - 0.5
    for invalid in [below, upper, float("nan"), float("inf"), -float("inf")]:
        for beta, alpha in [(0.0, invalid), (invalid, 0.0)]:
            try:
                matrix.addmm(matrix, matrix, beta=beta, alpha=alpha)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{dtype} accepted beta={beta}, alpha={alpha}")
    assert matrix.addmm(matrix, matrix, beta=0, alpha=lower).item() == int(lower)

for input_dtype, left_dtype, right_dtype, expected_dtype in [
    (torch.float32, torch.float32, torch.float64, torch.float64),
    (torch.float64, torch.float32, torch.float32, torch.float64),
    (torch.int32, torch.int64, torch.int64, torch.int64),
    (torch.int32, torch.float32, torch.float32, torch.float32),
]:
    input = torch.ones(1, 1, dtype=input_dtype)
    left = torch.ones(1, dtype=left_dtype)
    right = torch.ones(1, dtype=right_dtype)
    output = input.addr(left, right)
    assert output.dtype == expected_dtype
    assert output.tolist() == [[2]]

vector = torch.tensor([1 + 2**-23], dtype=torch.float32)
output = torch.zeros(1, 1, dtype=torch.float64).addr(vector, vector, beta=0)
assert output.item() == (1 + 2**-23) ** 2
scalar = torch.tensor(1.0, dtype=torch.float64)
vector = torch.ones(2, dtype=torch.float32)
assert scalar.addr(vector, vector).dtype == torch.float64
ignored = torch.full((2, 2), float("nan"), dtype=torch.float64)
assert ignored.addr(vector, vector, beta=0).tolist() == [[1.0, 1.0], [1.0, 1.0]]
assert scalar.addr(torch.ones(0, dtype=torch.float32), vector).shape == (0, 2)
print(f"M1 readiness reference checks passed (torch {torch.__version__})")
