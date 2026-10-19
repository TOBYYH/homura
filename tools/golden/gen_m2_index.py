"""Generates tests/golden/m2_index.safetensors: advanced indexing (x[...] with index tensors and
masks) and index_put (M2 step 6), with PyTorch's outputs and gradients. tests/index_advanced.rs
reads the file.

Case "<name>.<dtype>" stores the input "x", the index tensors "i0", "i1", ... of the expression
that CASES gives for <name> (integer tensors and bool masks, as given), for an index_put case
(name starting with "put_") the values "v", the upstream gradient "g", the output "out", and the
first derivatives "grad_x" (and "grad_v") from torch.autograd.grad(out, inputs, g). Cases of
SECOND (f64 only) also store "gg_x" (and "gg_v") and the second derivatives "grad2_<variable>":
the gradients of sum(grad_x * gg_x [+ grad_v * gg_v]) with respect to x, v and g, with the first
derivatives computed with create_graph and g as a variable too; a variable whose second
derivative is None has no entry. Half precision cases avoid repeated positions, which PyTorch
sums with a rounding after each addition and homura in f64 with one rounding.

Usage: python tools/golden/gen_m2_index.py (needs torch and safetensors).
"""

from pathlib import Path

import torch
from safetensors.torch import save_file

import reference

OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "m2_index.safetensors"

reference.check()
torch.manual_seed(0)
DTYPES = {
    "f64": torch.float64,
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def t(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype)


def mask(values):
    return torch.tensor(values, dtype=torch.bool)


# name -> (shape of x, index tensors, expression building the index from them, values shape for
# index_put cases or None, accumulate). The expressions are mirrored by `spec` in
# tests/index_advanced.rs.
CASES = {}


def get(name, shape, indices, expr):
    CASES[name] = (shape, indices, expr, None, False)


def put(name, shape, indices, expr, values_shape, accumulate):
    CASES[name] = (shape, indices, expr, values_shape, accumulate)


I3 = t([2, 0, 3])
J3 = t([1, 4, 0])
M4 = mask([True, False, True, True])
M45 = mask([[True, False, False, True, False], [False, True, False, False, False],
            [False, False, False, False, True], [True, True, False, False, False]])

get("get_one", (4, 3), [I3], lambda i: (i[0],))
get("get_repeated_negative", (4, 3), [t([[1, -1], [1, 0]])], lambda i: (i[0],))
get("get_adjacent", (3, 4, 5, 2), [I3, J3], lambda i: (slice(None), i[0], i[1]))
get("get_non_adjacent", (4, 3, 5, 2), [I3, J3], lambda i: (i[0], slice(None), i[1]))
get("get_int_between", (3, 4, 2, 5), [I3, J3], lambda i: (slice(None), i[0], 1, i[1]))
get("get_full_slice_between", (3, 4, 2, 5), [I3, J3],
    lambda i: (slice(None), i[0], slice(0, 2), i[1]))
get("get_new_axis_between", (3, 4, 5), [I3, J3], lambda i: (slice(None), i[0], None, i[1]))
get("get_zero_width_ellipsis", (3, 4, 5), [I3, J3], lambda i: (slice(None), i[0], ..., i[1]))
get("get_ellipsis_between", (4, 3, 2, 5), [I3, J3], lambda i: (i[0], ..., i[1]))
get("get_broadcast", (4, 5, 3), [t([[0], [3]]), J3], lambda i: (i[0], i[1]))
get("get_after_slice", (4, 3, 5), [J3], lambda i: (slice(1, 4), slice(None), i[0]))
get("get_i32", (2, 4, 5), [t([3, 1], torch.int32), t([[0, 4]], torch.int32)],
    lambda i: (slice(None), i[0], i[1]))
get("get_mask", (4, 3), [M4], lambda i: (i[0],))
get("get_mask_2d", (4, 5, 2), [M45], lambda i: (i[0],))
get("get_mask_later", (3, 4, 5), [M45], lambda i: (slice(None), i[0]))
get("get_mask_then_slice", (4, 5, 2, 3), [M45, t([2, 0, 1, 1, 2, 0])],
    lambda i: (i[0], slice(None), i[1]))
get("get_mask_broadcast", (4, 4, 3), [t([[1], [2]]), M4], lambda i: (i[0], i[1]))
get("get_mask_all_false", (4, 3), [mask([False] * 4)], lambda i: (i[0],))
get("get_true", (3, 2), [], lambda i: (True,))
get("get_false", (3, 2), [], lambda i: (False,))
get("get_int_and_true", (4, 3), [I3], lambda i: (i[0], True))
get("get_scalar_int", (4, 3), [t(2)], lambda i: (i[0],))
get("get_scalar_int_then_tensor", (4, 3, 5), [t(-1), J3], lambda i: (slice(None), i[0], i[1]))
get("get_scalar_target_true", (), [], lambda i: (True,))
get("get_empty_index", (3, 4, 2), [t([], torch.int64)], lambda i: (slice(None), i[0]))
get("get_empty_other_dim", (0, 4, 2), [J3.clamp(max=3)], lambda i: (slice(None), i[0]))

for accumulate in (False, True):
    a = "_acc" if accumulate else ""
    put("put_one" + a, (5, 3), [t([4, 0, 2])], lambda i: (i[0],), (3, 3), accumulate)
    put("put_adjacent" + a, (3, 4, 5), [I3, J3], lambda i: (slice(None), i[0], i[1]),
        (3, 3), accumulate)
    put("put_non_adjacent" + a, (4, 3, 5), [I3, J3], lambda i: (i[0], slice(None), i[1]),
        (3, 3), accumulate)
    put("put_broadcast_values" + a, (4, 3, 5), [I3, J3], lambda i: (i[0], slice(None), i[1]),
        (1, 3), accumulate)
    put("put_scalar_values" + a, (4, 3), [I3], lambda i: (i[0],), (), accumulate)
    put("put_mask_scalar" + a, (4, 5, 2), [M45], lambda i: (i[0],), (), accumulate)
    put("put_mask_rows" + a, (4, 5, 2), [M45], lambda i: (i[0],), (6, 2), accumulate)
    put("put_ellipsis" + a, (3, 2, 4), [t([3, 0])], lambda i: (..., i[0]), (2,), accumulate)
    put("put_scalar_index" + a, (4,), [t(2)], lambda i: (i[0],), (), accumulate)
    put("put_i32" + a, (4, 5), [t([1, 3], torch.int32)], lambda i: (slice(None), i[0]),
        (4, 1), accumulate)
    put("put_empty" + a, (4, 3), [t([], torch.int64)], lambda i: (i[0],), (3,), accumulate)
put("put_repeated_acc", (4, 3), [t([[1, 3], [1, 1]])], lambda i: (i[0],), (2, 2, 3), True)

HALF = ["get_non_adjacent", "get_mask_2d", "get_true", "put_non_adjacent", "put_mask_rows",
        "put_non_adjacent_acc"]
SECOND = ["get_adjacent", "get_non_adjacent", "get_repeated_negative", "get_mask_then_slice",
          "get_true", "get_scalar_int_then_tensor", "get_empty_index", "put_non_adjacent",
          "put_non_adjacent_acc", "put_broadcast_values", "put_mask_scalar",
          "put_repeated_acc", "put_scalar_index"]


def evaluate(name, x, indices, v):
    _, _, expr, _, accumulate = CASES[name]
    spec = expr(indices)
    if v is None:
        return x[spec]
    # index_put takes tensors only: a full slice is None in ATen's list, and the ellipsis
    # expands to as many.
    entries = []
    for item in spec:
        if item is ...:
            entries += [None] * (x.dim() - sum(1 for e in spec if e is not ...))
        elif isinstance(item, slice):
            entries.append(None)
        else:
            entries.append(item)
    return torch.ops.aten.index_put(x, entries, v, accumulate)


def generate():
    data = {}

    def store(case, name, value):
        key = f"{case}/{name}"
        assert key not in data, key
        data[key] = value.detach().contiguous().clone()

    for name, (shape, indices, _, values_shape, _) in CASES.items():
        dtypes = ["f64", "f32"] + (["f16", "bf16"] if name in HALF else [])
        for dname in dtypes:
            dtype = DTYPES[dname]
            case = f"{name}.{dname}"
            x = torch.randn(shape, dtype=torch.float64).to(dtype).requires_grad_()
            v = None
            if values_shape is not None:
                v = torch.randn(values_shape, dtype=torch.float64).to(dtype).requires_grad_()
            out = evaluate(name, x, indices, v)
            g = torch.randn(out.shape, dtype=torch.float64).to(dtype).requires_grad_()
            variables = [x] + ([v] if v is not None else [])
            names = ["x"] + (["v"] if v is not None else [])
            second = name in SECOND and dname == "f64"
            grads = torch.autograd.grad(out, variables, g, create_graph=second)
            store(case, "x", x)
            for k, index in enumerate(indices):
                store(case, f"i{k}", index)
            if v is not None:
                store(case, "v", v)
            store(case, "g", g)
            store(case, "out", out)
            for n, gr in zip(names, grads):
                store(case, f"grad_{n}", gr)
            if not second:
                continue
            ggs = [torch.randn(gr.shape, dtype=torch.float64) for gr in grads]
            total = sum((gr * gg).sum() for gr, gg in zip(grads, ggs))
            seconds = torch.autograd.grad(total, variables + [g], allow_unused=True)
            for n, gg in zip(names, ggs):
                store(case, f"gg_{n}", gg)
            for n, s in zip(names + ["g"], seconds):
                if s is not None:
                    store(case, f"grad2_{n}", s)
    return data


if __name__ == "__main__":
    save_file(generate(), str(OUT), metadata=reference.metadata())
    print(f"wrote {OUT}")
