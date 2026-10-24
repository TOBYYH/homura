"""Check the reference behavior that tests/tensor_sum.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: Python's `sum` and `math.prod` over tensors,
which homura's `Sum` / `Product` for `Tensor` follow. The start value is a plain int, so the first
operation is `tensor + 0` (`tensor * 1`) with a weak scalar: the dtype and device come from the
tensors. Then the dtype matrix (every dtype, zero- and one-dimensional), the left-to-right order,
mixed dtypes, the meta device, the autograd nodes and gradients, inputs left unchanged, and the
empty case. Python's empty result is an int; homura returns a zero-dimensional I64 tensor on the
default device instead, which has no counterpart here beyond the value.
"""

import math

import torch

import reference

reference.check()
passed = 0


def block(f):
    global passed
    f()
    passed += 1


DTYPES = [
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
]


@block
def dtypes_of_sums_and_products():
    # A single tensor: the dtype stays, except that bool becomes int64; a tensor start would
    # promote zero-dimensional integer tensors to int64 instead.
    for dtype in DTYPES:
        for shape in [(), (2,)]:
            x = torch.full(shape, 3, dtype=dtype) if dtype != torch.bool else torch.ones(shape, dtype=dtype)
            expected = torch.int64 if dtype == torch.bool else dtype
            for fn in [sum, math.prod]:
                y = fn([x])
                assert y.dtype == expected and y.shape == x.shape, (dtype, shape, fn)
                assert torch.equal(y, x.to(expected))
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8) and shape == ():
                assert (torch.tensor(0) + x).dtype == torch.int64
    # Three tensors: [2, 3, 4] (bool: [true, true, false]).
    for dtype in DTYPES:
        if dtype == torch.bool:
            items = [torch.tensor(v) for v in [True, True, False]]
            assert sum(items).item() == 2 and math.prod(items).item() == 0
            assert sum(items).dtype == math.prod(items).dtype == torch.int64
        else:
            items = [torch.tensor(v, dtype=dtype) for v in [2, 3, 4]]
            assert sum(items).item() == 9 and math.prod(items).item() == 24
            assert sum(items).dtype == math.prod(items).dtype == dtype
    # Integer sums wrap in the inputs' dtype.
    i8 = [torch.tensor(100, dtype=torch.int8)] * 2
    assert sum(i8).dtype == torch.int8 and sum(i8).item() == -56
    assert math.prod(i8).item() == 16


@block
def sums_run_from_left_to_right():
    big = torch.tensor(65504.0, dtype=torch.float16)
    assert sum([big, big, -big]).item() == math.inf
    assert sum([-big, big, big]).item() == 65504.0


@block
def mixed_dtypes_promote_step_by_step():
    def check(items, dtype, values):
        y = sum(items)
        assert y.dtype == dtype and y.tolist() == values, (y, dtype, values)

    i = lambda v, d: torch.tensor(v, dtype=d)
    check([i([1, 2], torch.int32), i(0.5, torch.float32)], torch.float32, [1.5, 2.5])
    check([i(1, torch.int64), i([2], torch.float16)], torch.float16, [3.0])
    check([i([1], torch.float32), i(2, torch.float64)], torch.float32, [3.0])
    check([i([1], torch.int32), i(2, torch.int64)], torch.int32, [3])
    check([i([True], torch.bool), i(2, torch.int8)], torch.int64, [3])
    y = math.prod([i([2], torch.int16), i(1.5, torch.float64)])
    assert y.dtype == torch.float64 and y.tolist() == [3.0]


@block
def meta_inputs_keep_their_device():
    m = torch.ones(2, device="meta")
    for fn in [sum, math.prod]:
        y = fn([m, m])
        assert y.device.type == "meta" and y.shape == (2,)


@block
def nodes_and_gradients():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    for fn, name in [(sum, "AddBackward0"), (math.prod, "MulBackward0")]:
        y = fn([a])
        assert type(y.grad_fn).__name__ == name
        assert [type(f).__name__ for f, _ in y.grad_fn.next_functions] == ["AccumulateGrad", "NoneType"]
        assert y is not a and y.item() == 2.0
    sum([a, b, a]).backward()
    assert (a.grad.item(), b.grad.item()) == (2.0, 1.0)
    a.grad = b.grad = None
    math.prod([a, b]).backward()
    assert (a.grad.item(), b.grad.item()) == (3.0, 2.0)


@block
def inputs_are_unchanged():
    x = torch.tensor([1.0, 2.0])
    for fn in [sum, math.prod]:
        y = fn([x])
        y.add_(10)
        assert x.tolist() == [1.0, 2.0] and x._version == 0


@block
def empty_iterators():
    assert sum([]) == 0 and type(sum([])) is int
    assert math.prod([]) == 1 and type(math.prod([])) is int


print(f"{passed} blocks passed")
