"""Check the reference behavior that tests/autograd_formulas.rs and tests/m1_ops.rs encode (M1
step 4); does not write fixtures.

Each block mirrors a test there. Where homura deliberately differs, the PyTorch behavior is
asserted and the difference is noted: messages name homura's dtypes and operations, half and
single precision sums of repeated positions are rounded once from f64, an out-of-range position
is reported even when `self` is empty, `index_add` rejects zero-dimensional `self` like homura's
`index_select`, tensors must share a device, and the backward graph keeps one node per slice.
"""

import math

import torch

import reference

reference.check()


def message(f):
    try:
        f()
    except Exception as e:  # noqa: BLE001 - any error type carries the message we compare
        return str(e)
    raise AssertionError("expected an error")


def leaf(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype, requires_grad=True)


def grad(out, inputs, g):
    return torch.autograd.grad(out, inputs, torch.tensor(g, dtype=out.dtype))


def same(actual, expected):
    """Equal values, NaN equal to NaN, within the tolerance of the Rust tests."""
    actual = actual.detach().flatten().tolist()
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected):
        if math.isnan(e):
            assert math.isnan(a), (actual, expected)
        else:
            assert abs(a - e) <= 1e-6 * max(abs(e), 1.0), (actual, expected)


def node(t):
    edges = [None if f is None else f.name() for f, _ in t.grad_fn.next_functions]
    return t.grad_fn.name(), edges


def chain(t):
    names, f = [], t.grad_fn
    while f is not None:
        names.append(f.name())
        f = f.next_functions[0][0] if f.next_functions else None
    return names


ACC = "torch::autograd::AccumulateGrad"

# ---------------------------------------------------------------- nodes
x = leaf([[1.0, 2, 3], [4, 5, 6]])
index = torch.tensor([1, 0])
other = torch.zeros(3, 3, requires_grad=True)
for out, name, edges in [
    (torch.erf(x), "ErfBackward0", [ACC]),
    (torch.sign(x), "SignBackward0", [ACC]),
    (x.select(0, 1), "SelectBackward0", [ACC]),
    (x.narrow(1, 1, 2), "SliceBackward0", [ACC]),
    (x[1:], "SliceBackward0", [ACC]),
    (x.as_strided([2], [1], 1), "AsStridedBackward0", [ACC]),
    (x.index_select(1, index), "IndexSelectBackward0", [ACC]),
    (
        torch.zeros(3, 3).narrow(0, 0, 2).index_add(1, torch.tensor([1, 0, 2]), x),
        "IndexAddBackward0",
        [None, ACC],
    ),
    (
        other.narrow(0, 0, 2).index_add(1, torch.tensor([1, 0, 2]), x),
        "IndexAddBackward0",
        ["SliceBackward0", ACC],
    ),
    (x.gather(1, torch.tensor([[0], [2]])), "GatherBackward0", [ACC]),
    (
        other.scatter_add(1, torch.tensor([[0, 1, 2], [2, 1, 0]]), x),
        "ScatterAddBackward0",
        [ACC, ACC],
    ),
    (other.select_scatter(x.select(0, 1), 1, 0), "SelectScatterBackward0", [ACC, "SelectBackward0"]),
    (other.slice_scatter(x, 0, 1, None, 1), "SliceScatterBackward0", [ACC, ACC]),
]:
    assert node(out) == (name, edges), (node(out), name, edges)
assert chain(x[index]) == ["IndexBackward0", ACC]  # homura adds UnsafeViewBackward0 on top
# homura builds split and chunk from narrow, one SliceBackward0 per piece.
assert [t.grad_fn.name() for t in x.split(2, 1)] == ["SplitBackward0"] * 2
# homura records every slice; PyTorch skips full ones among several indices.
assert chain(x[:, 1:3]) == ["SliceBackward0", ACC]
assert torch.sign(torch.tensor([1, -2])).grad_fn is None

# ---------------------------------------------------------------- values
x = leaf([0.0, 0.5, -1.0, math.inf, math.nan, 30.0])
same(grad(torch.erf(x), x, [1.0, 2, 3, 4, 5, 6])[0], [1.1283792, 1.7575653, 1.2453226, 0, math.nan, 0])
x = leaf([-2.0, 0.0, 3.0, math.nan])
same(grad(torch.sign(x), x, [1.0, math.inf, math.nan, 2.0])[0], [0, 0, 0, 0])

x = leaf([[1.0, 2], [3, 4], [5, 6]])
g = [[1.0, 2], [3, 4], [5, 6]]
same(grad(x.index_select(0, torch.tensor([2, 0, 2])), x, g)[0], [3, 4, 0, 0, 6, 8])
same(grad(x.index_select(1, torch.tensor([1, 1], dtype=torch.int32)), x, g)[0], [0, 3, 0, 7, 0, 11])
same(grad(x.index_select(0, torch.tensor(1)), x, [[7.0, 8]])[0], [0, 0, 7, 8, 0, 0])
same(grad(x[torch.tensor([-1, 0, -1])], x, g)[0], [3, 4, 0, 0, 6, 8])
y = x[:, torch.tensor([[1, 0], [1, 1]])]
same(torch.autograd.grad(y, x, torch.arange(12.0).view(3, 2, 2))[0], [1, 5, 5, 17, 9, 29])

x = leaf([[1.0, 2, 3], [4, 5, 6]])
same(grad(x.select(1, -1), x, [7.0, 8])[0], [0, 0, 7, 0, 0, 8])
same(grad(x[:, ::2], x, [[1.0, 2], [3, 4]])[0], [1, 0, 2, 3, 0, 4])
same(grad(x[-5:1, 1:100], x, [[1.0, 2]])[0], [0, 1, 2, 0, 0, 0])
same(grad(x.narrow(1, 1, 2), x, [[1.0, 2], [3, 4]])[0], [0, 1, 2, 0, 3, 4])
same(grad(x.split(2, 1)[1], x, [[1.0], [2]])[0], [0, 0, 1, 0, 0, 2])
same(grad(x[1:, ::2], x, [[1.0, 2]])[0], [0, 0, 0, 1, 0, 2])

base = leaf([float(i) for i in range(10)])
view = base.narrow(0, 2, 6)
same(grad(view.as_strided([3], [2], 1), base, [1.0, 2, 3])[0], [0, 0, 0, 2, 0, 3, 0, 0, 0, 0])
same(grad(view.as_strided([3], [2], 1), view, [1.0, 2, 3])[0], [0, 2, 0, 3, 0, 0])
y = base.as_strided([2, 2], [1, 1], 0)
same(grad(y, base, [[1.0, 2], [3, 4]])[0], [1, 5, 4, 0, 0, 0, 0, 0, 0, 0])
same(grad(base.as_strided([3], [0], 4), base, [1.0, 2, 3])[0], [0, 0, 0, 0, 6, 0, 0, 0, 0, 0])
y = base[1::3].as_strided([2, 2], [3, 1], 1)
same(grad(y, base, [[1.0, 2], [3, 4]])[0], [0, 1, 0, 0, 3, 0, 0, 0, 0, 0])
same(torch.autograd.grad(base.as_strided([0], [1], 3).sum(), base)[0], [0] * 10)
# PyTorch divides among overlapping input elements; homura doesn't support those yet.
row = leaf([1.0, 2.0])
e = row.expand(3, 2)
same(torch.autograd.grad(e.as_strided([2], [1], 0).sum(), e)[0], [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])

source = leaf([[1.0, 2], [3, 4], [5, 6]])
x = leaf([[0.0, 0], [0, 0]])
gx, gs = grad(x.index_add(0, torch.tensor([1, 0, 1]), source), [x, source], [[1.0, 2], [3, 4]])
same(gx, [1, 2, 3, 4])
same(gs, [3, 4, 1, 2, 3, 4])
x = leaf([[1.0, 2, 3], [4, 5, 6]])
same(grad(x.gather(1, torch.tensor([[2, 2], [0, 1]])), x, [[1.0, 2], [3, 4]])[0], [0, 0, 3, 3, 4, 0])
x = leaf([[0.0, 0, 0], [0, 0, 0]])
src = leaf([[1.0, 2], [3, 4]])
gx, gs = grad(x.scatter_add(1, torch.tensor([[2, 2], [0, 1]]), src), [x, src], [[1.0, 2, 3], [4, 5, 6]])
same(gx, [1, 2, 3, 4, 5, 6])
same(gs, [3, 3, 4, 5])
src = leaf([[0.0] * 4] * 3)
y = torch.zeros(3, 4).scatter_add(1, torch.tensor([[0], [1], [2]]), src)
assert message(y.sum().backward) == (
    "Function ScatterAddBackward0 returned an invalid gradient at index 1 - got [3, 1] but "
    "expected shape compatible with [3, 4]"
)

x = leaf([[1.0, 2, 3], [4, 5, 6]])
g = [[1.0, 2, 3], [4, 5, 6]]
src = leaf([7.0, 8], dtype=torch.float64)
gx, gs = grad(x.select_scatter(src, 1, 1), [x, src], g)
same(gx, [1, 0, 3, 4, 0, 6])
assert gs.dtype == torch.float64 and gs.tolist() == [2, 5]
src = leaf([[7.0], [8]])
gx, gs = grad(x.slice_scatter(src, 1, 2, None, 1), [x, src], g)
same(gx, [1, 2, 0, 4, 5, 0])
same(gs, [3, 6])
src = leaf([[7.0, 8], [9, 10]])
gx, gs = grad(x.slice_scatter(src, 1, None, None, 2), [x, src], g)
same(gx, [0, 2, 0, 0, 5, 0])
same(gs, [1, 3, 4, 6])

# ---------------------------------------------------------------- higher order
for dtype in [torch.float32, torch.float64]:
    x = leaf([0.5, -1.25], dtype=dtype)
    d1 = torch.autograd.grad(torch.erf(x).sum(), x, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), x)[0]
    assert d1.dtype == d2.dtype == dtype
    same(d1, [0.87878263, 0.23652114])
    same(d2, [-0.87878263, 0.5913029])
x = leaf([1.0, -2.0])
d1 = torch.autograd.grad(torch.sign(x).sum(), x, create_graph=True)[0]
assert not d1.requires_grad and d1.tolist() == [0, 0]

# The linear formulas give `d<g, r>/du = f(r)` for the functions of the closure test.
x = leaf([[1.0, -2, 3], [0.5, 4, -1]])
u = torch.arange(6.0).view(2, 3) - 1.5
u.requires_grad_(True)
y = torch.zeros(2, 4).scatter_add(1, torch.tensor([[3, 0, 3], [1, 1, 2]]), x)
u4 = (torch.arange(8.0).view(2, 4) - 1.5).requires_grad_(True)
g = torch.autograd.grad(y, x, u4, create_graph=True)[0]
r = torch.arange(6.0).view(2, 3) + 0.5
gu = torch.autograd.grad((g * r).sum(), u4)[0]
same(gu, torch.zeros(2, 4).scatter_add(1, torch.tensor([[3, 0, 3], [1, 1, 2]]), r).flatten().tolist())

# ---------------------------------------------------------------- saved tensors
def second_pass_fails(make):
    y = make()
    y.backward()
    try:
        y.backward()
    except RuntimeError as e:
        assert str(e).startswith("Trying to backward through the graph a second time")
        return True
    return False


idx = torch.tensor([1, 0])
x3 = lambda: leaf([1.0, 2, 3])  # noqa: E731
m = lambda: leaf([[1.0, 2], [3, 4], [5, 6]])  # noqa: E731
for make, saves in [
    (lambda: torch.erf(x3()).sum(), True),
    (lambda: torch.sign(x3()).sum(), False),
    (lambda: x3().index_select(0, idx).sum(), True),
    (lambda: x3()[idx].sum(), True),
    (lambda: m().select(0, 1).sum(), False),
    (lambda: m()[1:].sum(), False),
    (lambda: m().narrow(0, 1, 2).sum(), False),
    (lambda: x3().as_strided([2], [2], 0).sum(), False),
    (lambda: x3().index_add(0, idx, torch.ones(2)).sum(), False),
    (lambda: torch.zeros(3).index_add(0, idx, leaf([1.0, 2])).sum(), True),
    (lambda: x3().gather(0, idx).sum(), True),
    (lambda: x3().scatter_add(0, idx, torch.ones(2)).sum(), False),
    (lambda: torch.zeros(3).scatter_add(0, idx, leaf([1.0, 2])).sum(), True),
    (lambda: m().select_scatter(leaf([1.0, 2]), 0, 1).sum(), False),
    (lambda: m().slice_scatter(leaf([[1.0, 2]]), 0, 1, 2, 1).sum(), False),
]:
    assert second_pass_fails(make) == saves


def index_modified(make):
    i = torch.tensor([1, 0])
    y = make(i)
    with torch.no_grad():
        i.add_(0)
    try:
        y.sum().backward()
    except RuntimeError as e:
        assert str(e).startswith(
            "one of the variables needed for gradient computation has been modified by an "
            "inplace operation: [torch.LongTensor [2]] is at version 1; expected version 0 "
            "instead."
        ), str(e)
        return True
    return False


src = lambda: leaf([1.0, 2])  # noqa: E731
for make, saved in [
    (lambda i: x3().index_select(0, i), True),
    (lambda i: x3()[i], True),
    (lambda i: x3().gather(0, i), True),
    (lambda i: torch.zeros(3).index_add(0, i, src()), True),
    (lambda i: x3().index_add(0, i, torch.ones(2)), False),
    (lambda i: torch.zeros(3).scatter_add(0, i, src()), True),
    (lambda i: x3().scatter_add(0, i, torch.ones(2)), False),
]:
    assert index_modified(make) == saved

# ---------------------------------------------------------------- forward: erf and sign
for dtype, erf_dtype in [
    (torch.int64, torch.float32),
    (torch.int32, torch.float32),
    (torch.uint8, torch.float32),
    (torch.bool, torch.float32),
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float64, torch.float64),
]:
    t = torch.zeros(2, dtype=dtype)
    assert torch.erf(t).dtype == erf_dtype and torch.sign(t).dtype == dtype
s = torch.sign(torch.tensor([math.nan, -0.0, 0.0, -2.5, math.inf, -math.inf]))
assert s.tolist() == [0, 0, 0, -1, 1, -1] and not torch.signbit(s[:3]).any()
assert torch.sign(torch.tensor([-128, 0, 127], dtype=torch.int8)).tolist() == [-1, 0, 1]
assert torch.sign(torch.tensor([0, 255], dtype=torch.uint8)).tolist() == [0, 1]
assert torch.sign(torch.tensor([True, False])).tolist() == [True, False]
e = torch.erf(torch.tensor([math.nan, math.inf, -math.inf, -0.0]))
assert math.isnan(e[0]) and e[1:].tolist() == [1, -1, 0] and torch.signbit(e[3])

# ---------------------------------------------------------------- forward: index_add
x, src, idx = torch.zeros(3, 4), torch.ones(2, 4), torch.tensor([0, 2])
assert message(lambda: x.index_add(0, torch.tensor([[0, 2]]), src)).startswith(
    "index_add_(): Index is supposed to be a vector"
)
assert message(lambda: x.index_add(0, idx.float(), src)) == (
    "index_add_(): Expected dtype int32/int64 for index but got: Float"
)
assert message(lambda: x.index_add(0, idx.short(), src)) == (
    "index_add_(): Expected dtype int32/int64 for index but got: Short"
)
# homura rejects a zero-dimensional self, like its index_select.
assert torch.tensor(1.0).index_add(0, torch.tensor([0]), torch.tensor(5.0)).item() == 6
assert message(lambda: x.index_add(2, idx, src)) == (
    "Dimension out of range (expected to be in range of [-2, 1], but got 2)"
)
assert message(lambda: x.index_add(0, idx, src.double())) == (
    "index_add_(): self (Float) and source (Double) must have the same scalar type"
)
assert message(lambda: x.index_add(0, torch.tensor([0, 1, 2]), src)) == (
    "index_add_(): Number of indices (3) should be equal to source.size(dim): (2), for dim: 0"
)
assert message(lambda: x.index_add(1, idx, torch.ones(2))) == (
    "index_add_(): Indexing dim 1 is out of bounds of the source tensor with dim 1"
)
for bad in [torch.ones(2, 3), torch.ones(2), torch.ones(2, 1)]:
    assert message(lambda: x.index_add(0, idx, bad)) == (
        "source tensor shape must match self tensor shape, excluding the specified dimension. "
        f"Got self.shape = [3, 4] source.shape = {list(bad.shape)}"
    )
assert message(lambda: x.index_add(0, torch.tensor([3, 0]), src)) == (
    "index 3 is out of bounds for dimension 0 with size 3"
)
assert message(lambda: x.index_add(0, torch.tensor([-1, 0]), src)) == (
    "index -1 is out of bounds for dimension 0 with size 3"
)
# homura reports the position; PyTorch skips the check for an empty self.
assert torch.zeros(0, 4).index_add(0, torch.tensor([0]), torch.ones(1, 4)).shape == (0, 4)

y = torch.zeros(3, 2).index_add(0, torch.tensor([2, 0, 2]), torch.tensor([[1.0, 2], [3, 4], [5, 6]]))
assert y.flatten().tolist() == [3, 4, 0, 0, 6, 8]
assert torch.zeros(3).index_add(0, torch.tensor(1, dtype=torch.int32), torch.ones(1)).tolist() == [0, 1, 0]
h = torch.tensor([2048.0, 1, 1, 1, 1], dtype=torch.float16)
# homura rounds once in both cases; PyTorch rounds after every addition in one dimension.
assert torch.zeros(1, dtype=torch.float16).index_add(0, torch.zeros(5, dtype=torch.long), h).tolist() == [2048]
y = torch.zeros(1, 1, dtype=torch.float16).index_add(0, torch.zeros(5, dtype=torch.long), h.view(5, 1))
assert y.flatten().tolist() == [2052]
b = torch.tensor([[256.0], [1], [1]], dtype=torch.bfloat16)
y = torch.zeros(1, 1, dtype=torch.bfloat16).index_add(0, torch.zeros(3, dtype=torch.long), b)
assert y.flatten().tolist() == [258]
# homura sums single precision in f64 and gets 1.
y = torch.zeros(1, 1).index_add(0, torch.zeros(3, dtype=torch.long), torch.tensor([[1e8], [1.0], [-1e8]]))
assert y.flatten().tolist() == [0]
u8 = torch.tensor([250], dtype=torch.uint8).index_add(
    0, torch.tensor([0, 0]), torch.tensor([3, 4], dtype=torch.uint8)
)
assert u8.tolist() == [1]
y = torch.tensor([False, False, True]).index_add(0, torch.tensor([0, 0, 2]), torch.tensor([True, False, False]))
assert y.tolist() == [True, False, True]
x = torch.arange(6.0).view(2, 3).t()
y = x.index_add(1, torch.tensor([1, 1]), torch.arange(12.0).view(3, 4)[:, ::2])
assert y.flatten().tolist() == [0, 5, 1, 14, 2, 23]
m = torch.zeros(3, 4, device="meta").index_add(0, torch.tensor([7, 9], device="meta"), torch.ones(2, 4, device="meta"))
assert m.shape == (3, 4)
# homura requires one device; PyTorch's message differs.
assert message(lambda: torch.zeros(3, 4, device="meta").index_add(0, idx, src.to("meta"))).startswith(
    "index_add(): self, index and source expected to be in the same device"
)

# ---------------------------------------------------------------- forward: gather and scatter_add
x, z = torch.arange(12.0).view(3, 4), torch.zeros(3, 4)
assert message(lambda: x.gather(1, torch.tensor([[0.0]]))) == "gather(): Expected dtype int32/int64 for index"
assert message(lambda: x.gather(1, torch.tensor([0, 1]))) == (
    "Index tensor must have the same number of dimensions as input tensor"
)
assert message(lambda: x.gather(0, torch.zeros(1, 5, dtype=torch.long))) == (
    "Size does not match at dimension 1 expected index [1, 5] to be no larger than self [3, 4] "
    "apart from dimension 0"
)
assert message(lambda: x.gather(1, torch.tensor([[4]]))) == "index 4 is out of bounds for dimension 1 with size 4"
assert message(lambda: x.gather(1, torch.tensor([[-1]]))) == (
    "index -1 is out of bounds for dimension 1 with size 4"
)
assert message(lambda: torch.zeros(3, 0).gather(1, torch.zeros(3, 2, dtype=torch.long))) == (
    "index 0 is out of bounds for dimension 1 with size 0"
)
assert message(lambda: torch.tensor(1.0).gather(1, torch.tensor(0))) == (
    "Dimension out of range (expected to be in range of [-1, 0], but got 1)"
)
assert message(lambda: z.scatter_add(1, torch.tensor([[True]]), torch.ones(1, 1))) == (
    "scatter(): Expected dtype int32/int64 for index"
)
assert message(lambda: z.scatter_add(1, torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.float64))) == (
    "scatter(): Expected self.dtype to be equal to src.dtype"
)
assert message(lambda: z.scatter_add(1, torch.tensor([0]), torch.ones(1, 1))) == (
    "Index tensor must have the same number of dimensions as self tensor"
)
# homura names the src tensor; PyTorch fails on a dimension of it.
assert message(lambda: z.scatter_add(1, torch.tensor([[0]]), torch.ones(1))).startswith("Dimension out of range")
assert message(lambda: z.scatter_add(1, torch.tensor([[0, 1]]), torch.ones(1, 1))) == (
    "Expected index [1, 2] to be no larger than self [3, 4] apart from dimension 1 and to be no "
    "larger size than src [1, 1]"
)
assert message(lambda: z.scatter_add(1, torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))) == (
    "Expected index [4, 1] to be no larger than self [3, 4] apart from dimension 1 and to be no "
    "larger size than src [4, 1]"
)
assert message(lambda: z.scatter_add(1, torch.tensor([[4]]), torch.ones(1, 1))) == (
    "index 4 is out of bounds for dimension 1 with size 4"
)
assert message(lambda: z.scatter_add(1, torch.tensor([[-1]]), torch.ones(1, 1))) == (
    "index -1 is out of bounds for dimension 1 with size 4"
)

assert x.gather(1, torch.tensor([[0, 1, 2, 3, 0, 1]])).flatten().tolist() == [0, 1, 2, 3, 0, 1]
assert x.gather(0, torch.tensor([[2], [0]], dtype=torch.int32)).flatten().tolist() == [8, 0]
y = torch.zeros(3, 4).scatter_add(1, torch.tensor([[0], [1], [2]]), x)
assert y.flatten().tolist() == [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0]
y = torch.zeros(3, 4).scatter_add(1, torch.zeros(1, 6, dtype=torch.long), torch.ones(1, 6))
assert y.flatten().tolist()[:4] == [6, 0, 0, 0]
y = torch.zeros(1, dtype=torch.float16).scatter_add(0, torch.zeros(5, dtype=torch.long), h)
assert y.tolist() == [2052]
assert torch.tensor([1, 2]).scatter_add(0, torch.tensor([1, 1]), torch.tensor([10, 20])).tolist() == [1, 32]
y = torch.tensor([False, False]).scatter_add(0, torch.tensor([1, 1]), torch.tensor([True, True]))
assert y.tolist() == [False, True]
index = torch.tensor([[0, 2], [1, 3], [1, 0]]).t()
assert x.t().gather(0, index).flatten().tolist() == [0, 5, 9, 2, 7, 8]

y = torch.tensor(5.0).gather(0, torch.tensor([0, 0]))
assert y.shape == (2,) and y.tolist() == [5, 5]
y = torch.tensor([5.0, 6.0]).gather(0, torch.tensor(1))
assert y.dim() == 0 and y.item() == 6
assert torch.tensor(1.0).gather(-1, torch.tensor(0)).item() == 1
y = torch.tensor(1.0).scatter_add(0, torch.tensor([0, 0]), torch.tensor([5.0, 6.0]))
assert y.dim() == 0 and y.item() == 12
assert torch.zeros(3).scatter_add(0, torch.tensor(1), torch.tensor(5.0)).tolist() == [0, 5, 0]
assert torch.ones(3, 4).gather(1, torch.zeros(3, 0, dtype=torch.long)).shape == (3, 0)
y = torch.ones(3, 4).scatter_add(1, torch.zeros(3, 0, dtype=torch.long), torch.ones(3, 2))
assert y.flatten().tolist() == [1] * 12
mi = torch.zeros(2, 2, dtype=torch.long, device="meta")
assert torch.zeros(3, 4, device="meta").gather(1, mi).shape == (2, 2)
assert torch.zeros(3, 4, device="meta").scatter_add(0, mi, torch.zeros(2, 3, device="meta")).shape == (3, 4)

# ---------------------------------------------------------------- forward: select_scatter and slice_scatter
x = torch.zeros(3, 4)
assert message(lambda: torch.tensor(1.0).select_scatter(torch.tensor(5.0), 0, 0)) == (
    "select() cannot be applied to a 0-dim tensor."
)
assert message(lambda: x.select_scatter(torch.ones(4), 0, 3)) == (
    "select(): index 3 out of range for tensor of size [3, 4] at dimension 0"
)
assert message(lambda: x.select_scatter(torch.ones(4), 0, -4)) == (
    "select(): index -4 out of range for tensor of size [3, 4] at dimension 0"
)
assert message(lambda: x.select_scatter(torch.ones(4), 2, 1)) == (
    "Dimension out of range (expected to be in range of [-2, 1], but got 2)"
)
for bad in [torch.ones(3), torch.ones(1)]:
    assert message(lambda: x.select_scatter(bad, 0, 1)) == (
        "expected src to have a size equal to the slice of self. src size = "
        f"{list(bad.shape)}, slice size = [4]"
    )
assert message(lambda: torch.tensor(1.0).slice_scatter(torch.tensor(5.0), 0)) == (
    "slice() cannot be applied to a 0-dim tensor."
)
assert message(lambda: x.slice_scatter(torch.ones(0, 4), 0, 0, 0, 0)) == "slice step must be positive"
assert message(lambda: x.slice_scatter(torch.ones(1, 4), 0, 0, 1, -1)) == "slice step must be positive"
assert message(lambda: x.slice_scatter(torch.ones(3, 4), 0, 1, 3)) == (
    "expected src to have a size equal to the slice of self. src size = [3, 4], slice size = [2, 4]"
)
# homura requires one device; PyTorch copies across devices here.
assert x.to("meta").select_scatter(torch.ones(4), 0, 1).shape == (3, 4)

x = torch.arange(12.0).view(3, 4)
y = x.select_scatter(torch.tensor([-1.0, -2, -3]), -1, -1)
assert y.flatten().tolist() == [0, 1, 2, -1, 4, 5, 6, -2, 8, 9, 10, -3]
y = x.t().select_scatter(torch.zeros(3), 0, 0)
assert y.flatten().tolist() == [0, 0, 0, 1, 5, 9, 2, 6, 10, 3, 7, 11]
y = torch.zeros(3, 4, dtype=torch.long).select_scatter(torch.full((4,), 2.7), 0, 1)
assert y[1].tolist() == [2, 2, 2, 2]
assert torch.zeros(2, dtype=torch.bool).select_scatter(torch.tensor(3.0), 0, 1).tolist() == [False, True]
row = torch.zeros(5)
assert row.slice_scatter(torch.ones(2), 0, -2).tolist() == [0, 0, 0, 1, 1]
assert row.slice_scatter(torch.ones(2), 0, 3, 100).tolist() == [0, 0, 0, 1, 1]
assert row.slice_scatter(torch.ones(5), 0, -100, 100).tolist() == [1] * 5
assert row.slice_scatter(torch.ones(0), 0, 4, 2).tolist() == [0] * 5
assert row.slice_scatter(torch.ones(1), 0, 1, None, 2**62).tolist() == [0, 1, 0, 0, 0]
y = x.slice_scatter(torch.zeros(3, 2), -1, None, None, 2)
assert y.flatten().tolist() == [0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11]
assert torch.zeros(3, 0).select_scatter(torch.ones(0), 0, 1).shape == (3, 0)
m = torch.zeros(3, 4, device="meta").slice_scatter(torch.ones(2, 4, device="meta"), 0, 1, 3)
assert m.shape == (3, 4)

# ---------------------------------------------------------------- forward: in-place variants
x = torch.zeros(3, 2)
x.index_add_(0, torch.tensor([2, 2]), torch.ones(2, 2))
x.scatter_add_(1, torch.tensor([[1], [1], [0]]), torch.ones(3, 1))
assert x.flatten().tolist() == [0, 1, 0, 1, 3, 2]
base = torch.zeros(2, 3)
base.narrow(1, 1, 2).index_add_(1, torch.tensor([0]), torch.ones(2, 1))
assert base.flatten().tolist() == [0, 1, 0, 0, 1, 0]
leaf_ = torch.zeros(3, requires_grad=True)
assert message(lambda: leaf_.index_add_(0, torch.tensor([0]), torch.ones(1))) == (
    "a leaf Variable that requires grad is being used in an in-place operation."
)
with torch.no_grad():
    leaf_.index_add_(0, torch.tensor([1]), torch.ones(1))
assert leaf_.tolist() == [0, 1, 0]

print("check_m1_formulas: all reference checks passed")
