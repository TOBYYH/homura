"""Check the reference behavior that tests/autograd_formulas.rs and tests/m1_ops.rs encode (M1
steps 4 and 6a), and the review fixes that tests/m1_regressions.rs encodes; does not write
fixtures.

Each block mirrors a test there. Where homura deliberately differs, the PyTorch behavior is
asserted and the difference is noted: messages name homura's dtypes and operations, half and
single precision sums of repeated positions are rounded once from f64, an out-of-range position
is reported even when `self` is empty, `index_add` rejects zero-dimensional `self` like homura's
`index_select`, tensors must share a device, the backward graph keeps one node per slice,
`as_strided` can't differentiate an input whose elements overlap, and `scalar / x` is one
division (Python computes `x.reciprocal() * scalar`).
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
    """Equal values, NaN equal to NaN and infinities only to themselves, within the tolerance of
    the Rust tests."""
    actual = actual.detach().flatten().tolist()
    assert len(actual) == len(expected), (actual, expected)
    for a, e in zip(actual, expected):
        if math.isnan(e):
            assert math.isnan(a), (actual, expected)
        elif math.isinf(e):
            assert a == e, (actual, expected)
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

# ---------------------------------------------------------------- step 6a: nodes
F = torch.nn.functional
inf, nan = math.inf, math.nan
x = leaf([1.0, 2, 3])
y = leaf([4.0, 5, 6])
constant = torch.tensor([1.0, 2, 3])
integers = torch.tensor([1, 2, 3])
cond = torch.tensor([True, False, True])
for out, name, edges in [
    (x.abs(), "AbsBackward0", [ACC]),
    (x.relu(), "ReluBackward0", [ACC]),
    (x.log(), "LogBackward0", [ACC]),
    (x.sqrt(), "SqrtBackward0", [ACC]),
    (x.rsqrt(), "RsqrtBackward0", [ACC]),
    (x.reciprocal(), "ReciprocalBackward0", [ACC]),
    (x.sin(), "SinBackward0", [ACC]),
    (x.cos(), "CosBackward0", [ACC]),
    (x.tanh(), "TanhBackward0", [ACC]),
    (x.sigmoid(), "SigmoidBackward0", [ACC]),
    (F.silu(x), "SiluBackward0", [ACC]),
    (F.gelu(x), "GeluBackward0", [ACC]),
    (F.gelu(x, approximate="tanh"), "GeluBackward0", [ACC]),
    (x / y, "DivBackward0", [ACC, ACC]),
    (x / 2.0, "DivBackward0", [ACC, None]),
    (x / 2, "DivBackward0", [ACC, None]),
    (x / constant, "DivBackward0", [ACC, None]),
    (x / integers, "DivBackward0", [ACC, None]),
    (integers / x, "DivBackward0", [None, ACC]),
    (torch.div(torch.tensor(2.0), x), "DivBackward0", [None, ACC]),
    (torch.where(cond, x, y), "WhereBackward0", [ACC, ACC]),
    (torch.where(cond, x, 0.0), "WhereBackward0", [ACC, None]),
    (torch.where(cond, 0.0, y), "WhereBackward0", [None, ACC]),
    (torch.where(cond, x, constant), "WhereBackward0", [ACC, None]),
]:
    assert node(out) == (name, edges), (node(out), name, edges)
# homura divides `2.0 / x` like torch.div above; Python's operator multiplies by the reciprocal.
assert node(2.0 / x) == ("MulBackward0", ["ReciprocalBackward0", None])
assert integers.abs().grad_fn is None and integers.relu().grad_fn is None

# ---------------------------------------------------------------- step 6a: values
X = [0.0, -0.0, -1.0, 2.0, inf, -inf, nan, 0.0, -2.0]
G = [1.0, 1.0, inf, nan, 1.0, 1.0, 1.0, inf, nan]
for f, expected in [
    (torch.abs, [0, 0, -inf, nan, 1, -1, 0, nan, nan]),
    (torch.relu, [0, 0, 0, nan, 1, 0, 1, 0, 0]),
    (torch.log, [inf, -inf, -inf, nan, 0, -0.0, nan, inf, nan]),
    (torch.sqrt, [inf, -inf, nan, nan, 0, nan, nan, inf, nan]),
    (torch.rsqrt, [-inf, inf, nan, nan, -0.0, nan, nan, -inf, nan]),
    (torch.reciprocal, [-inf, -inf, -inf, nan, -0.0, -0.0, nan, -inf, nan]),
    (torch.sin, [1, 1, inf, nan, nan, nan, nan, inf, nan]),
    (torch.cos, [-0.0, 0, inf, nan, nan, nan, nan, nan, nan]),
    (torch.tanh, [1, 1, inf, nan, 0, 0, nan, inf, nan]),
    (torch.sigmoid, [0.25, 0.25, inf, nan, 0, 0, nan, inf, nan]),
    (F.silu, [0.5, 0.5, inf, nan, nan, nan, nan, inf, nan]),
    (F.gelu, [0.5, 0.5, -inf, nan, nan, nan, nan, inf, nan]),
    (lambda x: F.gelu(x, approximate="tanh"), [0.5, 0.5, -inf, nan, nan, nan, nan, inf, nan]),
]:
    x = leaf(X)
    same(grad(f(x), x, G)[0], expected)
x = leaf([nan, 0.0, -0.0, 1e-30, -1e-30])
same(grad(x.relu(), x, [inf, inf, nan, inf, inf])[0], [inf, 0, 0, inf, 0])
for dtype in [torch.float16, torch.bfloat16]:
    x = leaf([0.25, 1.0, 4.0], dtype=dtype)
    (gx,) = torch.autograd.grad(x.sqrt(), x, torch.ones(3, dtype=dtype))
    assert gx.dtype == dtype and gx.tolist() == [1, 0.5, 0.25]

a = leaf([1.0, 0, inf, 1, -1, 2])
b = leaf([0.0, 0, inf, inf, -0.0, 4])
ga, gb = grad(a / b, [a, b], [1.0, 1, 1, 1, 1, inf])
same(ga, [inf, inf, 0, 0, -inf, inf])
same(gb, [-inf, nan, nan, -0.0, inf, -inf])
a = leaf([[1.0], [2]])
b = leaf([1.0, 2, 4])
ga, gb = grad(a / b, [a, b], [[1.0] * 3] * 2)
same(ga, [1.75, 1.75])
same(gb, [-3, -0.75, -0.1875])
x = leaf([0.0, 1, inf, -2])
for out in [2.0 / x, 2 / x, torch.div(torch.tensor(2.0), x)]:
    same(grad(out, x, [1.0] * 4)[0], [-inf, -2, -0.0, -0.5])
same(grad(x / 2.0, x, [1.0] * 4)[0], [0.5] * 4)
same(grad(x / 0.0, x, [1.0] * 4)[0], [inf] * 4)
x = leaf([0.0, 1, inf])
same(grad(0.0 / x, x, [1.0] * 3)[0], [nan, -0.0, -0.0])
x = leaf([1.0, 2, -4])
i = torch.tensor([2, -1, 4])
same(grad(x / i, x, [1.0] * 3)[0], [0.5, -1, 0.25])
same(grad(i / x, x, [1.0] * 3)[0], [-2, 0.25, -0.25])
a = leaf([1.0, 2])
b = leaf([3.0, 4], dtype=torch.float64)
ga, gb = grad(a / b, [a, b], [1.0, 1])
assert ga.dtype == torch.float32 and gb.dtype == torch.float64
same(ga, [1 / 3, 0.25])
same(gb, [-1 / 9, -0.125])

cond = torch.tensor([True, False, True, False])
x = leaf([1.0, 2, 3, 4])
y = leaf([5.0, 6, 7, 8])
gx, gy = grad(torch.where(cond, x, y), [x, y], [nan, inf, -inf, nan])
same(gx, [nan, 0, -inf, 0])
same(gy, [0, inf, 0, nan])
x = leaf([1.0, 2])
y = leaf(3.0)
gx, gy = grad(torch.where(torch.tensor([[True, False], [False, False]]), x, y), [x, y], [[1.0, 2], [3, 4]])
same(gx, [1, 0])
same(gy, [9])
cond = torch.tensor([[True], [False]])
x = leaf([1.0, 2])
same(grad(torch.where(cond, x, -1.0), x, [[1.0, 2], [3, 4]])[0], [1, 2])
same(grad(torch.where(cond, 0.5, x), x, [[1.0, 2], [3, 4]])[0], [3, 4])
x = leaf([1.0, 2])
y = leaf([3.0, 4], dtype=torch.float64)
gx, gy = grad(torch.where(torch.tensor([True, False]), x, y), [x, y], [1.0, 1])
assert gx.dtype == torch.float32 and gy.dtype == torch.float64
same(gx, [1, 0])
same(gy, [0, 1])

# ---------------------------------------------------------------- step 6a: higher order
for dtype in [torch.float32, torch.float64]:
    for name, f, first, second in [
        ("abs", torch.abs, [1, -1, 1], [0, 0, 0]),
        ("relu", torch.relu, [1, 0, 1], [0, 0, 0]),
        ("log", torch.log, [2, 0.8, 0.5], [-4, -0.64, -0.25]),
        ("sqrt", torch.sqrt, [0.70710677, 0.4472136, 0.35355338], [-0.70710677, -0.17888544, -0.08838835]),
        ("rsqrt", torch.rsqrt, [-1.4142135, -0.35777088, -0.1767767], [4.2426405, 0.42932505, 0.13258252]),
        ("reciprocal", torch.reciprocal, [-4, -0.64, -0.25], [16, -1.024, 0.25]),
        ("sin", torch.sin, [0.87758255, 0.31532237, -0.41614684], [-0.47942555, 0.9489846, -0.9092974]),
        ("cos", torch.cos, [-0.47942555, 0.9489846, -0.9092974], [-0.87758255, -0.31532237, 0.41614684]),
        ("tanh", torch.tanh, [0.78644773, 0.28041487, 0.07065082], [-0.72686198, 0.4757427, -0.13621869]),
        ("sigmoid", torch.sigmoid, [0.23500371, 0.17310479, 0.10499359], [-0.05755679, 0.09600387, -0.0799625]),
        ("silu", F.silu, [0.7399612, 0.006319155, 1.0907842], [0.441229, 0.22620474, 0.05006217]),
        ("gelu", F.gelu, [0.8674951, -0.12266158, 1.0852318], [0.6161143, 0.07990897, -0.10798193]),
        (
            "gelu with tanh",
            lambda x: F.gelu(x, approximate="tanh"),
            [0.8673699, -0.12249263, 1.0860993],
            [0.6155069, 0.08118098, -0.10787508],
        ),
    ]:
        positive = name in ("log", "sqrt", "rsqrt")
        x = leaf([0.5, 1.25 if positive else -1.25, 2.0], dtype=dtype)
        d1 = torch.autograd.grad(f(x).sum(), x, create_graph=True)[0]
        # Every first derivative depends on `x`, also those of `abs` (through `sgn`) and `relu`
        # (through `threshold_backward`), whose own derivatives are zero.
        assert d1.requires_grad, name
        d2 = torch.autograd.grad(d1.sum(), x)[0] + 0  # `+ 0` materializes PyTorch's zero tensors
        assert d1.dtype == d2.dtype == dtype
        same(d1, first)
        same(d2, second)

a = leaf([1.5, -2.0])
b = leaf([0.5, 4.0])
gb = torch.autograd.grad((a / b).sum(), b, create_graph=True)[0]
same(gb, [-6, 0.125])
gba, gbb = torch.autograd.grad(gb.sum(), [a, b])
same(gba, [-4, -0.0625])
same(gbb, [24, -0.0625])
ga = torch.autograd.grad((a / b).sum(), a, create_graph=True)[0]
gaa, gab = torch.autograd.grad(ga.sum(), [a, b], allow_unused=True)
assert gaa is None
same(gab, [-4, -0.0625])
x = leaf([0.5, -1.25, 2.0])
d1 = torch.autograd.grad((2.0 / x).sum(), x, create_graph=True)[0]
same(d1, [-8, -1.28, -0.5])
same(torch.autograd.grad(d1.sum(), x)[0], [32, -2.048, 0.5])
x = leaf([1.0, 2.0])
d1 = torch.autograd.grad(torch.where(torch.tensor([True, False]), x, 0.0).sum(), x, create_graph=True)[0]
assert not d1.requires_grad and d1.tolist() == [1, 0]

# ---------------------------------------------------------------- step 6a: saved tensors
x3 = lambda: leaf([1.0, 2, 3])  # noqa: E731
divisor = lambda: torch.tensor([2.0, 4, 8])  # noqa: E731
for make in [
    lambda: x3().abs().sum(),
    lambda: x3().relu().sum(),
    lambda: x3().log().sum(),
    lambda: x3().sqrt().sum(),
    lambda: x3().rsqrt().sum(),
    lambda: x3().reciprocal().sum(),
    lambda: x3().sin().sum(),
    lambda: x3().cos().sum(),
    lambda: x3().tanh().sum(),
    lambda: x3().sigmoid().sum(),
    lambda: F.silu(x3()).sum(),
    lambda: F.gelu(x3()).sum(),
    lambda: F.gelu(x3(), approximate="tanh").sum(),
    lambda: (x3() / x3()).sum(),
    lambda: (x3() / divisor()).sum(),
    lambda: (divisor() / x3()).sum(),
    lambda: (x3() / 2.0).sum(),
    lambda: (2.0 / x3()).sum(),
    lambda: torch.where(torch.tensor([True, False, True]), x3(), 0.0).sum(),
]:
    assert second_pass_fails(make)


def version_error(make, described):
    try:
        make()
    except RuntimeError as e:
        expected = (
            "one of the variables needed for gradient computation has been modified by an "
            f"inplace operation: {described} is at version 1; expected version 0 instead."
        )
        assert str(e).startswith(expected), str(e)
        return
    raise AssertionError("expected an error")


for requested in range(2):
    a = leaf([1.0, 2])
    b = leaf([3.0, 4])
    y = (a / b).sum()
    with torch.no_grad():
        a.mul_(1)
    version_error(lambda: torch.autograd.grad(y, [a, b][requested]), "[torch.FloatTensor [2]]")
a = leaf([1.0, 2])
b = torch.tensor([3.0, 4])
y = (a / b).sum()
with torch.no_grad():
    a.mul_(1)
y.backward()
y = (a / b).sum()
with torch.no_grad():
    b.mul_(1)
version_error(y.backward, "[torch.FloatTensor [2]]")
cond = torch.tensor([True, False])
y = torch.where(cond, leaf([1.0, 2]), 0.0).sum()
cond.copy_(torch.tensor([False, True]))
version_error(y.backward, "[CPUBoolType [2]]")  # homura: [bool [2]]
# Saved tensors with a history name the operation that computed them, not the node.
x = leaf([1.0, -2])
y = x.relu()
with torch.no_grad():
    y.mul_(1)
version_error(lambda: y.sum().backward(), "[torch.FloatTensor [2]], which is output 0 of Relu,")
a = x * 2.0
y = (a * a).sum()
with torch.no_grad():
    a.mul_(1)
version_error(y.backward, "[torch.FloatTensor [2]], which is output 0 of Mul,")

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

# ---------------------------------------------------------------- review fixes (m1_regressions.rs)
# A zero-dimensional scatter_add src gets a zero-dimensional gradient, summed by the engine.
for device in ["cpu", "meta"]:
    for self_shape in [(), (1,), (3,)]:
        for index_shape, hits in [((), 1.0), ((1,), 1.0), ((0,), 0.0)]:
            for dim, both in [(0, False), (0, True), (-1, False), (-1, True)]:
                x = torch.ones(self_shape, device=device, requires_grad=both)
                src = torch.full((), 3.0, device=device, requires_grad=True)
                index = torch.zeros(index_shape, dtype=torch.long, device=device)
                y = x.scatter_add(dim, index, src)
                w = torch.full(y.shape, 2.0, device=device, requires_grad=True)
                inputs = (src, x) if both else (src,)
                g = torch.autograd.grad(y, inputs, w, create_graph=True)
                assert g[0].shape == ()
                (gw,) = torch.autograd.grad(g[0], w)
                assert gw.shape == y.shape
                if both:
                    assert g[1].shape == self_shape
                if device == "cpu":
                    assert g[0].item() == 2 * hits
                    assert gw.flatten().tolist() == [hits] + [0.0] * (y.numel() - 1)
                    if both:
                        assert g[1].flatten().tolist() == [2.0] * x.numel()
    src = torch.ones(1, device=device, requires_grad=True)
    y = torch.ones(3, device=device).scatter_add(0, torch.zeros((), dtype=torch.long, device=device), src)
    assert message(lambda: y.sum().backward()) == (
        "Function ScatterAddBackward0 returned an invalid gradient at index 1 - got [] but expected "
        "shape compatible with [1]"
    )

# Meta as_strided gradients of shapes near the size limit.
for n in [1 << 20, (2**63 - 1) // 32]:
    x = torch.zeros(n, device="meta", requires_grad=True)
    every_other = x[::2]
    for y in [
        x.as_strided((n,), (1,), 0),
        x.as_strided((2, n - 1), (1, 1), 0),
        x.as_strided((n - 1,), (1,), 1),
        every_other.as_strided((n // 4,), (4,), 2),
    ]:
        u = torch.zeros(y.shape, device="meta", requires_grad=True)
        (g,) = torch.autograd.grad(y, x, u, create_graph=True)
        assert (g.shape, g.dtype, g.device.type) == ((n,), torch.float32, "meta")
        assert torch.autograd.grad(g.sum(), u)[0].shape == y.shape
# PyTorch supports an input whose elements overlap; homura rejects it (see the docstring above).
row = torch.zeros(2, device="meta", requires_grad=True)
row.expand(3, 2).as_strided((2,), (1,), 0).sum().backward()
assert row.grad.shape == (2,)

# f16 and bf16 products and quotients keep a scalar in f32; other operations round it first.
c = 2 / math.sqrt(math.pi)
inf = math.inf
ops = [
    lambda x: x * 1e5,
    lambda x: 1e5 * x,
    lambda x: x * 1e-8,
    lambda x: x * c,
    lambda x: x * 100000,
    lambda x: x / c,
    lambda x: x / 1e5,
    lambda x: x / 1e-8,
    lambda x: x.clone().mul_(1e5),
    lambda x: x.clone().div_(1e-8),
    lambda x: x - 1e5,
    lambda x: x + c,
    lambda x: x**c,
    lambda x: (x == 1.0001).float(),
]
expected = {
    torch.float16: [
        [0, inf, inf, inf],
        [0, inf, inf, inf],
        [0, 0.0006551742553710938, 5.960464477539063e-08, 0],
        [0, inf, 3.384765625, 1.1279296875],
        [0, inf, inf, inf],
        [0, 58048, 2.658203125, 0.88623046875],
        [0, 0.6552734375, 2.9981136322021484e-05, 1.0013580322265625e-05],
        [0, inf, inf, inf],
        [0, inf, inf, inf],
        [0, inf, inf, inf],
        [-inf, -inf, -inf, -inf],
        [1.1279296875, 65504, 4.12890625, 2.12890625],
        [0, inf, 3.453125, 1],
        [0, 0, 0, 1],
    ],
    torch.bfloat16: [
        [0, 6543114240, 299008, 99840],
        [0, 6543114240, 299008, 99840],
        [0, 0.0006561279296875, 3.003515303134918e-08, 1.0011717677116394e-08],
        [0, 73728, 3.390625, 1.125],
        [0, 6543114240, 299008, 99840],
        [0, 58112, 2.65625, 0.88671875],
        [0, 0.65625, 3.0040740966796875e-05, 1.0013580322265625e-05],
        [0, 6562710028288, 299892736, 100139008],
        [0, 6543114240, 299008, 99840],
        [0, 6562710028288, 299892736, 100139008],
        [-99840, -34304, -99840, -99840],
        [1.125, 65536, 4.125, 2.125],
        [0, 262144, 3.4375, 1],
        [0, 0, 0, 1],
    ],
}
quotients = {torch.float16: [inf, 49984, 24992, inf], torch.bfloat16: [inf, 49920, 24960, 199680]}
for dtype, rows in expected.items():
    x = torch.tensor([0, 65504, 3, 1], dtype=torch.float64).to(dtype)
    for i, (op, row) in enumerate(zip(ops, rows)):
        assert op(x).double().tolist() == row, (dtype, i, op(x).tolist())
    d = torch.tensor([0, 2, 4, 0.5], dtype=dtype)
    assert (1e5 / d).double().tolist() == quotients[dtype]
for dtype, products, divided in [
    (
        torch.float32,
        [3.3851375579833984, 7.898654460906982, 1.1283792206876573e-30],
        [2.6586806774139404, 6.203588008880615, 8.862269142205257e-31],
    ),
    (
        torch.float64,
        [3.385137501286538, 7.898654169668588, 1.1283791670955127e-30],
        [2.658680776358274, 6.203588478169306, 8.862269254527581e-31],
    ),
]:
    x = torch.tensor([3, 7, 1e-30], dtype=torch.float64).to(dtype)
    assert (x * c).double().tolist() == products and (x / c).double().tolist() == divided

# The scalar factor of a half precision product is kept in f32 in the backward pass too.
for dtype, product, erf in [
    (torch.float16, 3.384765625, [0.87890625, 0.70947265625, -0.04132080078125]),
    (torch.bfloat16, 3.390625, [0.87890625, 0.7109375, -0.041259765625]),
]:
    for scalar_first in [False, True]:
        x = torch.tensor([3.0], dtype=dtype, requires_grad=True)
        y = c * x if scalar_first else x * c
        u = torch.tensor([3.0], dtype=dtype, requires_grad=True)
        (g,) = torch.autograd.grad(y, x, u, create_graph=True)
        assert g.tolist() == [product]
        r = torch.tensor([3.0], dtype=dtype)
        assert torch.autograd.grad((g * r).sum(), u)[0].tolist() == [product]
    x = torch.tensor([3.0], dtype=dtype, requires_grad=True)
    assert torch.autograd.grad(x * 1e5, x, torch.zeros_like(x))[0].tolist() == [0]
    x = torch.tensor([0.5, -1.25, 2.0], dtype=dtype, requires_grad=True)
    upstream = torch.tensor([1.0, 3.0, -2.0], dtype=dtype)
    assert torch.autograd.grad(x.erf(), x, upstream)[0].tolist() == erf

# gradcheck compares like torch.allclose: equal values (also equal infinities) are close,
# otherwise both must be finite and within atol + rtol * |numerical|; NaN never is.
for a, n, expected in [
    (1.0, 1.0, True),
    (inf, inf, True),
    (-inf, -inf, True),
    (-inf, inf, False),
    (0.0, inf, False),
    (1e308, inf, False),
    (inf, 1e308, False),
    (math.nan, math.nan, False),
    (math.nan, 1.0, False),
    (1.0, math.nan, False),
    (0.0, 0.5e-5, True),
    (0.0, 2e-5, False),
    (1.0005, 1.0, True),
    (1.002, 1.0, False),
    (1e308, -1e308, False),
]:
    t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    assert torch.allclose(t(a), t(n), rtol=1e-3, atol=1e-5) == expected, (a, n)
x = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
assert not torch.autograd.gradcheck(lambda x: x.detach() * 1e308, (x,), eps=1.0, raise_exception=False)
# PyTorch's checks don't pass in no-grad mode, not even for a correct function: gradcheck fails
# and gradgradcheck raises. homura's turn grad mode on for the check instead.
x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
with torch.no_grad():
    assert not torch.autograd.gradcheck(lambda x: x * x, (x,), raise_exception=False)
    try:
        torch.autograd.gradgradcheck(lambda x: x * x, (x,), raise_exception=False)
        raise AssertionError("expected an error")
    except TypeError:
        pass
    assert not torch.is_grad_enabled()
assert torch.autograd.gradcheck(lambda x: x * x, (x,))
assert torch.autograd.gradgradcheck(lambda x: x * x, (x,))
assert not torch.autograd.gradgradcheck(lambda x: x * x.detach(), (x,), raise_exception=False)

# relu's gradient has zero derivatives with respect to the result and to x, also for non-finite
# gradients in the second pass, and selects again with respect to the upstream gradient (R32).
for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    x = torch.tensor([-1.0, 0, 1, nan, inf, -inf], dtype=dtype, requires_grad=True)
    y = x.relu()
    u = torch.tensor([1.0, 2, 3, 4, 5, 6], dtype=dtype, requires_grad=True)
    (dx,) = torch.autograd.grad(y, x, u, create_graph=True)
    assert dx.requires_grad
    same(dx, [0, 0, 3, 4, 5, 0])
    gx, gy, gu = torch.autograd.grad(dx, [x, y, u], torch.tensor([inf, nan, 2, inf, nan, -inf], dtype=dtype))
    same(gx + 0, [0] * 6)  # `+ 0` materializes PyTorch's zero tensors
    same(gy + 0, [0] * 6)
    same(gu, [0, 0, 2, inf, nan, 0])
x = leaf([-1.0, 0, 1])
y = x.relu()
(dx,) = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)
same(torch.autograd.grad(dx.sum(), y)[0] + 0, [0, 0, 0])

# Half precision gelu gradients are computed in f32 and rounded once, so they stay finite for
# large finite inputs (R33). Native kernels: on this CPU the default dispatch sends the exact form
# to oneDNN, which rejects half precision backward passes (its f32 results differ slightly too).
X = [-300.0, -10, -1.25, 0, 0.5, 2, 10, 300, inf, -inf, nan]
TAIL = [1, 1, nan, nan, nan]
with torch.backends.mkldnn.flags(enabled=False):
    for dtype, approximate, head in [
        (torch.float16, "tanh", [0, 0, -0.12249755859375, 0.5, 0.8671875, 1.0859375]),
        (torch.float16, "none", [0, -0.0, -0.1226806640625, 0.5, 0.86767578125, 1.0849609375]),
        (torch.bfloat16, "tanh", [0, 0, -0.12255859375, 0.5, 0.8671875, 1.0859375]),
        (torch.bfloat16, "none", [0, -7.709323308994218e-22, -0.12255859375, 0.5, 0.8671875, 1.0859375]),
        (torch.float32, "tanh", [0, 0, -0.12249262630939484, 0.5, 0.8673698902130127, 1.0860992670059204]),
        (torch.float32, "none", [0, -7.694599251752704e-22, -0.12266159057617188, 0.5, 0.8674951195716858, 1.0852317810058594]),
        (torch.float64, "tanh", [0, 0, -0.12249263224964863, 0.5, 0.8673699035346424, 1.0860992566236183]),
        (torch.float64, "none", [0, -7.694598626706419e-22, -0.12266158306942215, 0.5, 0.8674951246561629, 1.085231801078197]),
    ]:
        x = leaf(X, dtype=dtype)
        y = F.gelu(x, approximate=approximate)
        (g,) = torch.autograd.grad(y, x, torch.ones_like(y))
        assert g.dtype == dtype
        same(g, head + TAIL)
    for approximate, second in [
        ("tanh", [0, 0.08118106424808502, 0.6155069470405579, -0.10787512362003326, 0]),
        ("none", [0, 0.07990896701812744, 0.6161143183708191, -0.10798193514347076, 0]),
    ]:
        for dtype in [torch.float32, torch.float16]:
            x = leaf([-300.0, -1.25, 0.5, 2, 300], dtype=dtype)
            u = torch.ones(5, dtype=dtype, requires_grad=True)
            (d1,) = torch.autograd.grad(F.gelu(x, approximate=approximate), x, u, create_graph=True)
            d2x, d2u = torch.autograd.grad(d1, [x, u], torch.ones(5, dtype=dtype))
            same(d2u, d1.tolist())
            if dtype == torch.float32:
                same(d2x, second)
            else:
                # PyTorch's own half precision second derivative, computed step by step, overflows
                # at +-300; homura differentiates the f32 computation and gets 0 there.
                assert math.isnan(d2x[0]) and math.isnan(d2x[4]), d2x
    x = leaf([0.5, -1.25], dtype=torch.float16)
    y = F.gelu(x, approximate="tanh").sum()
    with torch.no_grad():
        x.mul_(1)
    version_error(y.backward, "[torch.HalfTensor [2]]")  # homura: [f16 [2]]
    assert second_pass_fails(lambda: F.gelu(leaf([0.5, -1.25], dtype=torch.bfloat16)).sum())

print("check_m1_formulas: all reference checks passed")
