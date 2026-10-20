"""Check the reference behavior that tests/nn_pad_unfold.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: the node that F.pad (constant mode) records
for every pad, its values, layouts and storage, its argument errors, how it converts the fill
value (only when something is filled), meta and empty inputs; then Tensor.unfold: its view
layout and nodes, its argument errors and edge cases, the gradients it gives (overlapping
windows add up, in the dtype and in window order), the node of the gradient's own derivative,
and in-place writes into its result, which PyTorch allows even where windows overlap (homura
doesn't, a documented difference).
"""

import warnings

import torch
import torch.nn.functional as F

import reference

reference.check()
warnings.simplefilter("ignore")
d64 = torch.float64
INF, NAN = float("inf"), float("nan")


def raises(exc, message, f):
    """Asserts that f raises exc with the first line message (any message if None)."""
    try:
        f()
    except exc as e:
        assert message is None or str(e).splitlines()[0] == message, (str(e), message)
        return
    raise AssertionError(f"no {exc.__name__}: {message}")


def chain(t):
    """The node names of t's graph, depth first, as (depth, name)."""
    out = []

    def walk(fn, depth):
        if fn is None:
            return
        out.append((depth, type(fn).__name__))
        for nf, _ in fn.next_functions:
            walk(nf, depth + 1)

    walk(t.grad_fn, 0)
    return out


def leaf(*shape, dtype=d64):
    n = 1
    for s in shape:
        n *= s
    return (torch.arange(n, dtype=dtype) + 1).reshape(shape).clone().requires_grad_()


def bits(t):
    """The values of a float tensor with the sign of zeros, as Python floats with repr."""
    return [repr(v) for v in t.flatten().tolist()]


acc = "AccumulateGrad"
pad_node = [(0, "ConstantPadNdBackward0"), (1, acc)]

# pad_records_constant_pad_nd
x = leaf(2, 3)
for pad in [(1, 2), (-1, 0), (0, 0), (-1, 2, 1, -1), (), (0, 0, 0, 0)]:
    assert chain(F.pad(x, pad)) == pad_node, pad
assert chain(F.pad(x, (1, 1), value=2.5)) == pad_node
g = torch.ones(2, 6, dtype=d64, requires_grad=True)
(gx,) = torch.autograd.grad(F.pad(x, (1, 2)), x, g, create_graph=True)
assert chain(gx) == [(0, "ConstantPadNdBackward0"), (1, acc)]
assert gx.tolist() == [[1.0] * 3] * 2
with torch.no_grad():
    assert F.pad(x, (1, 1)).grad_fn is None
assert F.pad(torch.ones(2, dtype=torch.int64), (1, 1)).grad_fn is None

# pad_values_layouts_and_storage
x = torch.arange(6, dtype=d64).reshape(2, 3)
assert F.pad(x, (1, 2)).tolist() == [[0, 0, 1, 2, 0, 0], [0, 3, 4, 5, 0, 0]]
assert F.pad(x, (0, 0, 1, 0), value=-1.0).tolist() == [[-1, -1, -1], [0, 1, 2], [3, 4, 5]]
assert F.pad(x, (-1, 0)).tolist() == [[1, 2], [4, 5]]
assert F.pad(x, (-1, 2, 1, -1), value=7.0).tolist() == [[7, 7, 7, 7], [1, 2, 7, 7]]
assert F.pad(x, (-2, -1)).shape == (2, 0)
assert F.pad(x, (0, 0, -2, 0)).shape == (0, 3)
assert F.pad(x, (-3, 0, 1, 1)).shape == (4, 0)
s = torch.tensor(1.5, dtype=d64)
assert F.pad(s, ()).shape == () and F.pad(s, ()).item() == 1.5
# Every result is new storage; a result without filling is the clone of the cropped input, with
# its dense strides, a result with filling is contiguous.
t = torch.arange(6, dtype=d64).reshape(2, 3).t()
assert t.stride() == (1, 3)
for pad, stride in [((0, 0), (1, 3)), ((-1, 0), (1, 3)), ((0, 0, -1, 0), (1, 2)), ((1, 0), (3, 1)),
                    ((0, 0, 1, 0), (2, 1))]:
    y = F.pad(t, pad)
    assert y.stride() == stride, (pad, y.stride())
    assert y.untyped_storage().data_ptr() != t.untyped_storage().data_ptr()
y = F.pad(x, (0, 0))
y.add_(1)
assert x[0, 0].item() == 0.0

# pad_checks_its_arguments
x = torch.ones(2, 3)
raises(RuntimeError, "Padding length must be divisible by 2", lambda: F.pad(x, (1,)))
raises(RuntimeError, "Padding length must be divisible by 2", lambda: F.pad(x, (1, 1, 1)))
raises(RuntimeError, "Padding length should be less than or equal to two times the input "
       "dimension but got padding length 6 and input of dimension 2",
       lambda: F.pad(x, (1, 1, 1, 1, 1, 1)))
raises(RuntimeError, "Padding length should be less than or equal to two times the input "
       "dimension but got padding length 2 and input of dimension 0",
       lambda: F.pad(torch.tensor(1.0), (1, 1)))
# A crop is checked before the padding of the other side is added: (-4, 2) fails for size 3.
for pad in [(-2, -2), (-4, 0), (-4, 2), (0, -4), (0, 0, -1, -2), (5, -9)]:
    raises(RuntimeError, "narrow(): length must be non-negative.", lambda: F.pad(x, pad))
# The length is checked before anything else, the crops dimension by dimension from the first.
raises(RuntimeError, "Padding length must be divisible by 2", lambda: F.pad(x, (-9, 0, 1)))
raises(RuntimeError, "narrow(): length must be non-negative.",
       lambda: F.pad(torch.ones(2, 3, dtype=torch.uint8), (-4, 0), value=1e10))

# pad_converts_the_value_only_when_filling
assert F.pad(torch.ones(2, dtype=torch.int64), (1, 1), value=1.5).tolist() == [1, 1, 1, 1]
assert F.pad(torch.ones(2, dtype=torch.int64), (1, 0), value=-2.7).tolist() == [-2, 1, 1]
raises(RuntimeError, "value cannot be converted to type uint8_t without overflow",
       lambda: F.pad(torch.ones(2, dtype=torch.uint8), (1, 1), value=300))
raises(RuntimeError, "value cannot be converted to type c10::Half without overflow",
       lambda: F.pad(torch.ones(2, dtype=torch.float16), (1, 0), value=1e5))
raises(RuntimeError, "value cannot be converted to type uint8_t without overflow",
       lambda: F.pad(torch.ones(0, 2, dtype=torch.uint8), (1, 0), value=1e10))
for pad in [(0, 0), (-1, 0), ()]:
    assert F.pad(torch.ones(2, dtype=torch.uint8), pad, value=1e10).dtype == torch.uint8
assert F.pad(torch.zeros(2, dtype=torch.bool), (1, 0), value=2).tolist() == [True, False, False]
assert F.pad(torch.ones(2, dtype=torch.bool), (1, 0), value=0).tolist() == [False, True, True]
y = F.pad(torch.ones(2), (1, 0), value=NAN)
assert y.isnan().tolist() == [True, False, False]
assert bits(F.pad(torch.ones(1), (1, 0), value=-0.0)) == ["-0.0", "1.0"]
assert F.pad(torch.ones(1, dtype=torch.float16), (1, 0), value=INF).tolist() == [INF, 1.0]

# pad_on_meta_and_empty_inputs
m = torch.ones(2, 3, device="meta")
assert F.pad(m, (1, 1)).shape == (2, 5) and F.pad(m, (1, 1)).device.type == "meta"
assert F.pad(m, (-1, 0, 2, 2)).shape == (6, 2)
assert F.pad(torch.ones(2, dtype=torch.uint8, device="meta"), (1, 1), value=1e10).shape == (4,)
raises(RuntimeError, "narrow(): length must be non-negative.", lambda: F.pad(m, (-4, 0)))
ml = torch.ones(2, 3, device="meta", requires_grad=True)
(gm,) = torch.autograd.grad(F.pad(ml, (1, -1)).sum(), ml)
assert gm.device.type == "meta" and gm.shape == (2, 3)
assert F.pad(torch.ones(0, 3), (1, 1, 1, 1)).shape == (2, 5)
assert F.pad(torch.ones(0, 3), (1, 1, 1, 1)).tolist() == [[0.0] * 5] * 2
e = torch.ones(2, 0, dtype=d64, requires_grad=True)
y = F.pad(e, (1, 2))
assert y.tolist() == [[0.0] * 3] * 2
(ge,) = torch.autograd.grad(y.sum(), e)
assert ge.shape == (2, 0)

# pad_gradients_crop_the_upstream_gradient
x = leaf(2, 3)
g = torch.tensor([[1.0, INF, 2.0, 3.0], [NAN, 4.0, 5.0, -6.0]], dtype=d64)
(gx,) = torch.autograd.grad(F.pad(x, (-1, 2)), x, g)
assert bits(gx) == ["0.0", "1.0", "inf", "0.0", "nan", "4.0"]
(gx,) = torch.autograd.grad(F.pad(x, (0, 0)), x, torch.full((2, 3), -0.0, dtype=d64))
assert bits(gx) == ["-0.0"] * 6

print("pad: all checks passed")

# unfold_is_a_view
x = leaf(2, 5)
u = x.unfold(1, 3, 2)
assert u.shape == (2, 2, 3) and u.stride() == (5, 2, 1) and u._is_view() and u._base is x
assert chain(u) == [(0, "UnfoldBackward0"), (1, acc)]
assert u.tolist() == [[[1, 2, 3], [3, 4, 5]], [[6, 7, 8], [8, 9, 10]]]
t = torch.arange(12.0).reshape(3, 4).t()[1:]
v = t.unfold(1, 2, 1)
assert v.shape == (3, 2, 2) and v.stride() == (1, 4, 4) and v.storage_offset() == 1
assert v._base is t._base
u2 = leaf(4, 4).unfold(0, 2, 1).unfold(1, 2, 1)
assert u2.shape == (3, 3, 2, 2) and u2.stride() == (4, 1, 4, 1)
assert chain(u2) == [(0, "UnfoldBackward0"), (1, "UnfoldBackward0"), (2, acc)]
assert x.unfold(1, 5, 3).shape == (2, 1, 5) and x.unfold(1, 5, 3).stride() == (5, 3, 1)
# (homura's test of a step whose product with the stride overflows has no reference: PyTorch's
# int64 strides wrap.)
with torch.no_grad():
    assert x.unfold(1, 2, 1).grad_fn is None

# unfold_checks_its_arguments
x = torch.ones(2, 5)
raises(RuntimeError, "maximum size for tensor at dimension 1 is 5 but size is 6",
       lambda: x.unfold(1, 6, 1))
raises(RuntimeError, "maximum size for tensor at dimension 1 is 5 but size is 6",
       lambda: x.unfold(-1, 6, 0))
raises(RuntimeError, "step is 0 but must be > 0", lambda: x.unfold(1, 2, 0))
raises(IndexError, "Dimension out of range (expected to be in range of [-2, 1], but got 2)",
       lambda: x.unfold(2, 2, 1))
raises(IndexError, "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
       lambda: torch.tensor(1.0).unfold(1, 1, 1))
raises(RuntimeError, "maximum size for tensor at dimension 0 is 1 but size is 2",
       lambda: torch.tensor(3.0).unfold(0, 2, 1))

# unfold_edge_cases
x = torch.ones(2, 3)
assert x.unfold(1, 0, 1).shape == (2, 4, 0) and x.unfold(1, 0, 1).stride() == (3, 1, 1)
assert x.unfold(1, 0, 2).shape == (2, 2, 0)
assert x.unfold(-2, 1, 1).shape == (2, 3, 1) and x.unfold(-2, 1, 1).stride() == (3, 1, 3)
s = torch.tensor(3.0)
for step in [1, 5]:
    assert s.unfold(0, 1, step).shape == (1,) and s.unfold(-1, 1, step).stride() == (1,)
    assert s.unfold(0, 1, step).tolist() == [3.0]
assert s.unfold(0, 0, 1).shape == (0,)
assert torch.arange(5).unfold(0, 2, 2).tolist() == [[0, 1], [2, 3]]
assert torch.ones(2, 5, dtype=torch.bool).unfold(1, 3, 1).shape == (2, 3, 3)
m = torch.ones(2, 5, device="meta").unfold(1, 3, 2)
assert m.shape == (2, 2, 3) and m.stride() == (5, 2, 1) and m.device.type == "meta"
assert torch.ones(2, 0).unfold(1, 0, 1).shape == (2, 1, 0)
assert torch.ones(0, 5).unfold(1, 3, 1).shape == (0, 3, 3)
assert torch.ones(0, 5).unfold(0, 0, 3).shape == (1, 5, 0)

# unfold_gradients
x = leaf(2, 5)
(g,) = torch.autograd.grad(x.unfold(1, 3, 1).sum(), x)
assert g.tolist() == [[1, 2, 3, 2, 1]] * 2
(g,) = torch.autograd.grad(x.unfold(1, 2, 2).sum(), x)
assert g.tolist() == [[1, 1, 1, 1, 0]] * 2
(g,) = torch.autograd.grad(x.unfold(1, 1, 3).sum(), x)
assert g.tolist() == [[1, 0, 0, 1, 0]] * 2
e = torch.tensor([2.0], dtype=d64, requires_grad=True)
(g,) = torch.autograd.grad(e.expand(4).unfold(0, 2, 1).sum(), e)
assert g.tolist() == [6.0]
base = torch.arange(12.0, dtype=d64).reshape(4, 3).requires_grad_()
t = base.t()[1:]
assert t.stride() == (1, 3) and t.storage_offset() == 1
w = torch.arange(1.0, 1 + 2 * 3 * 2, dtype=d64).reshape(2, 3, 2)
(g,) = torch.autograd.grad((t.unfold(1, 2, 1) * w).sum(), base)
assert g.tolist() == [[0, 1, 7], [0, 5, 17], [0, 9, 21], [0, 6, 12]]
# Upstream non-finite values reach only the covered positions.
x = leaf(1, 4)
gin = torch.tensor([[[INF, 1.0], [NAN, 2.0]]], dtype=d64)
(g,) = torch.autograd.grad(x.unfold(1, 2, 2), x, gin)
assert bits(g) == ["inf", "1.0", "nan", "2.0"]
(g,) = torch.autograd.grad(x.unfold(1, 2, 1).narrow(1, 0, 2), x, gin)
assert bits(g) == ["inf", "nan", "2.0", "0.0"]
# Windows that don't overlap copy the gradient (keeping -0.0), overlapping ones add it to zero.
z = torch.full((1, 2, 2), -0.0, dtype=d64)
assert bits(torch.autograd.grad(x.unfold(1, 2, 2), x, z)[0]) == ["-0.0"] * 4
z = torch.full((1, 3, 2), -0.0, dtype=d64)
assert bits(torch.autograd.grad(x.unfold(1, 2, 1), x, z)[0]) == ["0.0"] * 4
# Half precision adds the windows in the dtype, one at a time in window order: position 2
# gets 2048 + 1 + 1, rounded to 2048 after each addition, position 4 gets 1 + 1 + 2048 = 2050.
xh = torch.zeros(7, dtype=torch.float16, requires_grad=True)
gh = torch.zeros(5, 3, dtype=torch.float16)
gh[0, 2], gh[1, 1], gh[2, 0] = 2048, 1, 1
gh[2, 2], gh[3, 1], gh[4, 0] = 1, 1, 2048
(g,) = torch.autograd.grad(xh.unfold(0, 3, 1), xh, gh)
assert g.tolist() == [0, 0, 2048, 0, 2050, 0, 0], g.tolist()
# In F32 the same windows give 2050 at position 2.
assert torch.autograd.grad(xh.float().unfold(0, 3, 1), xh, gh.float())[0].tolist()[2] == 2050
# Zero-size windows and zero-dimensional inputs.
x = leaf(2, 3)
(g,) = torch.autograd.grad(x.unfold(-1, 0, 1), x, torch.ones(2, 4, 0, dtype=d64))
assert g.tolist() == [[0.0] * 3] * 2
gin = torch.ones(2, 4, 0, dtype=d64, requires_grad=True)
(g,) = torch.autograd.grad(x.unfold(-1, 0, 1), x, gin, create_graph=True)
assert chain(g) == [(0, "UnfoldBackwardBackward0"), (1, acc)]
m = torch.ones(2, 3, device="meta", requires_grad=True)
um = m.unfold(-1, 0, 1)
assert um.shape == (2, 4, 0) and um.stride() == (3, 1, 1)
(g,) = torch.autograd.grad(um.sum(), m)
assert g.device.type == "meta" and g.shape == (2, 3)
for size, step in [(2, 1), (2, 2)]:
    (g,) = torch.autograd.grad(m.unfold(1, size, step).sum(), m)
    assert g.device.type == "meta" and g.shape == (2, 3)
s = torch.tensor(3.0, dtype=d64, requires_grad=True)
(g,) = torch.autograd.grad(s.unfold(0, 1, 4), s, torch.tensor([5.0], dtype=d64))
assert g.shape == () and g.item() == 5.0
(g,) = torch.autograd.grad(s.unfold(0, 0, 1).sum() + s * 0, s)
assert g.item() == 0.0

# unfold_backward_records_its_own_node
x = leaf(2, 5)
gin = torch.ones(2, 3, 3, dtype=d64, requires_grad=True)
(g,) = torch.autograd.grad(x.unfold(1, 3, 1), x, gin, create_graph=True)
assert chain(g) == [(0, "UnfoldBackwardBackward0"), (1, acc)]
r = torch.arange(10.0, dtype=d64).reshape(2, 5).requires_grad_()
(gg,) = torch.autograd.grad(g, gin, r, create_graph=True)
assert torch.equal(gg, r.unfold(1, 3, 1))
assert chain(gg) == [(0, "UnfoldBackward0"), (1, acc)]
gin = torch.ones(2, 2, 2, dtype=d64, requires_grad=True)
(g,) = torch.autograd.grad(x.unfold(1, 2, 2), x, gin, create_graph=True)
assert chain(g) == [(0, "UnfoldBackwardBackward0"), (1, acc)]

# writes_into_unfold_results
with torch.no_grad():
    v = torch.zeros(5)
    v.unfold(0, 3, 1).add_(1)
    assert v.tolist() == [1, 2, 3, 2, 1]
    v = torch.zeros(5)
    v.unfold(0, 2, 2).fill_(4)
    assert v.tolist() == [4, 4, 4, 4, 0]
w = torch.ones(5, requires_grad=True)
z = w * 1
z.unfold(0, 2, 2).mul_(2)
assert chain(z) == [(0, "CopySlices"), (1, "MulBackward0"), (2, acc)]
assert z.tolist() == [2, 2, 2, 2, 1]
z = w * 1
z.unfold(0, 3, 1).add_(1)
assert z.tolist() == [2, 3, 4, 3, 2]
raises(RuntimeError, "a view of a leaf Variable that requires grad is being used in an in-place "
       "operation.", lambda: w.unfold(0, 3, 1).add_(1))

print("unfold: all checks passed")
