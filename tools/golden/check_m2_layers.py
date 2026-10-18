"""Check the reference behavior that tests/nn_layers.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: the formulas and edge cases of nn.init, the
default initialization and edge cases of nn.Linear, the branches and nodes of F.linear and its
errors, and the activation layers. Values that come from random numbers (homura's differ) are
checked through their bounds only.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

import reference

reference.check()
warnings.simplefilter("ignore")


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


# init_gains_and_fans_follow_pytorch
assert init.calculate_gain("linear") == 1
assert init.calculate_gain("conv2d") == 1
assert init.calculate_gain("conv_transpose3d") == 1
assert init.calculate_gain("sigmoid") == 1
assert init.calculate_gain("tanh") == 5.0 / 3
assert init.calculate_gain("relu") == math.sqrt(2.0)
assert init.calculate_gain("relu", 3.0) == math.sqrt(2.0)
assert init.calculate_gain("leaky_relu") == math.sqrt(2.0 / (1 + 0.01**2))
assert init.calculate_gain("leaky_relu", 0.2) == math.sqrt(2.0 / 1.04)
assert init.calculate_gain("leaky_relu", math.sqrt(5)) == 0.5773502691896257
assert init.calculate_gain("selu") == 0.75
assert init._calculate_fan_in_and_fan_out(torch.empty(5, 3, 2, 2)) == (12, 20)
assert init._calculate_fan_in_and_fan_out(torch.empty(5, 3)) == (3, 5)
assert init._calculate_fan_in_and_fan_out(torch.empty(5, 0, 2)) == (0, 10)
fewer = "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
raises(ValueError, fewer, lambda: init._calculate_fan_in_and_fan_out(torch.empty(3)))

# init_samples_lie_within_the_bounds: the bounds and standard deviations of [64, 50].
fan_in, fan_out = 50, 64
assert math.isclose(math.sqrt(3) * math.sqrt(2) / math.sqrt(fan_in), math.sqrt(6 / fan_in))
w = torch.empty(64, 50, dtype=torch.float64)
init.kaiming_uniform_(w, a=0, mode="fan_out", nonlinearity="relu")
assert w.abs().max() <= math.sqrt(3) * math.sqrt(2) / math.sqrt(fan_out)
init.xavier_uniform_(w, gain=2.0)
assert w.abs().max() <= 2.0 * math.sqrt(6 / (fan_in + fan_out))
init.kaiming_uniform_(w, a=math.sqrt(5))
assert w.abs().max() <= 1 / math.sqrt(fan_in)
# The bound of nn.Linear's weights is computed through the standard deviation and rounds below
# 1 / sqrt(fan_in) for fan_in = 4.
assert math.sqrt(3.0) * (init.calculate_gain("leaky_relu", math.sqrt(5)) / math.sqrt(4)) == 0.49999999999999994

# init_edge_cases
for f in [init.kaiming_uniform_, init.kaiming_normal_]:
    assert f(torch.empty(0)).shape == (0,)
    assert f(torch.empty(3, 0)).shape == (3, 0)
    raises(ValueError, fewer, lambda: f(torch.empty(3)))
    raises(ValueError, fewer, lambda: f(torch.empty(())))
for f in [init.xavier_uniform_, init.xavier_normal_]:
    raises(ValueError, fewer, lambda: f(torch.empty(0)))
    assert f(torch.empty(3, 0)).shape == (3, 0)
    raises(ZeroDivisionError, "division by zero", lambda: f(torch.empty(0, 0)))
for f in [init.kaiming_uniform_, init.xavier_uniform_, init.uniform_]:
    raises(NotImplementedError, '"check_uniform_bounds" not implemented for \'Long\'',
           lambda: f(torch.empty(3, 4, dtype=torch.int64)))
for f in [init.kaiming_normal_, init.xavier_normal_, init.normal_]:
    raises(NotImplementedError, '"normal_kernel_cpu" not implemented for \'Long\'',
           lambda: f(torch.empty(3, 4, dtype=torch.int64)))
assert init.constant_(torch.empty(3, dtype=torch.int64), -2.7).tolist() == [-2, -2, -2]
assert init.constant_(torch.empty(3, dtype=torch.bool), 0.5).tolist() == [True] * 3
assert init.ones_(torch.empty(2, dtype=torch.int64)).tolist() == [1, 1]
assert init.zeros_(torch.empty(2, dtype=torch.bool)).tolist() == [False, False]
leaf = torch.zeros(3, 4, requires_grad=True)
init.kaiming_uniform_(leaf)
assert leaf.is_leaf and leaf.grad_fn is None and leaf.requires_grad
meta = torch.empty(3, 4, device="meta")
for f in [init.kaiming_uniform_, init.kaiming_normal_, init.xavier_uniform_, init.xavier_normal_,
          init.uniform_, init.normal_, init.ones_, init.zeros_]:
    assert f(meta).is_meta
raises(RuntimeError, "uniform_ expects to return a [from, to) range, but found from=2 > to=1",
       lambda: init.uniform_(torch.empty(3), 2.0, 1.0))
raises(RuntimeError, "normal expects std >= 0.0, but found std -1",
       lambda: init.normal_(torch.empty(3), 0.0, -1.0))

# linear_layers_initialize_like_pytorch
lin = nn.Linear(4, 3)
assert list(lin.state_dict()) == ["weight", "bias"]
assert lin.weight.shape == (3, 4) and lin.bias.shape == (3,)
assert lin.weight.abs().max() <= 0.5 and lin.bias.abs().max() <= 0.5
assert repr(lin) == "Linear(in_features=4, out_features=3, bias=True)"
lin = nn.Linear(4, 3, bias=False)
assert lin.bias is None and list(lin.state_dict()) == ["weight"]
lin = nn.Linear(0, 3)
assert lin.weight.shape == (3, 0) and lin.bias.tolist() == [0.0] * 3
assert lin(torch.empty(2, 0)).tolist() == [[0.0] * 3] * 2
lin = nn.Linear(3, 0)
assert lin.weight.shape == (0, 3) and lin.bias.shape == (0,)
lin = nn.Linear(3, 2, device="meta")
assert lin.weight.is_meta and lin.bias.is_meta and lin(torch.empty(5, 3, device="meta")).shape == (5, 2)
raises(RuntimeError, "Only Tensors of floating point and complex dtype can require gradients",
       lambda: nn.Linear(3, 2, dtype=torch.int64))

# linear_records_the_reference_nodes: homura has TransposeBackward0 where these have
# TBackward0, and a single node for matmul (see the test).
w = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
b = torch.randn(3, dtype=torch.float64, requires_grad=True)


def x_of(*shape):
    return torch.randn(*shape, dtype=torch.float64, requires_grad=True)


assert chain(F.linear(x_of(2, 4), w, b)) == [
    (0, "AddmmBackward0"), (1, "AccumulateGrad"), (1, "AccumulateGrad"),
    (1, "TBackward0"), (2, "AccumulateGrad")]
for shape in [(4,), (5, 2, 4), (2, 3, 2, 4)]:
    assert chain(F.linear(x_of(*shape), w, b)) == [
        (0, "ViewBackward0"), (1, "AddmmBackward0"), (2, "AccumulateGrad"),
        (2, "ViewBackward0"), (3, "AccumulateGrad"), (2, "TBackward0"), (3, "AccumulateGrad")]
assert chain(F.linear(x_of(2, 4), w)) == [
    (0, "MmBackward0"), (1, "AccumulateGrad"), (1, "TBackward0"), (2, "AccumulateGrad")]
xt = torch.randn(2, 4, 5, dtype=torch.float64).transpose(1, 2).requires_grad_()
top = chain(F.linear(xt, w, b))
assert top[0] == (0, "AddBackward0") and top[-1] == (1, "AccumulateGrad"), top
# A 2-D bias with a contiguous 4-D input also takes the matmul + add_ branch.
assert chain(F.linear(x_of(2, 3, 2, 4), w, torch.randn(2, 3, dtype=torch.float64)))[0] == (0, "AddBackward0")

# Review round 47 (R56): only a contiguous bias with a single dimension that isn't 1 takes the
# flattened addmm; a 2-D input always takes the addmm.
def strided_bias():
    return torch.randn(3, 2, dtype=torch.float64)[:, 0].requires_grad_()


for x_shape, bias, node in [
    ((2, 3, 2, 4), torch.randn(2, 3, dtype=torch.float64), "AddBackward0"),
    ((2, 5, 4), torch.randn(5, 3, dtype=torch.float64), "AddBackward0"),
    ((2, 5, 4), torch.randn(2, 5, 3, dtype=torch.float64), "AddBackward0"),
    ((2, 5, 4), torch.randn((), dtype=torch.float64), "AddBackward0"),
    ((2, 5, 4), strided_bias(), "AddBackward0"),
    ((4,), strided_bias(), "AddBackward0"),
    ((2, 3, 2, 4), torch.randn(1, 3, dtype=torch.float64), "ViewBackward0"),
    ((2, 4), strided_bias(), "AddmmBackward0"),
    ((2, 4), torch.randn(2, 3, dtype=torch.float64), "AddmmBackward0"),
    ((2, 1, 2, 4), torch.randn(2, 1, 1, 3, dtype=torch.float64), "AddBackward0"),
]:
    assert chain(F.linear(x_of(*x_shape), w, bias))[0] == (0, node), (x_shape, bias.shape)
try:
    F.linear(x_of(2, 5, 4), w, torch.randn(1, 1, 3, dtype=torch.float64))
    raise AssertionError("a [1, 1, 3] bias should fail in the addmm")
except RuntimeError:
    pass
try:
    F.linear(x_of(2, 1, 2, 4), w, torch.randn(2, 1, dtype=torch.float64))
    raise AssertionError("a [2, 1] bias should fail in the addmm")
except RuntimeError:
    pass

# linear_rejects_invalid_arguments
raises(RuntimeError, "both arguments to linear need to be at least 1D, but they are 0D and 2D",
       lambda: F.linear(torch.randn(()), torch.randn(3, 1)))
raises(RuntimeError, "t() expects a tensor with <= 2 dimensions, but self is 3D",
       lambda: F.linear(torch.randn(2, 4), torch.randn(1, 3, 4)))
raises(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (2x5 and 4x3)",
       lambda: F.linear(torch.randn(2, 5), torch.randn(3, 4)))
raises(RuntimeError, "expected m1 and m2 to have the same dtype, but got: float != double",
       lambda: F.linear(torch.randn(2, 4), torch.randn(3, 4, dtype=torch.float64)))
raises(RuntimeError, "self and mat2 must have the same dtype, but got Double and Float",
       lambda: F.linear(torch.randn(2, 4), torch.randn(3, 4), torch.randn(3, dtype=torch.float64)))

# activation_layers_apply_their_functions
x = torch.randn(3, 4)
assert nn.ReLU()(x) is not x
assert torch.equal(nn.ReLU()(x), x.relu())
assert torch.equal(nn.GELU(approximate="tanh")(x), F.gelu(x, approximate="tanh"))
assert torch.equal(nn.Softmax(-1)(x), x.softmax(-1))
assert repr(nn.Softmax(1)) == "Softmax(dim=1)"
assert list(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).state_dict()) == [
    "0.weight", "0.bias", "2.weight", "2.bias"]

# scalar_fills_check_overflow_like_pytorch (R57): fill_, full, nn.init.constant_ and masked_fill
# check the value against the dtype, also without elements, but not on the meta device.
inf, nan = float("inf"), float("nan")
FILL = [
    (torch.uint8, 300, None), (torch.uint8, 300.0, None), (torch.uint8, -1, 255),
    (torch.uint8, -255, 1), (torch.uint8, -256, None), (torch.uint8, -0.5, None),
    (torch.uint8, -1.0, None), (torch.uint8, 255.9, 255), (torch.uint8, 256.0, None),
    (torch.uint8, True, 1), (torch.int8, 128.0, None), (torch.int8, 127.9, 127),
    (torch.int8, -128.9, None), (torch.int8, -0.9, 0), (torch.int8, -128, -128),
    (torch.int8, -129, None), (torch.int32, 2147483647.5, 2147483647),
    (torch.int32, -2147483648.5, None), (torch.int64, 2.0**63, None), (torch.int64, 9.3e18, None),
    (torch.int64, inf, None), (torch.int64, nan, None), (torch.float16, 65504.0, 65504.0),
    (torch.float16, 65519.0, None), (torch.float16, 65505, None), (torch.float16, 1e6, None),
    (torch.float16, -inf, -inf), (torch.float16, nan, nan), (torch.bfloat16, 65504.0, 65536.0),
    (torch.bfloat16, 3.4e38, None), (torch.float32, 3.4e38, float(torch.tensor(3.4e38))),
    (torch.float32, 3.5e38, None), (torch.float32, 1e40, None), (torch.float32, 1e-50, 0.0),
    (torch.float64, 1e300, 1e300), (torch.bool, nan, True), (torch.bool, -256, True),
    (torch.bool, 0.0, False),
]
FILLS = [
    lambda s, d, v: torch.zeros(s, dtype=d).fill_(v),
    lambda s, d, v: torch.full(s, v, dtype=d),
    lambda s, d, v: init.constant_(torch.zeros(s, dtype=d), v),
    lambda s, d, v: torch.zeros(s, dtype=d).masked_fill(torch.ones(s, dtype=torch.bool), v),
]
for dtype, value, expected in FILL:
    for f in FILLS:
        for shape in [(2,), (0,)]:
            if expected is None:
                raises(RuntimeError, None, lambda: f(shape, dtype, value))
            elif shape == (2,):
                t = f(shape, dtype, value)
                assert t.dtype == dtype
                assert all(x == expected or (x != x and expected != expected) for x in t.tolist()), (dtype, value, t)
            else:
                assert f(shape, dtype, value).shape == shape
    meta = torch.zeros(2, dtype=dtype, device="meta")
    meta.fill_(value)
    torch.full((2,), value, dtype=dtype, device="meta")
    meta_mask = torch.ones(2, dtype=torch.bool, device="meta")
    # The meta masked_fill is a Python decomposition that converts the value with int() for
    # integer dtypes: it doesn't check the range, but fails for infinities and NaN (homura's meta
    # tensors take those too).
    if not dtype.is_floating_point and dtype != torch.bool and not math.isfinite(value):
        raises((OverflowError, ValueError), None, lambda: meta.masked_fill(meta_mask, value))
    else:
        meta.masked_fill(meta_mask, value)
raises(RuntimeError, "value cannot be converted to type float without overflow",
       lambda: torch.full((2,), 1e40))
leaf = torch.tensor([1.0, 2.0], requires_grad=True)
leaf.grad = torch.tensor([5.0, 6.0])
grad = leaf.grad
saved = leaf * leaf
raises(RuntimeError, "value cannot be converted to type float without overflow",
       lambda: init.constant_(leaf, 1e40))
assert torch.is_grad_enabled()
raises(RuntimeError, "a leaf Variable that requires grad is being used in an in-place operation.",
       lambda: leaf.fill_(1e40))
assert leaf.tolist() == [1.0, 2.0] and leaf.grad is grad and leaf.is_leaf
y = leaf * 1.0
node = y.grad_fn
version = y._version
raises(RuntimeError, "value cannot be converted to type float without overflow", lambda: y.fill_(1e40))
assert y.grad_fn is node and y._version == version
saved.sum().backward()
assert leaf.grad.tolist() == [7.0, 10.0]

# masked_fill_checks_tensor_values_like_scalars (R58): a zero-dimensional tensor value is read as
# a scalar and checked like one.
for dtype, v in [
    (torch.uint8, torch.tensor(300)),
    (torch.uint8, torch.tensor(-1.0)),
    (torch.float16, torch.tensor(1e6, dtype=torch.float64)),
    (torch.float32, torch.tensor(1e40, dtype=torch.float64)),
]:
    for shape in [(2,), (0,)]:
        for fill in [True, False]:
            mask = torch.full(shape, fill)
            x = torch.zeros(shape, dtype=dtype)
            raises(RuntimeError, None, lambda: x.masked_fill(mask, v))
            raises(RuntimeError, None, lambda: x.masked_fill_(mask, v))
            assert (x == 0).all()
mask = torch.tensor([True, False])
assert torch.zeros(2, dtype=torch.uint8).masked_fill(mask, torch.tensor(-1)).tolist() == [255, 0]
assert torch.zeros(2, dtype=torch.uint8).masked_fill(mask, torch.tensor(255.9, dtype=torch.float64)).tolist() == [255, 0]
assert torch.zeros(2, dtype=torch.int64).masked_fill(mask, torch.tensor(2**63 - 1)).tolist() == [2**63 - 1, 0]
assert torch.zeros(2, dtype=torch.float16).masked_fill(mask, torch.tensor(inf, dtype=torch.float64)).tolist() == [inf, 0.0]
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
v = torch.tensor(2.7, dtype=torch.float64, requires_grad=True)
y = x.masked_fill(torch.tensor([True, False, True]), v)
assert y.tolist() == [float(torch.tensor(2.7)), 2.0, float(torch.tensor(2.7))]
(y * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
assert v.grad.tolist() == 4.0 and x.grad.tolist() == [0.0, 2.0, 0.0]
leaf = torch.tensor([1.0, 2.0], requires_grad=True)
y = leaf * 1.0
saved = y * y
node, version = y.grad_fn, y._version
big = torch.tensor(1e40, dtype=torch.float64, requires_grad=True)
raises(RuntimeError, "value cannot be converted to type float without overflow",
       lambda: y.masked_fill_(torch.tensor([True, True]), big))
assert y.grad_fn is node and y._version == version and y.tolist() == [1.0, 2.0]
saved.sum().backward()
assert leaf.grad.tolist() == [2.0, 4.0] and big.grad is None
meta = torch.zeros(2, dtype=torch.uint8, device="meta")
meta.masked_fill(torch.ones(2, dtype=torch.bool, device="meta"), torch.tensor(300, device="meta"))

print("check_m2_layers: all reference assertions passed")
