"""Check the reference behavior that tests/nn_conv.rs and tests/nn_pool.rs encode; does not write
fixtures.

Each block mirrors a test there, in the same order (the golden tests excepted): F.conv2d's node,
argument errors in PyTorch's order, dtypes, devices and degenerate shapes, its half precision
cancellation cases (and that this CPU can't compute half precision convolution gradients, so
gen_m2_conv_pool.py derives them); then F.max_pool2d: its selection rules, argument errors,
windows without input and the check for them on every small geometry, nodes and saved tensors,
shapes and dtypes; then F.adaptive_avg_pool2d: its windows and mean path, argument errors, nodes
and saved tensors. Where homura differs on purpose, the reference behavior is asserted and the
difference named in a comment.
"""

import itertools
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


def counting(*shape, dtype=d64):
    n = 1
    for s in shape:
        n *= s
    return (torch.arange(n, dtype=dtype) + 1).reshape(shape).clone()


def leaf(*shape, dtype=d64):
    return counting(*shape, dtype=dtype).requires_grad_()


acc = "AccumulateGrad"

# ---------------------------------------------------------------- conv2d

# conv2d_records_its_operations
# PyTorch records one ConvolutionBackward0 (ConvolutionBackwardBackward0 for its gradients);
# homura records the operations of its composite, a documented difference.
x, w, b = leaf(2, 3, 5, 5), leaf(4, 3, 3, 3), leaf(4)
assert chain(F.conv2d(x, w, b)) == [(0, "ConvolutionBackward0"), (1, acc), (1, acc), (1, acc)]
assert chain(F.conv2d(x[0], w)) == [
    (0, "SqueezeBackward1"), (1, "ConvolutionBackward0"), (2, "UnsqueezeBackward0"),
    (3, "SelectBackward0"), (4, acc), (2, acc)]
assert chain(F.conv2d(x, leaf(4, 3, 2, 2), padding="same")) == [
    (0, "ConvolutionBackward0"), (1, "ConstantPadNdBackward0"), (2, acc), (1, acc)]
g = torch.ones(2, 4, 3, 3, dtype=d64, requires_grad=True)
grads = torch.autograd.grad(F.conv2d(x, w, b), [x, w, b], g, create_graph=True)
assert all(type(t.grad_fn).__name__ == "ConvolutionBackwardBackward0" for t in grads)
with torch.no_grad():
    assert F.conv2d(x, w, b).grad_fn is None

# conv2d_checks_its_arguments
x, w, b = torch.randn(2, 3, 5, 5, dtype=d64), torch.randn(4, 3, 3, 3, dtype=d64), torch.randn(4, dtype=d64)
raises(RuntimeError, "Input type (double) and bias type (float) should be the same",
       lambda: F.conv2d(x, w, b.float()))
raises(RuntimeError, "Input type (double) and bias type (float) should be the same",
       lambda: F.conv2d(x[0, 0], w, b.float()))
# PyTorch's "valid" and "same" paths don't compare the bias dtype; homura does for all paddings.
assert F.conv2d(x, w, b.float(), padding="valid").dtype == d64
raises(RuntimeError, "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of "
       "size: [5, 5]", lambda: F.conv2d(x[0, 0], w, groups=0))
raises(RuntimeError, "weight should have at least three dimensions", lambda: F.conv2d(x, w[0, 0], groups=0))
raises(RuntimeError, "non-positive groups is not supported", lambda: F.conv2d(x, w[0], groups=0))
raises(RuntimeError, "non-positive groups is not supported", lambda: F.conv2d(x, w, stride=0, groups=0))
raises(RuntimeError, "non-positive groups is not supported",
       lambda: F.conv2d(x, w, stride=0, groups=0, padding="same"))
raises(RuntimeError, "expected stride to be a single integer value or a list of 1 values to match "
       "the convolution dimensions, but got stride=[0, 0]", lambda: F.conv2d(x, w[0], stride=0))
raises(RuntimeError, "expected stride to be a single integer value or a list of 3 values to match "
       "the convolution dimensions, but got stride=[1, 1]", lambda: F.conv2d(x, w[None]))
raises(RuntimeError, "Expected 3-dimensional input for 3-dimensional weight[3, 3, 3], but got "
       "4-dimensional input of size [2, 3, 5, 5] instead", lambda: F.conv2d(x, w[0], padding="same"))
raises(RuntimeError, "padding='same' is not supported for strided convolutions",
       lambda: F.conv2d(x, w, padding="same", stride=(1, 2)))
raises(RuntimeError, "padding='same' is not supported for strided convolutions",
       lambda: F.conv2d(x, w, padding="same", stride=0))
raises(RuntimeError, "non-positive stride is not supported", lambda: F.conv2d(x, w, stride=0, groups=2))
raises(RuntimeError, "non-positive stride is not supported", lambda: F.conv2d(x, w, stride=0, dilation=0))
raises(RuntimeError, "Given groups=1, expected weight to be at least 1 at dimension 0, but got "
       "weight of size [0, 3, 3, 3] instead", lambda: F.conv2d(x, w[:0], dilation=0))
raises(RuntimeError, "Given groups=4, expected weight to be at least 4 at dimension 0, but got "
       "weight of size [1, 1, 3, 3] instead",
       lambda: F.conv2d(torch.ones(1, 4, 5, 5), torch.ones(1, 1, 3, 3), groups=4))
raises(RuntimeError, "Given groups=2, expected weight to be divisible by 2 at dimension 0, but got "
       "weight of size [[3, 2, 3, 3]] instead",
       lambda: F.conv2d(torch.ones(1, 4, 5, 5), torch.ones(3, 2, 3, 3), groups=2, padding="same"))
raises(RuntimeError, "Given groups=1, weight of size [4, 3, 3, 3], expected input[2, 2, 5, 5] to "
       "have 3 channels, but got 2 channels instead", lambda: F.conv2d(x[:, :2], w, b[:3]))
raises(RuntimeError, "Given groups=1, weight of size [4, 3, 3, 3], expected input[1, 2, 5, 5] to "
       "have 3 channels, but got 2 channels instead", lambda: F.conv2d(x[0, :2], w.float(), dilation=0))
raises(RuntimeError, "Given weight of size [4, 3, 3, 3], expected bias to be 1-dimensional with 4 "
       "elements, but got bias of size [1, 4] instead", lambda: F.conv2d(x, w, b[None]))
raises(RuntimeError, "Given weight of size [4, 3, 3, 3], expected bias to be 1-dimensional with 4 "
       "elements, but got bias of size [2] instead", lambda: F.conv2d(x, w, b[:2], dilation=0))
raises(RuntimeError, "Given padding=4611686018427387904 at dimension 0 , expected padding to be at "
       "most 4611686018427387903", lambda: F.conv2d(x, w, padding=2**62))
raises(RuntimeError, "Calculated padded input size per channel: (2 x 2). Kernel size: (3 x 3). "
       "Kernel size can't be greater than actual input size", lambda: F.conv2d(x[..., :2, :2], w.float()))
raises(RuntimeError, "Calculated padded input size per channel: (5 x 5). Kernel size: (7 x 7). "
       "Kernel size can't be greater than actual input size", lambda: F.conv2d(x, w, dilation=3))
raises(RuntimeError, "Calculated padded input size per channel: (2 x 7). Kernel size: (3 x 3). "
       "Kernel size can't be greater than actual input size",
       lambda: F.conv2d(torch.ones(1, 3, 0, 5, dtype=d64), w, padding=1))
# "same" pads first, so a kernel larger than the input is accepted.
assert F.conv2d(x[..., :2, :2], w, padding="same").shape == (2, 4, 2, 2)
raises(RuntimeError, "dilation should be greater than zero, but got [0, 0]",
       lambda: F.conv2d(x, w.float(), dilation=0))
raises(RuntimeError, "dilation should be greater than zero, but got [0, 1]",
       lambda: F.conv2d(x, w, padding="same", dilation=(0, 1)))
raises(RuntimeError, "kernel size should be greater than zero, but got kernel_height: 0 "
       "kernel_width: 3", lambda: F.conv2d(x, torch.ones(4, 3, 0, 3, dtype=d64)))
raises(RuntimeError, "expected scalar type Double but found Float", lambda: F.conv2d(x, w.float()))
raises(RuntimeError, "expected scalar type Long but found Double", lambda: F.conv2d(x.long(), w))
# An F32 or half precision input gets the oneDNN backend's message; homura reports
# "expected scalar type .. but found .." for every dtype.
raises(RuntimeError, "Input type (torch.FloatTensor) and weight type (torch.DoubleTensor) should be "
       "the same or input should be a MKLDNN tensor and weight is a dense tensor",
       lambda: F.conv2d(x.float(), w))
raises(NotImplementedError, "\"slow_conv2d_cpu\" not implemented for 'Bool'",
       lambda: F.conv2d(torch.ones(1, 1, 3, 3, dtype=torch.bool), torch.ones(1, 1, 2, 2, dtype=torch.bool), padding=1))
# PyTorch's float32 path names dilation 0 differently, and its meta kernel accepts it (homura
# rejects dilation 0 everywhere with the message above).
raises(RuntimeError, "non-positive dilation is not supported", lambda: F.conv2d(x.float(), w.float(), dilation=0))
assert F.conv2d(x.to("meta"), w.to("meta"), dilation=0).shape == (2, 4, 5, 5)

# conv2d_dtypes_and_devices
for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
    y = F.conv2d(torch.ones(1, 1, 3, 3, dtype=dt), torch.ones(1, 1, 2, 2, dtype=dt))
    assert y.dtype == dt and y.tolist() == [[[[4, 4], [4, 4]]]], dt
# Integer products and sums wrap.
assert F.conv2d(torch.full((1, 1, 1, 2), 200, dtype=torch.uint8),
                torch.ones(1, 1, 1, 2, dtype=torch.uint8)).tolist() == [[[[144]]]]
assert F.conv2d(torch.full((1, 1, 1, 2), 2**62, dtype=torch.int64),
                torch.ones(1, 1, 1, 2, dtype=torch.int64)).tolist() == [[[[-2**63]]]]
y = F.conv2d(x.to("meta"), w.to("meta"), b.to("meta"), stride=2, dilation=2, padding=2)
assert y.shape == (2, 4, 3, 3) and y.device.type == "meta"
# PyTorch accepts a meta input with CPU parameters, and a meta weight of another dtype; homura
# requires one device and one dtype.
assert F.conv2d(x.to("meta"), w).device.type == "meta"
assert F.conv2d(x.to("meta"), w.float().to("meta")).dtype == d64
raises(RuntimeError, "Input type (double) and bias type (float) should be the same",
       lambda: F.conv2d(x.to("meta"), w.to("meta"), b.float().to("meta")))

# conv2d_degenerate_shapes
xe = torch.zeros(0, 3, 5, 5, dtype=d64, requires_grad=True)
we = torch.randn(4, 3, 3, 3, dtype=d64, requires_grad=True)
y = F.conv2d(xe, we)
assert y.shape == (0, 4, 3, 3)
(gw,) = torch.autograd.grad(y.sum(), we)
assert gw.abs().sum().item() == 0.0
# Without input channels PyTorch returns an output without channels (homura panics).
assert F.conv2d(torch.ones(1, 0, 5, 5, dtype=d64), torch.ones(4, 0, 3, 3, dtype=d64)).shape == (1, 0, 3, 3)
assert F.conv2d(torch.ones(0, 5, 5, dtype=d64), torch.ones(4, 0, 3, 3, dtype=d64)).shape == (0, 3, 3)
assert F.conv2d(x, w[..., :1, :1], stride=2).shape == (2, 4, 3, 3)
assert F.conv2d(x[..., :3, :3], w).shape == (2, 4, 1, 1)
assert F.conv2d(x, w, padding="same").shape == (2, 4, 5, 5)
assert F.conv2d(x, w, padding="valid").shape == (2, 4, 3, 3)
assert F.conv2d(x, w, stride=(2, 1), padding=(1, 0), dilation=(1, 2)).shape == (2, 4, 3, 1)
# "same" pads asymmetrically for an even effective kernel: the extra row and column go after.
ones = torch.ones(1, 1, 4, 4, dtype=d64)
k2 = torch.ones(1, 1, 2, 2, dtype=d64)
assert F.conv2d(ones, k2, padding="same")[0, 0].tolist() == [
    [4, 4, 4, 2], [4, 4, 4, 2], [4, 4, 4, 2], [2, 2, 2, 1]]
# Kernel height 2 with dilation 3: one row before, two after.
assert F.conv2d(ones, torch.ones(1, 1, 2, 3, dtype=d64), padding="same", dilation=(3, 1))[0, 0].tolist() == [
    [2, 3, 3, 2], [4, 6, 6, 4], [2, 3, 3, 2], [2, 3, 3, 2]]

# conv2d_half_precision_rounds_once
# Half precision is computed in F32 and rounded once, so a bias can cancel what the products
# would overflow or round away in the dtype.
y = F.conv2d(torch.full((1, 1, 1, 1), 60000.0, dtype=torch.float16),
             torch.full((1, 1, 1, 1), 2.0, dtype=torch.float16),
             torch.full((1,), -60000.0, dtype=torch.float16))
assert y.item() == 60000.0
for dt, expected in [(torch.float16, 0.0005002021789550781), (torch.bfloat16, 0.0004997253417968750)]:
    y = F.conv2d(torch.ones(1, 2, 1, 1, dtype=dt), torch.tensor([1.0, 0.0005], dtype=dt).reshape(1, 2, 1, 1),
                 torch.tensor([-1.0], dtype=dt))
    assert y.item() == expected, (dt, y.item())
    assert (F.conv2d(torch.ones(1, 2, 1, 1), torch.tensor([1.0, 0.0005], dtype=dt).float().reshape(1, 2, 1, 1),
                     torch.tensor([-1.0])).to(dt)).item() == expected
xh = torch.ones(1, 1, 3, 3, dtype=torch.float16, requires_grad=True)
for dt in [torch.float16, torch.bfloat16]:
    xh = torch.ones(1, 1, 3, 3, dtype=dt, requires_grad=True)
    raises(RuntimeError, "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2",
           lambda: torch.autograd.grad(F.conv2d(xh, torch.ones(1, 1, 2, 2, dtype=dt)).sum(), xh))

print("conv2d: all checks passed")

# ---------------------------------------------------------------- max_pool2d

# max_pool2d_selection_rules
# Several NaN: the last; ordinary ties: the first; padding never.
for dt in [torch.float16, torch.bfloat16, torch.float32, d64]:
    x = torch.tensor([[[[NAN, 2.0], [NAN, 1.0]]]], dtype=dt, requires_grad=True)
    y, i = F.max_pool2d(x, 2, return_indices=True)
    assert i.item() == 2 and y.isnan().item()
    (g,) = torch.autograd.grad(y.sum(), x)
    assert g.flatten().tolist() == [0, 0, 1, 0]
x = torch.tensor([[[[3.0, 3.0], [1.0, 3.0]]]], dtype=d64)
assert F.max_pool2d(x, 2, return_indices=True)[1].item() == 0
x = torch.full((1, 1, 1, 1), -INF, dtype=d64, requires_grad=True)
y, i = F.max_pool2d(x, 2, stride=1, padding=1, return_indices=True)
assert i.flatten().tolist() == [0, 0, 0, 0] and y.flatten().tolist() == [-INF] * 4
(g,) = torch.autograd.grad(y.sum(), x)
assert g.item() == 4.0
y, i = F.max_pool2d(torch.full((1, 2, 2), -128, dtype=torch.int8), 2, stride=1, padding=1, return_indices=True)
assert i.tolist() == [[[0, 0, 1], [0, 0, 1], [2, 2, 3]]] and y.unique().tolist() == [-128]
assert F.max_pool2d(torch.zeros(1, 2, 2, dtype=torch.uint8), 2, return_indices=True)[1].tolist() == [[[0]]]
# Upstream infinities and NaN reach only the selected positions.
x = torch.tensor([[[[1.0, 5.0, 2.0, 0.0]]]], dtype=d64, requires_grad=True)
y = F.max_pool2d(x, (1, 2))
(g,) = torch.autograd.grad(y, x, torch.tensor([[[[INF, NAN]]]], dtype=d64))
assert [repr(v) for v in g.flatten().tolist()] == ["0.0", "inf", "nan", "0.0"]
# ceil_mode: the last window must start inside the input or the left padding.
x = counting(1, 1, 5, 5)
y, i = F.max_pool2d(x, 2, ceil_mode=True, return_indices=True)
assert i[0, 0].tolist() == [[6, 8, 9], [16, 18, 19], [21, 23, 24]]
assert F.max_pool2d(counting(1, 1, 4, 4), 2, stride=2, padding=1, ceil_mode=True).shape == (1, 1, 3, 3)
assert F.max_pool2d(counting(1, 1, 5, 5), 3, stride=2, padding=1, ceil_mode=True).shape == (1, 1, 3, 3)
assert F.max_pool2d(counting(1, 1, 6, 6), 3, stride=2, ceil_mode=True).shape == (1, 1, 3, 3)
assert F.max_pool2d(counting(1, 1, 6, 6), 3, stride=2).shape == (1, 1, 2, 2)
# A third window would start in the right padding: two windows, as without ceil_mode.
y, i = F.max_pool2d(counting(1, 1, 5, 5), 2, stride=3, padding=1, ceil_mode=True, return_indices=True)
assert y.shape == (1, 1, 2, 2) and i[0, 0].tolist() == [[0, 3], [15, 18]]
# Dilation, a rectangular kernel, the default stride and plane indices.
y, i = F.max_pool2d(counting(1, 1, 5, 5), 2, stride=1, dilation=2, return_indices=True)
assert i[0, 0].tolist() == [[12, 13, 14], [17, 18, 19], [22, 23, 24]]
y, i = F.max_pool2d(counting(2, 2, 4, 6), (2, 3), return_indices=True)
assert y.shape == (2, 2, 2, 2) and i[1, 1].tolist() == [[8, 11], [20, 23]]
assert torch.equal(F.max_pool2d(counting(1, 3, 3), 2, stride=[]), F.max_pool2d(counting(1, 3, 3), 2, stride=2))
# Overlapping windows add their gradients where they select the same position, also with
# dilation when the stride equals the kernel (review round 65, R81).
x = counting(1, 1, 3, 3).requires_grad_()
(g,) = torch.autograd.grad(F.max_pool2d(x, 2, stride=1).sum(), x)
assert g[0, 0].tolist() == [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
x = torch.tensor([[[[0.0, 0.0, 10.0, 0.0, 0.0]]]], dtype=d64, requires_grad=True)
y, i = F.max_pool2d(x, (1, 2), stride=(1, 2), dilation=(1, 2), return_indices=True)
assert i.flatten().tolist() == [2, 2]
(g,) = torch.autograd.grad(y.sum(), x)
assert g.flatten().tolist() == [0, 0, 2, 0, 0]
# Half precision adds the selected gradients in the dtype, in output order.
x = torch.tensor([[[[0.0, 0.0, 10.0, 0.0, 0.0]]]], dtype=torch.float16, requires_grad=True)
for upstream, expected in [([2048.0, 1.0, 1.0], 2048.0), ([1.0, 1.0, 2048.0], 2050.0)]:
    y = F.max_pool2d(x, (1, 3), stride=(1, 1))
    (g,) = torch.autograd.grad(y, x, torch.tensor(upstream, dtype=torch.float16).reshape(1, 1, 1, 3))
    assert g.flatten().tolist() == [0, 0, expected, 0, 0], g

# max_pool2d_checks_its_arguments
m = torch.randn(1, 1, 5, 5, dtype=d64)
for f in [lambda: F.max_pool2d(m[0, 0], 2**31), lambda: F.max_pool2d(m, 2, stride=2**31),
          lambda: F.max_pool2d(m, 2, padding=2**31), lambda: F.max_pool2d(m, 2, dilation=2**31)]:
    raises(RuntimeError, "value cannot be converted to type int without overflow", f)
raises(RuntimeError, "non-empty 3D or 4D (batch mode) tensor expected for input", lambda: F.max_pool2d(m[0, 0], 0))
raises(RuntimeError, "non-empty 3D or 4D (batch mode) tensor expected for input", lambda: F.max_pool2d(m[None], 2))
raises(RuntimeError, "stride should not be zero", lambda: F.max_pool2d(m, 0))
raises(RuntimeError, "stride should not be zero", lambda: F.max_pool2d(m, 0, stride=0))
raises(RuntimeError, "stride should not be zero", lambda: F.max_pool2d(m, 2, stride=(1, 0)))
raises(RuntimeError, "pad should be at most half of effective kernel size, but got pad=2, "
       "kernel_size=2 and dilation=1", lambda: F.max_pool2d(m, 2, padding=2))
raises(RuntimeError, "pad should be at most half of effective kernel size, but got pad=2, "
       "kernel_size=2 and dilation=1", lambda: F.max_pool2d(m, (2, 3), padding=2, dilation=(1, 2)))
raises(RuntimeError, "pad should be at most half of effective kernel size, but got pad=1, "
       "kernel_size=2 and dilation=0", lambda: F.max_pool2d(m, 2, dilation=0, padding=1))
raises(RuntimeError, "pad should be at most half of effective kernel size, but got pad=2, "
       "kernel_size=2 and dilation=1", lambda: F.max_pool2d(torch.ones(1, 0, 4, 4), 2, padding=2))
raises(RuntimeError, "kernel size should be greater than zero, but got kH: 0 kW: 0", lambda: F.max_pool2d(m, 0, stride=1))
raises(RuntimeError, "dilation should be greater than zero, but got dilationH: 0 dilationW: 0",
       lambda: F.max_pool2d(m, 2, dilation=0))
for shape in [(1, 0, 4, 4), (0, 4, 4), (1, 1, 0, 4), (1, 0, 1, 1)]:
    raises(RuntimeError, "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for "
           f"input, but got:[{', '.join(map(str, shape))}]", lambda: F.max_pool2d(torch.ones(shape), 2))
# The pad must be at most half of the effective kernel (checked first, height then width) and of
# the kernel itself (review round 64, R78).
nine = torch.ones(1, 1, 9, 9)
raises(RuntimeError, "pad should be smaller than or equal to half of kernel size, but got padW = 2, "
       "padH = 2, kW = 3, kH = 3", lambda: F.max_pool2d(nine, 3, dilation=2, padding=2))
raises(RuntimeError, "pad should be smaller than or equal to half of kernel size, but got padW = 2, "
       "padH = 0, kW = 3, kH = 3", lambda: F.max_pool2d(nine, 3, dilation=2, padding=(0, 2)))
assert F.max_pool2d(nine, 3, dilation=2, padding=1).shape == (1, 1, 3, 3)
raises(RuntimeError, "Given input size: (1x1x1). Calculated output size: (1x0x0). Output size is too "
       "small", lambda: F.max_pool2d(torch.ones(1, 1, 1), 2))
raises(RuntimeError, "Given input size: (1x2x2). Calculated output size: (1x0x0). Output size is too "
       "small", lambda: F.max_pool2d(torch.ones(1, 1, 2, 2), 3, stride=1, padding=1, dilation=2))
raises(NotImplementedError, "\"max_pool2d\" not implemented for 'Bool'",
       lambda: F.max_pool2d(torch.ones(0, 1, 4, 4, dtype=torch.bool), 2))
# PyTorch's meta kernel accepts bool inputs (homura rejects them on every device).
assert F.max_pool2d(torch.ones(1, 4, 4, dtype=torch.bool, device="meta"), 2).shape == (1, 2, 2)

# max_pool2d_rejects_windows_without_input
# Only possible with dilation > 1. PyTorch outputs -inf with an index outside the plane and gives
# no gradient (homura panics on CPU and meta, M2.8 #10).
x = torch.ones(1, 1, 1, 1, requires_grad=True)
y, i = F.max_pool2d(x, 2, stride=1, padding=1, dilation=2, return_indices=True)
assert y.flatten().tolist() == [-INF] and i.flatten().tolist() == [2]
(g,) = torch.autograd.grad(y.sum(), x)
assert g.item() == 0.0
assert F.max_pool2d(torch.ones(1, 1, 1, 1, device="meta"), 2, stride=1, padding=1, dilation=2).shape == (1, 1, 1, 1)
# Controls: every window has an input, and the first column has none, only in width.
y, i = F.max_pool2d(counting(1, 1, 3, 3), 2, stride=2, padding=1, dilation=3, ceil_mode=True, return_indices=True)
assert i[0, 0].tolist() == [[8, 7], [5, 4]]
y, i = F.max_pool2d(counting(1, 1, 3, 1), (2, 2), stride=1, padding=(0, 1), dilation=(1, 2), return_indices=True)
assert y.shape == (1, 1, 2, 1) and y.flatten().tolist() == [-INF, -INF]


# max_pool2d_window_check_matches_the_windows
# The windows where PyTorch returns -inf are exactly those without an input position, found by
# listing every window's positions, which homura's check must match (review round 66, R83).
def pooling_oracle(n, k, s, p, d, ceil_mode):
    if p > ((k - 1) * d + 1) // 2 or p > k // 2:
        return None
    size = (n + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
    if ceil_mode and (size - 1) * s >= n + p:
        size -= 1
    if size < 1:
        return None
    empty = next((o for o in range(size) if not any(0 <= o * s - p + j * d < n for j in range(k))), None)
    return size, empty


valid = rejected = 0
for n, k, s, p, d, ceil_mode in itertools.product(range(1, 7), range(1, 5), range(1, 4), range(3), range(1, 5),
                                                  [False, True]):
    expected = pooling_oracle(n, k, s, p, d, ceil_mode)
    for dim in [0, 1]:
        at = lambda v, other: (v, other) if dim == 0 else (other, v)
        args = dict(kernel_size=at(k, 1), stride=at(s, 1), padding=at(p, 0), dilation=at(d, 1), ceil_mode=ceil_mode)
        try:
            y = F.max_pool2d(torch.ones(1, 1, *at(n, 2)), **args)
        except RuntimeError:
            assert expected is None, (n, k, s, p, d, ceil_mode)
            continue
        size, empty = expected
        assert y.shape == (1, 1, *at(size, 2))
        line = (y[0, 0, :, 0] if dim == 0 else y[0, 0, 0, :]).tolist()
        assert [o for o, v in enumerate(line) if v == -INF][:1] == ([] if empty is None else [empty])
    if expected is not None:
        valid, rejected = valid + (expected[1] is None), rejected + (expected[1] is not None)
assert (valid, rejected) == (616, 24)
# The meta kernel takes the same time for any output size.
huge = 1 << 50
assert F.max_pool2d(torch.ones(1, 1, huge, 1, device="meta"), 1).shape == (1, 1, huge, 1)
y, i = F.max_pool2d(torch.ones(1, 1, 3, huge, device="meta"), 3, stride=1, padding=1, dilation=(2, 1), ceil_mode=True,
                    return_indices=True)
assert y.shape == i.shape == (1, 1, 1, huge)

# max_pool2d_nodes_and_saved_tensors
x = leaf(1, 1, 4, 4)
for f in [lambda t: F.max_pool2d(t, 2), lambda t: F.max_pool2d(t, 2, return_indices=True)[0]]:
    assert chain(f(x)) == [(0, "MaxPool2DWithIndicesBackward0"), (1, acc)]
y, i = F.max_pool2d(x, 2, return_indices=True)
assert i.dtype == torch.int64 and not i.requires_grad and i.grad_fn is None
assert sorted(n for n in dir(y.grad_fn) if n.startswith("_saved_")) == [
    "_saved_ceil_mode", "_saved_dilation", "_saved_kernel_size", "_saved_padding", "_saved_result1",
    "_saved_self", "_saved_stride"]
z = x * 1
y = F.max_pool2d(z, 2)
gin = torch.ones(1, 1, 2, 2, dtype=d64, requires_grad=True)
(gx,) = torch.autograd.grad(y, z, gin, create_graph=True)
assert chain(gx) == [(0, "MaxPool2DWithIndicesBackwardBackward0"), (1, acc), (1, "MulBackward0"), (2, acc)]
d_z, d_g = torch.autograd.grad((gx * counting(1, 1, 4, 4)).sum(), [z, gin], retain_graph=True)
assert d_z.abs().sum().item() == 0.0 and d_g[0, 0].tolist() == [[6, 8], [14, 16]]
r = counting(1, 1, 4, 4).requires_grad_()
(gg,) = torch.autograd.grad(gx, gin, r, create_graph=True)
assert gg[0, 0].tolist() == [[6, 8], [14, 16]]
# Modifying the input or the indices after the forward fails the backward pass.
z = x * 1
y, i = F.max_pool2d(z, 2, return_indices=True)
z.add_(1)
raises(RuntimeError, None, lambda: torch.autograd.grad(y.sum(), x))
z = x * 1
y, i = F.max_pool2d(z, 2, return_indices=True)
i.zero_()
raises(RuntimeError, None, lambda: torch.autograd.grad(y.sum(), x))
# The second derivatives need the indices whichever are requested, so modifying them after the
# first backward pass fails the second one; these nodes don't save the input (review round 66, R82).
for modified in ["none", "indices", "input"]:
    for wrt in [[0], [1], [0, 1]]:
        z = x * 1
        y, i = F.max_pool2d(z, 2, return_indices=True)
        gin = torch.ones(1, 1, 2, 2, dtype=d64, requires_grad=True)
        (gx,) = torch.autograd.grad(y, z, gin, create_graph=True)
        if modified == "indices":
            i.zero_()
        elif modified == "input":
            with torch.no_grad():
                z.add_(1)
        inputs = [[z, gin][w] for w in wrt]
        second = lambda: torch.autograd.grad(gx, inputs, counting(1, 1, 4, 4))
        if modified == "indices":
            try:
                second()
            except RuntimeError as e:
                assert str(e).startswith("one of the variables needed for gradient computation has been modified "
                                         "by an inplace operation"), str(e)
            else:
                raise AssertionError((modified, wrt))
            continue
        for w, t in zip(wrt, second()):
            assert t.flatten().tolist() == [[0.0] * 16, [6, 8, 14, 16]][w], (modified, wrt)

# max_pool2d_shapes_dtypes_and_meta
assert F.max_pool2d(torch.ones(0, 1, 4, 4), 2).shape == (0, 1, 2, 2)
assert F.max_pool2d(counting(2, 4, 4), 2).shape == (2, 2, 2)
for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16]:
    y = F.max_pool2d(torch.arange(16).reshape(1, 4, 4).to(dt), 2)
    assert y.dtype == dt and y.tolist() == [[[5, 7], [13, 15]]], dt
y, i = F.max_pool2d(torch.ones(2, 3, 7, 7, device="meta"), 3, stride=2, padding=1, return_indices=True)
assert y.shape == i.shape == (2, 3, 4, 4) and i.dtype == torch.int64 and i.device.type == "meta"
xm = torch.ones(2, 3, 7, 7, device="meta", requires_grad=True)
(gm,) = torch.autograd.grad(F.max_pool2d(xm, 3).sum(), xm)
assert gm.shape == (2, 3, 7, 7) and gm.device.type == "meta"

print("max_pool2d: all checks passed")

# ---------------------------------------------------------------- adaptive_avg_pool2d

# adaptive_avg_pool2d_windows_and_mean_path
# Windows [floor(i·In/Out), ceil((i+1)·In/Out)), summed in order and divided by their height,
# then their width.
y = F.adaptive_avg_pool2d(torch.arange(5.0, dtype=d64).reshape(1, 1, 1, 5), (1, 3))
assert y.flatten().tolist() == [0.5, 2.0, 3.5]
y = F.adaptive_avg_pool2d(torch.tensor([[[[0.0, 1.0]]]], dtype=d64), (1, 3))
assert y.flatten().tolist() == [0.0, 0.5, 1.0]
x = torch.tensor([[[[1e8, -1e8], [1.0, 1.0], [0.0, 0.0]]]])
assert F.adaptive_avg_pool2d(x, (2, 1)).flatten().tolist() == [0.5, 0.5]
half = torch.tensor([[[[2048.0], [1.0], [1.0], [1.0]]]], dtype=torch.float16)
assert F.adaptive_avg_pool2d(half.reshape(1, 1, 1, 4), (1, 2)).flatten().tolist() == [1024.0, 1.0]
# (1, 1) is a mean over the last two dimensions, recorded as MeanBackward1, for any rank >= 3.
x = leaf(2, 3, 4, 5)
assert chain(F.adaptive_avg_pool2d(x, (1, 1))) == [(0, "MeanBackward1"), (1, acc)]
assert chain(F.adaptive_avg_pool2d(x, (1, 2))) == [(0, "AdaptiveAvgPool2DBackward0"), (1, acc)]
assert F.adaptive_avg_pool2d(torch.ones(1, 1, 1, 4, 4), (1, 1)).shape == (1, 1, 1, 1, 1)
assert F.adaptive_avg_pool2d(torch.ones(1, 0, 3), (1, 1)).isnan().all()
raises(RuntimeError, "mean(): could not infer output dtype. Input dtype must be either a floating "
       "point or complex dtype. Got: Long",
       lambda: F.adaptive_avg_pool2d(torch.ones(1, 1, 4, 4, dtype=torch.int64), (1, 1)))
# A zero output size gives an empty output; PyTorch's backward then fails (homura gives zeros).
x = leaf(1, 1, 4, 4)
y = F.adaptive_avg_pool2d(x, (0, 2))
assert y.shape == (1, 1, 0, 2)
raises(RuntimeError, "adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero size for "
       "non-batch dimensions, but grad_output has sizes [1, 1, 0, 2] with dimension 2 being empty",
       lambda: torch.autograd.grad(y.sum(), x))
assert F.adaptive_avg_pool2d(torch.ones(0, 1, 4, 4), (2, 2)).shape == (0, 1, 2, 2)
assert F.adaptive_avg_pool2d(torch.ones(1, 0, 4, 4), (2, 2)).shape == (1, 0, 2, 2)
assert F.adaptive_avg_pool2d(torch.ones(2, 4, 4, device="meta"), (3, 5)).shape == (2, 3, 5)
# PyTorch computes inputs whose strides look channels-last with another kernel, which divides
# once by the window size: its results can differ in the last bit (homura has one kernel).
t = torch.randn(2, 8, 7, 5).contiguous(memory_format=torch.channels_last)
assert not torch.equal(F.adaptive_avg_pool2d(t, (3, 2)), F.adaptive_avg_pool2d(t.contiguous(), (3, 2)))

# adaptive_avg_pool2d_checks_its_arguments
for shape in [(), (4,), (4, 4)]:
    for size in [(1, 1), (2, 2)]:
        raises(ValueError, "Input dimension should be at least 3",
               lambda: F.adaptive_avg_pool2d(torch.ones(shape), size))
raises(RuntimeError, "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got [1, 1, 1, 4, 4]",
       lambda: F.adaptive_avg_pool2d(torch.ones(1, 1, 1, 4, 4), (2, 2)))
raises(RuntimeError, "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch "
       "dimensions, but input has sizes [1, 0, 4] with dimension 1 being empty",
       lambda: F.adaptive_avg_pool2d(torch.ones(1, 0, 4, dtype=torch.int64), (2, 2)))
raises(RuntimeError, "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch "
       "dimensions, but input has sizes [1, 1, 3, 0] with dimension 3 being empty",
       lambda: F.adaptive_avg_pool2d(torch.ones(1, 1, 3, 0), (2, 2)))
raises(NotImplementedError, "\"adaptive_avg_pool2d\" not implemented for 'Long'",
       lambda: F.adaptive_avg_pool2d(torch.ones(1, 4, 4, dtype=torch.int64), (2, 2)))
raises(NotImplementedError, "\"adaptive_avg_pool2d\" not implemented for 'Bool'",
       lambda: F.adaptive_avg_pool2d(torch.ones(1, 4, 4, dtype=torch.bool), (2, 2)))
# PyTorch checks the dtype only where it computes elements (homura rejects integer inputs
# everywhere).
assert F.adaptive_avg_pool2d(torch.ones(1, 1, 4, 4, dtype=torch.int64), (0, 2)).shape == (1, 1, 0, 2)
assert F.adaptive_avg_pool2d(torch.ones(1, 1, 4, 4, dtype=torch.int64, device="meta"), (2, 2)).shape == (1, 1, 2, 2)

# adaptive_avg_pool2d_nodes_and_saved_tensors
a = leaf(1, 2, 5, 4)
z = a * 1
y = F.adaptive_avg_pool2d(z, (2, 3))
assert sorted(n for n in dir(y.grad_fn) if n.startswith("_saved_")) == ["_saved_self"]
gin = torch.ones(1, 2, 2, 3, dtype=d64, requires_grad=True)
(gx,) = torch.autograd.grad(y, z, gin, create_graph=True)
assert chain(gx) == [(0, "AdaptiveAvgPool2DBackwardBackward0"), (1, acc), (1, "MulBackward0"), (2, acc)]
d_z, d_g = torch.autograd.grad((gx * counting(1, 2, 5, 4)).sum(), [z, gin], retain_graph=True)
assert d_z.abs().sum().item() == 0.0
assert torch.equal(d_g, F.adaptive_avg_pool2d(counting(1, 2, 5, 4), (2, 3)))
r = counting(1, 2, 5, 4).requires_grad_()
(gg,) = torch.autograd.grad(gx, gin, r, create_graph=True)
assert chain(gg) == [(0, "AdaptiveAvgPool2DBackward0"), (1, acc)]
z.add_(1)
raises(RuntimeError, None, lambda: torch.autograd.grad(y.sum(), a))

print("adaptive_avg_pool2d: all checks passed")
