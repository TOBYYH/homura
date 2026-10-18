"""Check the reference behavior that tests/nn_loss.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: the nodes the losses record, the values and
derivatives of log_sigmoid at its edges, the dtypes of mixed inputs, the argument errors of
nll_loss / cross_entropy and of binary_cross_entropy_with_logits / mse_loss, the constant
weights, label smoothing outside [0, 1], single samples with a [1] target, and the saved
tensors' version checks; then the review round 50 cases: log_sigmoid's gradients from its buffer,
NLL without classes, spatial weights in any layout, spatial byte targets and bool subtraction.
"""

import math
import warnings

import torch
import torch.nn.functional as F

import reference

reference.check()
warnings.simplefilter("ignore")
d64 = torch.float64


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


def var(*shape):
    return torch.randn(shape, dtype=d64, requires_grad=True)


# losses_record_the_reference_nodes
x, t = var(3, 4), torch.tensor([1, 0, 3])
w = torch.rand(4, dtype=d64)
p = torch.rand(3, 4, dtype=d64)
acc = "AccumulateGrad"
assert chain(F.mse_loss(x, p)) == [(0, "MseLossBackward0"), (1, acc)]
assert chain(F.mse_loss(var(3, 1), var(1, 4))) == [
    (0, "MseLossBackward0"), (1, "ExpandBackward0"), (2, acc), (1, "ExpandBackward0"), (2, acc)]
assert chain(F.nll_loss(x, t, weight=w)) == [(0, "NllLossBackward0"), (1, acc)]
assert chain(F.nll_loss(var(2, 4, 3), torch.zeros(2, 3, dtype=torch.long))) == [
    (0, "NllLoss2DBackward0"), (1, "ViewBackward0"), (2, acc)]
assert chain(F.nll_loss(var(2, 4, 3, 2), torch.zeros(2, 3, 2, dtype=torch.long))) == [
    (0, "NllLoss2DBackward0"), (1, acc)]
assert chain(F.cross_entropy(x, t)) == [
    (0, "NllLossBackward0"), (1, "LogSoftmaxBackward0"), (2, acc)]
assert chain(F.cross_entropy(x, t, weight=w, label_smoothing=0.1)) == [
    (0, "AddBackward0"),
    (1, "MulBackward1"), (2, "NllLossBackward0"), (3, "LogSoftmaxBackward0"), (4, acc),
    (1, "MulBackward1"), (2, "DivBackward0"), (3, "SumBackward0"), (4, "MaskedFillBackward0"),
    (5, "NegBackward0"), (6, "SumBackward1"), (7, "MulBackward0"), (8, "LogSoftmaxBackward0"),
    (9, acc)]
pt, pw = var(3, 4), var(4)
assert chain(F.cross_entropy(x, pt, weight=pw)) == [
    (0, "DivBackward1"), (1, "NegBackward0"), (2, "SumBackward0"), (3, "MulBackward0"),
    (4, "MulBackward0"), (5, "LogSoftmaxBackward0"), (6, acc), (5, acc),
    (4, "ViewBackward0"), (5, acc)]
# The mean of no positions: -(..).sum().fill_(NaN).
assert chain(F.cross_entropy(var(0, 4), torch.zeros(0, 4, dtype=d64))) == [
    (0, "NegBackward0"), (1, "FillBackward2"), (2, "SumBackward0"), (3, "MulBackward0"),
    (4, "LogSoftmaxBackward0"), (5, acc)]
assert chain(F.binary_cross_entropy_with_logits(x, pt, weight=w, pos_weight=w)) == [
    (0, "BinaryCrossEntropyWithLogitsBackward0"), (1, acc), (1, acc)]
assert chain(F.logsigmoid(x)) == [(0, "LogSigmoidBackward0"), (1, acc)]
assert F.cross_entropy(x.detach(), t).grad_fn is None
with torch.no_grad():
    assert F.binary_cross_entropy_with_logits(x, pt).grad_fn is None

# log_sigmoid_edges
v = torch.tensor([40.0, 0.0, -800.0, 800.0], dtype=d64, requires_grad=True)
out = F.logsigmoid(v)
assert out.tolist() == [-4.248354255291589e-18, -math.log(2), -800.0, 0.0]
(g,) = torch.autograd.grad(out.sum(), v, create_graph=True)
assert g.tolist() == [4.248354255291589e-18, 0.5, 1.0, 0.0]
(gg,) = torch.autograd.grad(g.sum(), v)
assert gg[1].item() == -0.25
# PyTorch's half precision gradient uses its buffer e^-|x| rounded to the dtype: subnormal here.
x16 = torch.tensor([15.0, 17.0], dtype=torch.float16, requires_grad=True)
(g16,) = torch.autograd.grad(F.logsigmoid(x16), x16, torch.full((2,), 1000.0, dtype=torch.float16))
assert g16.tolist() == [0.0002980232238769531, 5.9604644775390625e-05]
assert F.logsigmoid(torch.tensor(40.0, dtype=torch.float16)).item() == 0.0
assert math.copysign(1, F.logsigmoid(torch.tensor(40.0, dtype=torch.float16)).item()) == -1

# loss_dtypes_follow_pytorch
f32, f64 = torch.ones(2), torch.ones(2, dtype=d64)
bce = F.binary_cross_entropy_with_logits
assert bce(f32, f64 * 0.5).dtype == d64
assert bce(f64, f32 * 0.5).dtype == torch.float32
assert bce(f32, f32 * 0.5, weight=f64, pos_weight=f64).dtype == torch.float32
assert F.mse_loss(f32, torch.ones(2, dtype=torch.long)).dtype == torch.float32
assert F.mse_loss(f32, f64).dtype == d64
assert F.cross_entropy(torch.ones(2, 3, dtype=d64), torch.ones(2, 3) / 3).dtype == d64
assert F.cross_entropy(torch.ones(2, 3), torch.ones(2, 3, dtype=d64) / 3).dtype == d64
for half in (torch.float16, torch.bfloat16):
    assert F.cross_entropy(torch.ones(2, 3, dtype=half), torch.tensor([0, 2])).dtype == half
raises(RuntimeError, "expected scalar type Double but found Float",
       lambda: F.nll_loss(torch.ones(2, 3, dtype=d64), torch.tensor([0, 1]), weight=torch.ones(3)))
raises(RuntimeError, "expected scalar type Float but found Double",
       lambda: F.cross_entropy(torch.ones(2, 3), torch.tensor([0, 1]),
                               weight=torch.ones(3, dtype=d64)))

# nll_loss_and_cross_entropy_reject_invalid_arguments
x, t = torch.randn(3, 4, dtype=d64), torch.tensor([1, 0, 3])
raises(ValueError, "Expected 1 or more dimensions (got 0)",
       lambda: F.nll_loss(torch.tensor(1.0), torch.tensor(0)))
raises(ValueError, "Expected input batch_size (3) to match target batch_size (2).",
       lambda: F.nll_loss(x, torch.tensor([1, 0])))
raises(ValueError, "Expected input batch_size (3) to match target batch_size (0).",
       lambda: F.nll_loss(x, torch.tensor(2)))
raises(ValueError, "For 1D input, 1D target must have size 1, but got target size: 2",
       lambda: F.nll_loss(x[0], torch.tensor([2, 1])))
raises(RuntimeError, "0D or 1D target tensor expected, multi-target not supported",
       lambda: F.nll_loss(x, t.view(3, 1)))
for dtype, name in [(torch.int32, "Int"), (torch.bool, "Bool"), (torch.float32, "Float")]:
    raises(RuntimeError, f"expected target dtype to be Long or Byte, but got {name}",
           lambda: F.nll_loss(x, t.to(dtype)))
raises(RuntimeError, "weight tensor should be defined either for all 4 classes or no classes but "
       "got weight tensor of shape: [3]", lambda: F.nll_loss(x, t, weight=torch.ones(3, dtype=d64)))
raises(RuntimeError, "weight tensor should be defined either for all 4 classes or no classes but "
       "got weight tensor of shape: [1, 4]",
       lambda: F.nll_loss(x, t, weight=torch.ones(1, 4, dtype=d64)))
x3 = torch.randn(2, 4, 3, dtype=d64)
raises(RuntimeError, "Expected target size [2, 3], got [2, 2]",
       lambda: F.nll_loss(x3, torch.zeros(2, 2, dtype=torch.long)))
raises(RuntimeError, "Expected target size [2, 3], got [2]",
       lambda: F.nll_loss(x3, torch.zeros(2, dtype=torch.long)))
raises(RuntimeError, "weight tensor should be defined either for all or no classes",
       lambda: F.nll_loss(x3, torch.zeros(2, 3, dtype=torch.long), weight=torch.ones(3, dtype=d64)))
x4 = torch.randn(2, 4, 2, 3, dtype=d64)
raises(RuntimeError, "only batches of spatial targets supported (3D tensors) but got targets of "
       "dimension: 2", lambda: F.nll_loss(x4, torch.zeros(2, 6, dtype=torch.long)))
raises(RuntimeError, "size mismatch (got input: [2, 4, 2, 3] , target: [2, 2, 2]",
       lambda: F.nll_loss(x4, torch.zeros(2, 2, 2, dtype=torch.long)))
raises(RuntimeError, "expected target dtype to be Long or Byte, but got Int",
       lambda: F.nll_loss(x4, torch.zeros(2, 2, 3, dtype=torch.int32)))
# The 2-D implementation of more dimensions only counts the weight's elements.
assert F.nll_loss(x4, torch.zeros(2, 2, 3, dtype=torch.long),
                  weight=torch.ones(1, 4, dtype=d64)).isfinite()
for target in ([1, 4, 0], [1, -1, 0]):
    bad = next(v for v in target if not 0 <= v < 4)
    raises(IndexError, f"Target {bad} is out of bounds.", lambda: F.nll_loss(x, torch.tensor(target)))
# Without reduction, PyTorch checks the targets in parallel: with several bad ones, which it
# reports varies between runs.
raises(IndexError, "Target -2 is out of bounds.",
       lambda: F.nll_loss(x, torch.tensor([1, -2, 0]), reduction="none"))
raises(IndexError, "Target 255 is out of bounds.",
       lambda: F.nll_loss(x, torch.tensor([1, 255, 0], dtype=torch.uint8)))
raises(IndexError, "Target 9 is out of bounds.",
       lambda: F.nll_loss(x4, torch.full((2, 2, 3), 9, dtype=torch.long)))
F.nll_loss(x, torch.tensor([1, 255, 0], dtype=torch.uint8), ignore_index=255)
raises(NotImplementedError, None, lambda: F.nll_loss(torch.ones(3, 4, dtype=torch.long), t))
raises(RuntimeError, "Expected floating point type for target with class probabilities, got Long",
       lambda: F.cross_entropy(x, torch.ones(3, 4, dtype=torch.long)))
raises(RuntimeError, "Expected floating point type for target with class probabilities, got Long",
       lambda: F.cross_entropy(torch.tensor(1.0), torch.tensor(0)))
raises(RuntimeError, "ignore_index is not supported for floating point target",
       lambda: F.cross_entropy(x, x.softmax(1), ignore_index=0))
F.cross_entropy(x, x.softmax(1), ignore_index=-1)
raises(RuntimeError, "cross_entropy: weight tensor should be defined either for all 4 classes or "
       "no classes but got weight tensor of shape: [3]",
       lambda: F.cross_entropy(x, x.softmax(1), weight=torch.ones(3, dtype=d64)))
raises(RuntimeError, "weight tensor should be defined either for all 4 classes or no classes but "
       "got weight tensor of shape: [3]",
       lambda: F.cross_entropy(x, t, weight=torch.ones(3, dtype=d64), label_smoothing=0.1))
raises(NotImplementedError, None, lambda: F.cross_entropy(torch.ones(3, 4, dtype=torch.long), t))

# bce_and_mse_reject_invalid_arguments
y = torch.rand(3, 4, dtype=d64)
raises(ValueError, "Target size (torch.Size([4])) must be the same as input size "
       "(torch.Size([3, 4]))", lambda: bce(x, y[0]))
raises(RuntimeError, "output with shape [3, 4] doesn't match the broadcast shape [2, 3, 4]",
       lambda: bce(x, y, weight=torch.rand(2, 3, 4, dtype=d64)))
raises(RuntimeError, "output with shape [3, 4] doesn't match the broadcast shape [2, 3, 4]",
       lambda: bce(x, y, pos_weight=torch.rand(2, 3, 4, dtype=d64)))
raises(RuntimeError, "The size of tensor a (5) must match the size of tensor b (4) at "
       "non-singleton dimension 1", lambda: bce(x, y, pos_weight=torch.rand(5, dtype=d64)))
raises(RuntimeError, "result type Float can't be cast to the desired output type Long",
       lambda: bce(torch.ones(3), torch.ones(3, dtype=torch.long)))
raises(NotImplementedError, "Subtraction, the `-` operator, with two bool tensors is not supported. "
       "Use the `^` or `logical_xor()` operator instead.",
       lambda: bce(torch.ones(3), torch.ones(3, dtype=torch.bool)))
raises(NotImplementedError, None,
       lambda: bce(torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.long)))
raises(NotImplementedError, None, lambda: F.logsigmoid(torch.ones(3, dtype=torch.long)))
raises(RuntimeError, "The size of tensor a (3) must match the size of tensor b (4) at "
       "non-singleton dimension 0", lambda: F.mse_loss(torch.ones(3), torch.ones(4)))
for dtype in (torch.long, torch.bool):
    raises(NotImplementedError, None,
           lambda: F.mse_loss(torch.ones(3, dtype=dtype), torch.ones(3, dtype=dtype)))

# constant_weights_reject_requires_grad
x, t, y = var(3, 4), torch.tensor([1, 0, 3]), torch.rand(3, 4, dtype=d64)
wr = torch.rand(4, dtype=d64, requires_grad=True)
message = ("The function '{}' is not differentiable with respect to argument '{}'. This input "
           "cannot have requires_grad True.")
raises(RuntimeError, message.format("nll_loss_forward", "weight"),
       lambda: F.nll_loss(x, t, weight=wr))
raises(RuntimeError, message.format("nll_loss_forward", "weight"),
       lambda: F.nll_loss(x, t.view(3, 1), weight=wr))
raises(RuntimeError, message.format("nll_loss2d_forward", "weight"),
       lambda: F.nll_loss(var(2, 4, 3), torch.zeros(2, 3, dtype=torch.long), weight=wr))
raises(RuntimeError, message.format("nll_loss_forward", "weight"),
       lambda: F.cross_entropy(x, t, weight=wr))
raises(RuntimeError, message.format("nll_loss_forward", "weight"),
       lambda: F.cross_entropy(x, t, weight=wr, label_smoothing=0.1))
raises(RuntimeError, message.format("binary_cross_entropy_with_logits", "weight"),
       lambda: bce(x, y, weight=wr))
raises(RuntimeError, message.format("binary_cross_entropy_with_logits", "pos_weight"),
       lambda: bce(x, y, pos_weight=wr))
with torch.no_grad():
    F.nll_loss(x, t, weight=wr)
    bce(x, y, weight=wr, pos_weight=wr)
(gw,) = torch.autograd.grad(F.cross_entropy(x, x.detach().softmax(1), weight=wr), wr)
assert gw.shape == (4,)

# label_smoothing_outside_the_unit_interval
for smoothing in (-0.5, float("nan")):
    assert F.cross_entropy(x, t, label_smoothing=smoothing).item() == F.cross_entropy(x, t).item()
    probs = x.detach().softmax(1)
    assert (F.cross_entropy(x, probs, label_smoothing=smoothing).item()
            == F.cross_entropy(x, probs).item())
raises(RuntimeError, "label_smoothing must be between 0.0 and 1.0. Got: 1.5",
       lambda: F.cross_entropy(x, t, label_smoothing=1.5))
raises(RuntimeError, "label_smoothing must be between 0.0 and 1.0. Got: 1.5",
       lambda: F.cross_entropy(x, x.softmax(1), label_smoothing=1.5))

# single_samples_with_a_one_element_target: PyTorch's forward takes a [1] target for a 1-D
# input, its backward doesn't.
xs = var(4)
loss = F.nll_loss(xs, torch.tensor([2]), reduction="none")
assert loss.shape == () and loss.item() == F.nll_loss(xs, torch.tensor(2)).item()
raises(RuntimeError, "size mismatch (got input: [4], target: [1])", lambda: loss.backward())

# saved_tensors_follow_the_version_rules
modified = "one of the variables needed for gradient computation has been modified by an inplace"
x, t = var(3, 4), torch.tensor([1, 0, 3])
loss = F.nll_loss(x, t)
t.add_(0)
raises(RuntimeError, None, lambda: loss.backward())
w = torch.rand(4, dtype=d64)
loss = F.nll_loss(x, torch.tensor([1, 0, 3]), weight=w)
w.mul_(1)
raises(RuntimeError, None, lambda: loss.backward())
y = torch.rand(3, 4, dtype=d64)
loss = F.mse_loss(x, y)
y.add_(1)
raises(RuntimeError, None, lambda: loss.backward())
w = torch.rand(4, dtype=d64)
loss = bce(x, torch.rand(3, 4, dtype=d64), weight=w)
w.add_(1)
raises(RuntimeError, None, lambda: loss.backward())
loss = bce(x, torch.rand(3, 4, dtype=d64))
loss.backward()
raises(RuntimeError, None, lambda: loss.backward())


def logsigmoid_grad(x, g, create_graph=False):
    (grad,) = torch.autograd.grad(F.logsigmoid(x), x, g, create_graph=create_graph)
    return grad


def close(actual, expected):
    return actual != 0 and abs(actual - expected) <= 1e-6 * abs(expected)


# log_sigmoid_gradients_follow_the_reference_buffer
for dtype, v, g, expected in [(torch.float32, 90.0, 1e38, 0.08194008469581604),
                              (d64, 710.0, 1e308, 0.44762862256751296)]:
    for create_graph in (False, True):
        x = torch.tensor([v], dtype=dtype, requires_grad=True)
        grad = logsigmoid_grad(x, torch.tensor([g], dtype=dtype), create_graph)
        assert close(grad.item(), expected), (dtype, grad)
x = torch.tensor([-40.0, 0.0], dtype=d64, requires_grad=True)
grad = logsigmoid_grad(x, torch.tensor([1e18, 1.0], dtype=d64), True)
assert grad.tolist() == [1e18, 0.5]
assert type(grad.grad_fn).__name__ == "LogSigmoidBackwardBackward0"
(second,) = torch.autograd.grad(grad.sum(), x)
assert close(second[0].item(), -4.248354255291589) and second[1].item() == -0.25
x = torch.tensor([710.0], dtype=d64, requires_grad=True)
g = torch.tensor([1.0], dtype=d64, requires_grad=True)
(dg,) = torch.autograd.grad(logsigmoid_grad(x, g, True), g, torch.tensor([1e308], dtype=d64))
assert close(dg.item(), 0.44762862256751296)
x = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
assert logsigmoid_grad(x, torch.tensor([1000.0], dtype=torch.bfloat16)).item() == 268.0

# nll_loss_without_classes
for shape, target_shape in [((0,), ()), ((2, 0), (2,)), ((2, 0, 2, 2), (2, 2, 2))]:
    for reduction in ("none", "mean", "sum"):
        for w in (None, torch.zeros(0, dtype=d64)):
            x = torch.zeros(shape, dtype=d64, requires_grad=True)
            out = F.nll_loss(x, torch.full(target_shape, -100), weight=w, reduction=reduction)
            if reduction == "none":
                assert out.shape == (target_shape if len(shape) > 1 else ()) and (out == 0).all()
            elif reduction == "mean":
                assert out.isnan()
            else:
                assert out.item() == 0.0
            (grad,) = torch.autograd.grad(out, x, torch.full(out.shape, float("inf"), dtype=d64),
                                          create_graph=True)
            assert grad.shape == shape
            (second,) = torch.autograd.grad(grad.sum(), x)
            assert second.shape == shape


def nll_second_derivatives(x, target, weight, reduction):
    """The derivatives of sum(dx) with respect to x and the upstream gradient (ones)."""
    out = F.nll_loss(x, target, weight=weight, reduction=reduction)
    g = torch.ones(out.shape, dtype=out.dtype, device=out.device, requires_grad=True)
    (dx,) = torch.autograd.grad(out, x, g, create_graph=True)
    dxx, dg = torch.autograd.grad(dx.sum(), (x, g))
    return out, dxx, dg


def check_empty_upstream(out, dg, reduction):
    assert dg.shape == out.shape
    assert dg.isnan().all() if reduction == "mean" else (dg == 0).all()


# R63: the derivative with respect to the upstream gradient exists without classes too.
for shape, target_shape in [((0,), ()), ((2, 0), (2,)), ((2, 0, 2, 2), (2, 2, 2))]:
    for reduction in ("none", "mean", "sum"):
        for w in (None, torch.zeros(0, dtype=d64)):
            x = torch.zeros(shape, dtype=d64, requires_grad=True)
            out, dxx, dg = nll_second_derivatives(x, torch.full(target_shape, -100), w, reduction)
            assert dxx.shape == shape
            check_empty_upstream(out, dg, reduction)
for shape, target in [((2, 3), torch.full((2,), -100)), ((0, 3), torch.zeros(0, dtype=torch.long))]:
    for reduction in ("none", "mean", "sum"):
        x = torch.zeros(shape, dtype=d64, requires_grad=True)
        out, dxx, dg = nll_second_derivatives(x, target, None, reduction)
        assert dxx.shape == shape
        check_empty_upstream(out, dg, reduction)
x = torch.zeros(2, 0, device="meta", requires_grad=True)
out, dxx, dg = nll_second_derivatives(x, torch.zeros(2, dtype=torch.long, device="meta"), None,
                                      "none")
assert dxx.shape == (2, 0) and dg.shape == (2,) and dg.is_meta
x = torch.zeros(2, 0, dtype=d64, requires_grad=True)
t = torch.full((2,), -100)
assert F.cross_entropy(x, t, reduction="sum").item() == 0.0
assert F.cross_entropy(x, t, weight=torch.zeros(0, dtype=d64), label_smoothing=0.1,
                       reduction="none").isnan().all()
loss = F.cross_entropy(x, t, weight=torch.zeros(0, dtype=d64), label_smoothing=0.1)
assert loss.isnan() and torch.autograd.grad(loss, x)[0].shape == (2, 0)
raises(IndexError, "Target 0 is out of bounds.", lambda: F.nll_loss(x, torch.tensor([0, -100])))

# spatial_weights_are_version_checked_in_any_layout
def spatial_run(weight, modify, retain):
    x = torch.ones(1, 4, 1, 1, dtype=d64, requires_grad=True)
    loss = F.nll_loss(x, torch.zeros(1, 1, 1, dtype=torch.long), weight=weight)
    if retain:
        loss.backward(retain_graph=True)
    modify()
    return torch.autograd.grad(loss, x)[0].flatten().tolist()


def base():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=d64)


for transposed in (False, True):
    make = (lambda: base().t()) if transposed else base
    assert spatial_run(make(), lambda: None, False) == [-1.0, 0.0, 0.0, 0.0]
    for retain in (False, True):
        w = make()
        raises(RuntimeError, None, lambda: spatial_run(w, lambda: w.add_(1), retain))
    b = base()
    w = b.t() if transposed else b.view(2, 2)
    raises(RuntimeError, None, lambda: spatial_run(w, lambda: b[1].mul_(2), False))

# spatial_byte_targets_extend_pytorch: the checks allow Byte, the CPU kernel doesn't.
for shape, target_shape in [((2, 3, 2), (2, 2)), ((2, 3, 2, 2), (2, 2, 2))]:
    t = (torch.arange(math.prod(target_shape)) % 3).view(target_shape)
    x = torch.randn(shape, dtype=d64)
    F.nll_loss(x, t)
    raises(RuntimeError, "expected scalar type Long but found Byte",
           lambda: F.nll_loss(x, t.to(torch.uint8)))

# bool_operands_of_subtraction_are_rejected
single = ("Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying "
          "to invert a mask, use the `~` or `logical_not()` operator instead.")
two = ("Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or "
       "`logical_xor()` operator instead.")
mask = torch.tensor([True, False])
longs, floats = torch.tensor([2, 3]), torch.tensor([2.0, 3.0])
for f in (lambda: mask - 1, lambda: 1 - mask, lambda: longs - mask, lambda: floats - mask,
          lambda: mask - 1.5, lambda: longs - True, lambda: torch.zeros(0, dtype=torch.bool) - 1):
    raises(NotImplementedError, single, f)
raises(NotImplementedError, two, lambda: mask - mask)
raises(NotImplementedError, two, lambda: mask - True)
# PyTorch's meta kernels skip the check; homura rejects bool subtraction on Meta too.
assert (torch.zeros(2, dtype=torch.bool, device="meta") - 1).dtype == torch.long
target = longs.clone()
raises(NotImplementedError, single, lambda: target.sub_(mask))
assert target.tolist() == [2, 3] and target._version == 0
x = torch.randn(2, dtype=d64, requires_grad=True)
y = x * 2
saved = (y * y).sum()
raises(NotImplementedError, single, lambda: y.sub_(mask))
assert y._version == 0 and type(y.grad_fn).__name__ == "MulBackward0"
saved.backward()

print("check_m2_loss: all reference assertions passed")
