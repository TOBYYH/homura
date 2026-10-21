"""Check the reference behavior that tests/nn_cnn_layers.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: the construction, initialization and checks
of nn.Conv2d, the buffers, counting and modes of nn.BatchNorm2d, the pooling layers, nn.Flatten,
the layers as wrappers of their functions, the CNN of step 9 and the layers as modules. Values
that come from random numbers (homura's differ) are checked through their bounds only. The
setters that homura adds (a new bias, groups, statistics after construction) have no PyTorch
API; their blocks check what PyTorch's constructor gives for the same options.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

import reference

reference.check()
warnings.simplefilter("ignore")
torch.manual_seed(0)


def raises(exc, message, f):
    """Asserts that f raises exc with the first line message."""
    try:
        f()
    except exc as e:
        assert str(e).splitlines()[0] == message, (str(e), message)
        return
    raise AssertionError(f"no {exc.__name__}: {message}")


def constant(shape, seed):
    """tests/nn_cnn_layers.rs's constant: distinct values of both signs."""
    n = math.prod(shape)
    values = []
    for i in range(n):
        magnitude = 0.3 + ((i * 7 + seed * 3) % 11) / 10 + i * 1e-3
        values.append(-magnitude if (i + seed) % 3 == 0 else magnitude)
    return torch.tensor(values, dtype=torch.float64).reshape(shape)


def var(shape, seed):
    return constant(shape, seed).requires_grad_()


def grads(out, inputs, g):
    return torch.autograd.grad(out, inputs, g, allow_unused=True)


def keys(m):
    return list(m.state_dict())


def same_values(a, b):
    """Same dtype, shape and values with zero tolerance (torch.equal: 0.0 equals -0.0)."""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


# conv2d_layers_initialize_like_pytorch
conv = nn.Conv2d(3, 8, 3, padding=1)
assert keys(conv) == ["weight", "bias"]
assert conv.weight.shape == (8, 3, 3, 3) and conv.bias.shape == (8,)
assert conv.weight.requires_grad and conv.weight.is_leaf
assert conv.kernel_size == (3, 3)
assert repr(conv) == "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))"
bound = 1 / math.sqrt(27)
for t in [conv.weight, conv.bias]:
    assert t.abs().max() <= bound and (t != 0).any()
conv = nn.Conv2d(4, 6, (3, 2), stride=(2, 1), dilation=2, groups=2, bias=False)
assert keys(conv) == ["weight"]
assert conv.weight.shape == (6, 2, 3, 2)
assert conv.weight.abs().max() <= 1 / math.sqrt(12)
assert repr(conv) == ("Conv2d(4, 6, kernel_size=(3, 2), stride=(2, 1), dilation=(2, 2), "
                      "groups=2, bias=False)")
assert repr(nn.Conv2d(4, 6, 3, padding="valid")) == \
    "Conv2d(4, 6, kernel_size=(3, 3), stride=(1, 1), padding=valid)"
assert repr(nn.Conv2d(4, 6, 3, padding=0)) == "Conv2d(4, 6, kernel_size=(3, 3), stride=(1, 1))"
for c, o, k in [(0, 6, 3), (3, 0, 3), (3, 4, 0)]:
    conv = nn.Conv2d(c, o, k)
    assert conv.weight.numel() == 0
    if c == 0 or k == 0:
        assert (conv.bias == 0).all()
conv = nn.Conv2d(2, 2, 2)
before = conv.weight.detach().clone()
conv.reset_parameters()
assert not torch.equal(conv.weight, before)
conv = nn.Conv2d(3, 2, 3, device="meta")
assert conv.weight.is_meta
y = conv(torch.zeros(5, 3, 4, 4, device="meta"))
assert y.shape == (5, 2, 2, 2) and y.is_meta

# conv2d_layers_check_their_options
raises(ValueError, "groups must be a positive integer", lambda: nn.Conv2d(4, 6, 3, groups=0))
raises(ValueError, "in_channels must be divisible by groups", lambda: nn.Conv2d(4, 6, 3, groups=3))
raises(ValueError, "out_channels must be divisible by groups", lambda: nn.Conv2d(4, 6, 3, groups=4))
same = "padding='same' is not supported for strided convolutions"
raises(ValueError, same, lambda: nn.Conv2d(4, 6, 3, padding="same", stride=2))
raises(ValueError, same, lambda: nn.Conv2d(4, 6, 3, padding="same", stride=(1, 2)))
conv = nn.Conv2d(4, 6, 3, padding="same", dilation=2)
assert conv(torch.ones(1, 4, 5, 5)).shape == (1, 6, 5, 5)
conv = nn.Conv2d(4, 6, 3, stride=0)
raises(RuntimeError, "non-positive stride is not supported", lambda: conv(torch.ones(1, 4, 5, 5)))

# conv2d_setters_keep_the_parameters_options: the layers PyTorch constructs for the options.
for options in [{"dtype": torch.float64}, {"device": "meta"}]:
    conv = nn.Conv2d(4, 6, 3, groups=2, **options)
    for t in [conv.weight, conv.bias]:
        assert t.dtype == options.get("dtype", torch.float32)
        assert t.is_meta == ("device" in options)
        assert t.requires_grad and t.is_leaf
    assert conv.weight.shape == (6, 2, 3, 3)
conv = nn.Conv2d(4, 6, 3, groups=2)
for t in [conv.weight, conv.bias]:
    assert t.abs().max() <= 1 / math.sqrt(18)
# Some of the 108 weights exceed the bound of twice the fan_in (all below has odds 0.707^108).
assert conv.weight.abs().max() > 1 / math.sqrt(36)
assert nn.Conv2d(4, 6, 3, groups=2, bias=False).bias is None

BN_KEYS = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]

# batch_norm_layers_initialize_like_pytorch
bn = nn.BatchNorm2d(3)
assert keys(bn) == BN_KEYS
sd = bn.state_dict()
assert sd["weight"].tolist() == [1.0] * 3 and sd["bias"].tolist() == [0.0] * 3
assert sd["running_mean"].tolist() == [0.0] * 3 and sd["running_var"].tolist() == [1.0] * 3
nbt = sd["num_batches_tracked"]
assert nbt.shape == () and nbt.dtype == torch.int64 and nbt.item() == 0
assert [k for k, _ in bn.named_parameters()] == ["weight", "bias"]
assert [k for k, _ in bn.named_buffers()] == BN_KEYS[2:]
assert repr(bn) == ("BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, bias=True, "
                    "track_running_stats=True)")
bn = nn.BatchNorm2d(3, eps=1e-3, momentum=None, affine=False, track_running_stats=False)
assert keys(bn) == [] and list(bn.named_buffers()) == []
assert repr(bn) == ("BatchNorm2d(3, eps=0.001, momentum=None, affine=False, bias=False, "
                    "track_running_stats=False)")
assert keys(nn.BatchNorm2d(3, bias=False)) == ["weight", *BN_KEYS[2:]]
assert keys(nn.BatchNorm2d(3, track_running_stats=False)) == ["weight", "bias"]
bn = nn.BatchNorm2d(3).double()
assert [t.dtype for t in bn.state_dict().values()] == [torch.float64] * 4 + [torch.int64]

# batch_norm_setters_keep_the_members_options: the layers PyTorch constructs for the options.
for options in [{"dtype": torch.float64}, {"device": "meta"}]:
    bn = nn.BatchNorm2d(3, **options)
    dtype = options.get("dtype", torch.float32)
    for t in [bn.weight, bn.bias, bn.running_mean, bn.running_var]:
        assert t.dtype == dtype and t.is_meta == ("device" in options)
    assert bn.num_batches_tracked.dtype == torch.int64
    assert bn.num_batches_tracked.is_meta == ("device" in options)
    assert bn.num_batches_tracked.shape == ()
    assert bn.weight.requires_grad and bn.weight.is_leaf and bn.bias.requires_grad
assert nn.BatchNorm2d(3, affine=False, bias=True).bias is None
bn = nn.BatchNorm2d(3, device="meta")
assert [(t.dtype, t.device.type) for t in bn.state_dict().values()] == \
    [(torch.float32, "meta")] * 4 + [(torch.int64, "meta")]

# batch_norm_layers_match_pytorch: the golden data (gen_m2_cnn_layers.py).

# batch_norm_layers_count_like_pytorch
x = constant((3, 2, 2, 2), 0)
bn = nn.BatchNorm2d(2).double()


def count(bn):
    return bn.num_batches_tracked.item()


raises(ValueError, "expected 4D input (got 3D input)", lambda: bn(constant((2, 2, 2), 0)))
assert count(bn) == 0
raises(ValueError, "Expected more than 1 value per channel when training, got input size "
       "torch.Size([1, 2, 1, 1])", lambda: bn(constant((1, 2, 1, 1), 0)))
raises(RuntimeError, "running_mean should contain 3 elements not 2",
       lambda: bn(constant((2, 3, 2, 2), 0)))
assert count(bn) == 2
assert bn.running_mean.tolist() == [0.0] * 2 and bn.running_var.tolist() == [1.0] * 2
with torch.no_grad():
    bn(x)
assert count(bn) == 3
bn.eval()
with torch.enable_grad():
    bn(x)
assert count(bn) == 3

bn = nn.BatchNorm2d(2, momentum=None).double()
bn.num_batches_tracked = None
bn(x)
assert bn.running_mean.tolist() == [0.0] * 2 and bn.running_var.tolist() == [1.0] * 2
bn = nn.BatchNorm2d(2).double()
bn.track_running_stats = False
y = bn(x)
assert count(bn) == 0 and bn.running_mean.tolist() == [0.0] * 2
w, b = bn.weight, bn.bias
assert same_values(y, F.batch_norm(x, None, None, w, b, True))
bn.eval()
assert same_values(bn(x), F.batch_norm(x, bn.running_mean, bn.running_var, w, b, False))
bn.running_mean = None
raises(RuntimeError, "running_mean must be defined in evaluation mode", lambda: bn(x))
bn.running_var = None
assert same_values(bn(x), F.batch_norm(x, None, None, w, b, True))
bn = nn.BatchNorm2d(2)
bn.eval()
bn.running_var = None
raises(RuntimeError, "running_var must be defined in evaluation mode", lambda: bn(x.float()))

bn = nn.BatchNorm2d(2).double()
xv = var((3, 2, 2, 2), 1)
bn(xv)
bn(xv)
assert count(bn) == 2
bn.eval()
out = bn(xv)
bn.reset_running_stats()
assert count(bn) == 0
assert bn.running_mean.tolist() == [0.0] * 2 and bn.running_var.tolist() == [1.0] * 2
assert not bn.training
inplace = "one of the variables needed for gradient computation has been modified by an inplace operation"
try:
    grads(out, [xv], constant((3, 2, 2, 2), 2))
    raise AssertionError("no error")
except RuntimeError as e:
    assert str(e).startswith(inplace), str(e)
bn.weight.detach().fill_(2.0)
bn.train()
out = bn(xv)
bn.reset_parameters()
assert bn.weight.tolist() == [1.0] * 2 and count(bn) == 0 and bn.training
try:
    grads(out, [xv], constant((3, 2, 2, 2), 2))
    raise AssertionError("no error")
except RuntimeError as e:
    assert str(e).startswith(inplace), str(e)
nn.BatchNorm2d(2, track_running_stats=False).reset_running_stats()

# batch_norm_layers_reject_a_zero_cumulative_count
def with_count(dtype, count, momentum):
    bn = nn.BatchNorm2d(2, momentum=momentum).to(dtype)
    state = bn.state_dict()
    state["num_batches_tracked"] = torch.tensor(count)
    bn.load_state_dict(state, strict=True)
    return bn


for dtype in [torch.float64, torch.float32]:
    x = constant((3, 2, 2, 2), 0).to(dtype)
    for stats in [True, False]:
        for grad in [True, False]:
            bn = with_count(dtype, -1, None)
            if stats:
                bn.eval()
                xv = x.clone().requires_grad_()
                out = bn(xv)
                bn.train()
                versions = (bn.running_mean._version, bn.running_var._version)
            else:
                bn.running_mean = None
                bn.running_var = None
            with torch.set_grad_enabled(grad):
                raises(ZeroDivisionError, "division by zero", lambda: bn(x))
            assert count(bn) == 0
            if stats:
                assert bn.running_mean.tolist() == [0.0] * 2
                assert bn.running_var.tolist() == [1.0] * 2
                assert (bn.running_mean._version, bn.running_var._version) == versions
                assert grads(out, [xv], torch.ones_like(out))[0] is not None
    for start, factor in [(-2, -1.0), (0, 1.0), (3, 0.25)]:
        bn = with_count(dtype, start, None)
        rm, rv = bn.running_mean.clone(), bn.running_var.clone()
        want = F.batch_norm(x, rm, rv, bn.weight, bn.bias, True, factor)
        assert same_values(bn(x), want)
        assert same_values(bn.running_mean, rm) and same_values(bn.running_var, rv)
        assert count(bn) == start + 1
    bn = with_count(dtype, -1, 0.1)
    bn(x)
    assert count(bn) == 0
    bn = with_count(dtype, -1, None)
    bn.eval()
    bn(x)
    assert count(bn) == -1
    bn = with_count(dtype, -1, None)
    bn.track_running_stats = False
    bn(x)
    assert count(bn) == -1
    bn = with_count(dtype, -1, None)
    bn.num_batches_tracked = None
    bn(x)

# batch_norm_layers_on_meta
x = torch.zeros(2, 2, 3, 3, device="meta")
bn = nn.BatchNorm2d(2).to("meta")
y = bn(x)
assert y.shape == x.shape and y.is_meta
bn = nn.BatchNorm2d(2, momentum=None).to("meta")
raises(RuntimeError, "Tensor.item() cannot be called on meta tensors", lambda: bn(x))
bn.eval()
assert bn(x).shape == x.shape

# batch_norm_layers_load_state_dicts: a state dict of the current version without the count
# fails a strict load, like homura's; a non-strict load lists it. PyTorch takes a state dict
# without version metadata for one of an old version and keeps the count (homura has no
# versions and reports it missing).
bn = nn.BatchNorm2d(2)
bn.num_batches_tracked.fill_(7)
other = nn.BatchNorm2d(2)
other.load_state_dict(bn.state_dict(), strict=True)
assert count(other) == 7
current = bn.state_dict()
del current["num_batches_tracked"]
raises(RuntimeError, "Error(s) in loading state_dict for BatchNorm2d:",
       lambda: other.load_state_dict(current, strict=True))
assert other.load_state_dict(current, strict=False).missing_keys == ["num_batches_tracked"]
assert count(other) == 7
old = {k: v for k, v in bn.state_dict().items() if k != "num_batches_tracked"}
other.num_batches_tracked.fill_(5)
assert other.load_state_dict(old, strict=True).missing_keys == []
assert count(other) == 5

# pooling_layers_apply_their_functions
x = var((2, 3, 5, 6), 0)
pool = nn.MaxPool2d(2)
assert keys(pool) == [] and list(pool.named_buffers()) == []
assert repr(pool) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)"
assert same_values(pool(x), F.max_pool2d(x, 2))
pool = nn.MaxPool2d((3, 2), stride=1, padding=(1, 0), dilation=(1, 2), ceil_mode=True,
                    return_indices=True)
values, indices = pool(x)
want, want_indices = F.max_pool2d(x, (3, 2), 1, (1, 0), (1, 2), True, True)
assert same_values(values, want) and same_values(indices, want_indices)
assert type(values.grad_fn).__name__ == "MaxPool2DWithIndicesBackward0"
raises(RuntimeError, "stride should not be zero", lambda: nn.MaxPool2d(2, stride=0)(x))
pool = nn.AdaptiveAvgPool2d(1)
assert keys(pool) == [] and repr(pool) == "AdaptiveAvgPool2d(output_size=1)"
y = pool(x)
assert same_values(y, F.adaptive_avg_pool2d(x, (1, 1)))
assert type(y.grad_fn).__name__ == "MeanBackward1"
y = nn.AdaptiveAvgPool2d((2, 4))(x)
assert same_values(y, F.adaptive_avg_pool2d(x, (2, 4)))
assert type(y.grad_fn).__name__ == "AdaptiveAvgPool2DBackward0"

# flatten_layers_merge_dimensions
flatten = nn.Flatten()
assert (flatten.start_dim, flatten.end_dim) == (1, -1) and keys(flatten) == []
assert repr(flatten) == "Flatten(start_dim=1, end_dim=-1)"
x = var((2, 3, 4), 0)
y = flatten(x)
assert y.shape == (2, 12) and type(y.grad_fn).__name__ == "ViewBackward0"
with torch.no_grad():
    x[0, 0].fill_(5.0)
assert y[0, :4].tolist() == [5.0] * 4
x = var((2, 3, 4), 1)
t = x.transpose(1, 2)
z = flatten(t)
assert same_values(z, t.contiguous().reshape(2, 12))
assert type(z.grad_fn).__name__ == "UnsafeViewBackward0"
with torch.no_grad():
    x.fill_(5.0)
assert (z != 5.0).all()
g = constant((2, 12), 2)
assert same_values(grads(z, [x], g)[0], g.reshape(2, 4, 3).transpose(1, 2))
assert nn.Flatten(-2, -1)(x).shape == (2, 12) and nn.Flatten(0, 1)(x).shape == (6, 4)
assert flatten(torch.zeros(0, 3, 4)).shape == (0, 12)
for x in [torch.tensor(1.0), torch.ones(3)]:
    raises(IndexError, "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
           lambda: flatten(x))
raises(IndexError, "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
       lambda: nn.Flatten(-4, -1)(torch.ones(2, 3, 4)))
raises(RuntimeError, "flatten() has invalid args: start_dim cannot come after end_dim",
       lambda: nn.Flatten(2, 1)(torch.ones(2, 3, 4)))


# layers_compute_their_functions: PyTorch's layers give the values of their functions too, also
# for half precision inputs with F32 batch normalization parameters.
def output_and_grads(f, x, params):
    out = f(x)
    return [out, *grads(out, [x, *params], torch.ones_like(out))]


for dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
    x = constant((2, 4, 5, 5), 0).to(dtype).requires_grad_()
    conv = nn.Conv2d(4, 6, 3, stride=2, padding=1, groups=2).to(dtype)
    w, b = conv.weight, conv.bias
    if dtype in (torch.float16, torch.bfloat16):
        # This CPU can't compute PyTorch's half precision convolution gradients (as in
        # check_m2_conv_pool.py): compare the outputs.
        assert same_values(conv(x), F.conv2d(x, w, b, 2, 1, 1, 2))
        try:
            output_and_grads(conv, x, [w, b])
            raise AssertionError("PyTorch computed a half precision convolution gradient")
        except RuntimeError as e:
            assert "DNNL does not support bf16/f16 backward" in str(e), e
    else:
        got = output_and_grads(conv, x, [w, b])
        want = output_and_grads(lambda x: F.conv2d(x, w, b, 2, 1, 1, 2), x, [w, b])
        assert all(same_values(a, e) for a, e in zip(got, want))
    for p in [dtype] + ([torch.float32] if dtype in (torch.float16, torch.bfloat16) else []):
        for training in [True, False]:
            bn = nn.BatchNorm2d(4).to(p)
            with torch.no_grad():
                bn.weight.copy_(constant((4,), 1))
                bn.bias.copy_(constant((4,), 2))
            bn.train(training)
            w, b = bn.weight, bn.bias
            rm, rv = bn.running_mean.clone(), bn.running_var.clone()
            want = output_and_grads(lambda x: F.batch_norm(x, rm, rv, w, b, training), x, [w, b])
            got = output_and_grads(bn, x, [w, b])
            assert all(same_values(a, e) for a, e in zip(got, want)), (dtype, p, training)
            assert same_values(bn.running_mean, rm) and same_values(bn.running_var, rv)
    got = output_and_grads(nn.MaxPool2d(3, 2, 1), x, [])
    assert all(same_values(a, e) for a, e in zip(got, output_and_grads(
        lambda x: F.max_pool2d(x, 3, 2, 1), x, [])))
    for size in [(1, 1), (2, 3)]:
        got = output_and_grads(nn.AdaptiveAvgPool2d(size), x, [])
        assert all(same_values(a, e) for a, e in zip(got, output_and_grads(
            lambda x: F.adaptive_avg_pool2d(x, size), x, [])))
    got = output_and_grads(nn.Flatten(), x, [])
    assert all(same_values(a, e) for a, e in zip(got, output_and_grads(
        lambda x: x.flatten(1, -1), x, [])))

# cnn_matches_pytorch: the keys (the values are golden data).
model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                      nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))
assert keys(model) == ["0.weight", "0.bias", *[f"1.{k}" for k in BN_KEYS], "4.weight",
                       "4.bias", "7.weight", "7.bias"]

# layers_pass_gradcheck
x = var((2, 4, 4, 4), 0)
for groups in [1, 2]:
    conv = nn.Conv2d(4, 2, (2, 3), padding=(1, 0), stride=(1, 2), groups=groups).double()
    inputs = (x, conv.weight, conv.bias)
    f = lambda x, w, b: F.conv2d(x, w, b, (1, 2), (1, 0), 1, groups)  # noqa: E731
    assert torch.autograd.gradcheck(f, inputs) and torch.autograd.gradgradcheck(f, inputs)
for training in [True, False]:
    bn = nn.BatchNorm2d(4).double()
    with torch.no_grad():
        bn.weight.copy_(constant((4,), 1))
    bn.train(training)
    f = lambda x, w, b: F.batch_norm(x, bn.running_mean, bn.running_var, w, b,  # noqa: E731
                                     training)
    assert torch.autograd.gradcheck(f, (x, bn.weight, bn.bias))
    if training:  # PyTorch's double backward of evaluation mode needs all three.
        assert torch.autograd.gradgradcheck(f, (x, bn.weight, bn.bias))
assert torch.autograd.gradcheck(lambda x: F.max_pool2d(x, 2, 1), (x,))
assert torch.autograd.gradgradcheck(lambda x: F.max_pool2d(x, 2, 1), (x,))
for size in [(1, 1), (3, 2)]:
    assert torch.autograd.gradcheck(lambda x: F.adaptive_avg_pool2d(x, size), (x,))
assert torch.autograd.gradcheck(lambda x: nn.Flatten(1, 2)(x.transpose(1, 3)), (x,))
assert torch.autograd.gradgradcheck(lambda x: nn.Flatten(1, 2)(x.transpose(1, 3)), (x,))

# layers_are_modules
model = model.double()
for k, t in model.state_dict().items():
    assert t.dtype == (torch.int64 if k.endswith("num_batches_tracked") else torch.float64), k
x = constant((2, 3, 4, 4), 0)
model(x)
model.eval()
model(x)
assert model.state_dict()["1.num_batches_tracked"].item() == 1
model.train()
model(x)
assert model.state_dict()["1.num_batches_tracked"].item() == 2
other = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                      nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4)).double()
other.load_state_dict(model.state_dict())
other.eval()
model.eval()
assert same_values(other(x), model(x))
x = torch.ones(2, 3, 4, 4)
shapes = [list(layer(x).shape) for layer in [nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3),
                                               nn.MaxPool2d(1), nn.AdaptiveAvgPool2d(2),
                                               nn.Flatten()]]
assert shapes == [[2, 3, 4, 4], [2, 3, 4, 4], [2, 3, 4, 4], [2, 3, 2, 2], [2, 48]]

print("check_m2_cnn_layers: all reference behavior matches")
