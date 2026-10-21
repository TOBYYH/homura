"""Check the reference behavior that tests/nn_batch_norm.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: the node that batch_norm records, its
argument errors and their order (with the mixed dtype rules), the update of the running
statistics, the checks of their writes (expanded or overlapping statistics, stale views),
which graphs save them, parameters matched by element count, the second
derivatives of a batch norm with a bias but no weight (which PyTorch gets wrong), signed zeros
and updates without grad mode, inputs without elements, Meta tensors, gradcheck and the
gradients of each subset of the inputs. Where homura differs, the block asserts PyTorch's
behavior and says so.
"""

import math
import warnings

import torch
import torch.nn.functional as F

import reference

reference.check()
warnings.simplefilter("ignore")
d64 = torch.float64
acc = "AccumulateGrad"


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


def constant(*shape, seed=0):
    """The values of the test's constant(): both signs, 0.3 to 1.3 in magnitude."""
    n = math.prod(shape)
    values = [(0.3 + ((i * 7 + seed * 3) % 11) / 10) * (-1 if (i + seed) % 3 == 0 else 1)
              for i in range(n)]
    return torch.tensor(values, dtype=d64).reshape(shape)


def var(*shape, seed=0):
    return constant(*shape, seed=seed).requires_grad_()


def stats(c):
    return constant(c, seed=5) * 0.5, constant(c, seed=6).abs() + 0.3


def train(x, rm=None, rv=None, w=None, b=None, momentum=0.1, **kw):
    return F.batch_norm(x, rm, rv, w, b, training=True, momentum=momentum, **kw)


def grads(out, inputs, g, create_graph=False):
    return torch.autograd.grad(out, inputs, g, create_graph=create_graph, allow_unused=True)


# batch_norm_records_the_reference_node
x, w, b = var(4, 3, 2), var(3, seed=1), var(3, seed=2)
rm, rv = stats(3)
node = [(0, "NativeBatchNormBackward0"), (1, acc), (1, acc), (1, acc)]
for training in [True, False]:
    out = F.batch_norm(x, rm.clone(), rv.clone(), w, b, training=training)
    assert chain(out) == node
assert chain(train(x, w=w, b=b)) == node
nf = train(x, b=b).grad_fn.next_functions
assert len(nf) == 3 and nf[0][0] is not None and nf[1][0] is None and nf[2][0] is not None
assert chain(train(x)) == [(0, "NativeBatchNormBackward0"), (1, acc)]
# PyTorch's gradients are one NativeBatchNormBackwardBackward0 (homura records the formulas).
for training in [True, False]:
    out = F.batch_norm(x, rm.clone(), rv.clone(), w, b, training=training)
    first = grads(out, [x, w, b], var(4, 3, 2, seed=3), create_graph=True)
    assert [t.grad_fn.name() for t in first] == ["NativeBatchNormBackwardBackward0"] * 3
with torch.no_grad():
    assert train(x, rm.clone(), rv.clone(), w, b).grad_fn is None

# batch_norm_checks_its_arguments
x = constant(4, 3)
rm, rv = stats(3)
w, b = constant(3, seed=1), constant(3, seed=2)
one = "Expected more than 1 value per channel when training, got input size"
raises(ValueError, f"{one} torch.Size([1, 3])", lambda: train(constant(1, 3)))
raises(ValueError, f"{one} torch.Size([1])", lambda: train(constant(1)))
raises(ValueError, f"{one} torch.Size([1, 3])", lambda: train(constant(1, 3), eps=0.0))
raises(ValueError, "batch_norm eps must be positive during training, but got 0.0",
       lambda: train(x, eps=0.0))
raises(ValueError, "batch_norm eps must be non-negative, but got -1.0",
       lambda: F.batch_norm(x, rm, rv, eps=-1.0))
train(x, eps=math.nan)
train(constant(1, 3, 2))
F.batch_norm(constant(1, 3), rm, rv)
# Fewer than 2 dimensions: PyTorch's messages are an IndexError or about running_mean (homura
# reports the dimension).
raises(RuntimeError, "running_mean should contain 0 elements not 3",
       lambda: F.batch_norm(constant(3), rm, rv))
raises(IndexError, "Dimension out of range (expected to be in range of [-1, 0], but got 1)",
       lambda: train(constant(3)))
raises(IndexError, "tuple index out of range", lambda: train(torch.tensor(1.0, dtype=d64)))
raises(RuntimeError, "running_var must be defined in evaluation mode",
       lambda: F.batch_norm(x, rm, None))
for m, v in [(None, rv), (None, None)]:
    raises(RuntimeError, "running_mean must be defined in evaluation mode",
           lambda: F.batch_norm(x, m, v))
both = "running_mean and running_var must either both be None or neither be None"
raises(ValueError, both, lambda: train(x, rm.clone(), None))
raises(ValueError, both, lambda: train(x, None, rv.clone()))
four, five = constant(4, seed=3), constant(5, seed=4)
raises(RuntimeError, "running_mean should contain 3 elements not 4", lambda: train(x, four))
raises(RuntimeError, "running_mean should contain 3 elements not 4",
       lambda: F.batch_norm(x, four, rv, five))
raises(RuntimeError, "running_var should contain 3 elements not 4", lambda: F.batch_norm(x, rm, four))
raises(RuntimeError, "weight should contain 3 elements not 5",
       lambda: F.batch_norm(x, rm, rv, five, constant(6, seed=5)))
raises(RuntimeError, "weight should contain 3 elements not 5", lambda: F.batch_norm(x, rm, rv, five))
raises(RuntimeError, "bias should contain 3 elements not 4", lambda: F.batch_norm(x, rm, rv, w, four))
grad = ("The function 'native_batch_norm' is not differentiable with respect to argument '{}'. "
        "This input cannot have requires_grad True.")
rm_grad, rv_grad = rm.clone().requires_grad_(), rv.clone().requires_grad_()
raises(RuntimeError, grad.format("running_mean"), lambda: F.batch_norm(x, rm_grad, rv))
raises(RuntimeError, grad.format("running_var"), lambda: F.batch_norm(x, rm, rv_grad))
raises(RuntimeError, grad.format("running_mean"), lambda: train(x, rm_grad, None))
raises(RuntimeError, "running_mean should contain 3 elements not 4",
       lambda: F.batch_norm(x, four.clone().requires_grad_(), rv))
raises(RuntimeError, grad.format("running_mean"),
       lambda: F.batch_norm(x, rm_grad.float(), rv.float()))
with torch.no_grad():
    F.batch_norm(x, rm_grad, rv)
raises(NotImplementedError, "\"batch_norm\" not implemented for 'Long'",
       lambda: F.batch_norm(x.long(), rm.long(), rv.long()))
raises(NotImplementedError, "\"batch_norm\" not implemented for 'Long'",
       lambda: F.batch_norm(x.long(), rm, rv))
raises(NotImplementedError, "\"batch_norm\" not implemented for 'Bool'", lambda: train(x > 0))
same = "mixed dtype (CPU): all inputs must share same datatype."
flt = "mixed dtype (CPU): expect parameter to have scalar type of Float"
f32, f16, bf16 = torch.float32, torch.float16, torch.bfloat16
cases = [
    (x, [None, None, rm.float(), rv.float()], same),
    (x, [w.float(), None, rm, rv], flt),
    (x.float(), [w, None, rm.float(), rv.float()], flt),
    (x.float(), [w.float(), b, None, None], "expected scalar type Float but found Double"),
    (x.float(), [w.float(), None, rm, rv], "expected scalar type Float but found Double"),
    (x.float(), [w.half(), None, None, None], flt),
    (x.half(), [w.float(), None, rm.half(), rv.half()], flt),
    (x.half(), [w.half(), None, rm.float(), rv.float()], "expected scalar type Half but found Float"),
    (x.half(), [w.to(bf16), None, None, None], flt),
    (x.to(bf16), [w.float(), b.half(), None, None], flt),
    (x.half(), [w.half(), b.float(), None, None], "expected scalar type Half but found Float"),
]
for inp, (cw, cb, crm, crv), message in cases:
    raises(RuntimeError, message, lambda: F.batch_norm(inp, crm, crv, cw, cb, training=crm is None))
F.batch_norm(x.half(), rm.float(), rv.float(), w.float(), b.float())
meta = lambda t: t.to("meta")
raises(RuntimeError, "Tensor on device cpu is not on the expected device meta!",
       lambda: F.batch_norm(meta(x), rm, rv))
raises(RuntimeError, "Tensor on device meta is not on the expected device cpu!",
       lambda: F.batch_norm(x, rm, rv, meta(w)))

# batch_norm_updates_the_running_statistics
x = constant(4, 3, 2)
rm0, rv0 = stats(3)
rm, rv = rm0.clone(), rv0.clone()
train(x, rm, rv, momentum=0.3)
mean, variance = x.mean((0, 2)), x.var((0, 2), correction=1)
expected_mean, expected_var = rm0 * 0.7 + mean * 0.3, rv0 * 0.7 + variance * 0.3
assert torch.allclose(rm, expected_mean, rtol=1e-15, atol=1e-15)
assert torch.allclose(rv, expected_var, rtol=1e-15, atol=1e-15)
F.batch_norm(x, rm, rv)
train(x)
assert torch.equal(rm, rm) and torch.allclose(rm, expected_mean, rtol=1e-15, atol=1e-15)
for f in [0.0, 1.0, 2.0]:
    rm, rv = rm0.clone(), rv0.clone()
    train(x, rm, rv, momentum=f)
    assert torch.allclose(rm, rm0 * (1 - f) + mean * f, rtol=1e-15, atol=1e-15)
    assert torch.allclose(rv, rv0 * (1 - f) + variance * f, rtol=1e-15, atol=1e-15)
rm, rv = rm0.clone(), rv0.clone()
train(x, rm, rv, momentum=math.nan)
assert rm.isnan().all() and rv.isnan().all()
storage = torch.zeros(3, 2, dtype=d64)
rv = rv0.clone()
train(x, storage[:, 0], rv, momentum=0.3)
assert torch.allclose(storage[:, 0], mean * 0.3, rtol=1e-15, atol=1e-15)
assert torch.equal(storage[:, 1], torch.zeros(3, dtype=d64))
# PyTorch rejects running statistics with more than one dimension (homura takes C elements).
raises(RuntimeError, "TensorAccessor expected 1 dims but tensor has 2",
       lambda: train(x, rm0.clone(), rv0.reshape(1, 3).clone(), momentum=0.3))
raises(RuntimeError, "TensorAccessor expected 1 dims but tensor has 2",
       lambda: F.batch_norm(x, rm0.reshape(3, 1), rv0))
rm, rv = rm0.float(), rv0.float()
raises(RuntimeError, "expected scalar type Float but found Double",
       lambda: train(x.float(), rm, rv, constant(3, seed=1).float(), constant(3, seed=2), momentum=0.3))
# PyTorch updates the running statistics before it checks the bias's dtype (homura checks
# everything before it writes); a count error comes before any write.
assert not torch.equal(rm, rm0.float()) and not torch.equal(rv, rv0.float())
rm, rv = rm0.clone(), rv0.clone()
raises(RuntimeError, "weight should contain 3 elements not 5", lambda: train(x, rm, rv, constant(5)))
assert torch.equal(rm, rm0) and torch.equal(rv, rv0)
assert torch.is_grad_enabled()

# batch_norm_checks_the_running_statistics_before_writing
# PyTorch writes into expanded statistics and doesn't check overlaps; homura rejects both
# before any write. PyTorch updates the channels in a parallel loop, so the result of channels
# sharing one element depends on the threads (seen with 2 and 24 threads: [8, 8], [4.75, 4.75],
# [7.5, 7.5], [1, 1]); with one thread each channel updates the shared element in turn.
threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    x = torch.tensor([[1.0, 10.0], [3.0, 20.0]], dtype=d64)
    rm, rv = torch.zeros(2, dtype=d64), torch.ones(1, dtype=d64).expand(2)
    train(x, rm, rv, momentum=0.5)
    # 0.5·50 + 0.5·(0.5·2 + 0.5·1).
    assert rm.tolist() == [1.0, 7.5] and rv.tolist() == [25.75, 25.75] and rm._version == 0
    rm, rv = torch.zeros(1, dtype=d64).expand(2), torch.ones(2, dtype=d64)
    train(x, rm, rv, momentum=0.5)
    # 0.5·15 + 0.5·(0.5·2 + 0.5·0).
    assert rm.tolist() == [8.0, 8.0]
finally:
    torch.set_num_threads(threads)
F.batch_norm(x, torch.zeros(1, dtype=d64).expand(2), torch.ones(1, dtype=d64).expand(2))
# A running mean that is a view of the input: the forward pass normalizes with the batch mean 3
# and writes 2 into the input; the backward pass combines the saved statistics with the written
# input (homura would recompute the statistics from it, so it rejects the overlap).
x = torch.tensor([[1.0], [3.0], [5.0]], dtype=d64, requires_grad=True)
rm = x.detach().select(0, 0)
out = train(x, rm, torch.ones(1, dtype=d64), momentum=0.5)
assert x.detach().flatten().tolist() == [2.0, 3.0, 5.0] and x._version == 0
(dx,) = grads(out, [x], torch.tensor([[1.0], [2.0], [4.0]], dtype=d64))
mean, invstd = 3.0, 1 / math.sqrt(8 / 3 + 1e-5)
xhat = (torch.tensor([2.0, 3.0, 5.0], dtype=d64) - mean) * invstd
gv = torch.tensor([1.0, 2.0, 4.0], dtype=d64)
expected = (gv - gv.mean() - xhat * (gv * xhat).mean()) * invstd
assert torch.allclose(dx.flatten(), expected, rtol=1e-12, atol=0), (dx, expected)
# A running mean that is the weight: the output uses the updated weight; one tensor for both
# statistics gets both updates.
x = constant(4, 2)
w = torch.ones(2, dtype=d64)
mean, unbiased = x.mean(0), x.var(0)
out = train(x, w, torch.ones(2, dtype=d64), w, momentum=0.5)
assert torch.allclose(w, 0.5 + 0.5 * mean, rtol=1e-15, atol=0)
assert torch.allclose(out, F.batch_norm(x, None, None, w, training=True), rtol=1e-15, atol=1e-15)
s = torch.ones(2, dtype=d64)
train(x, s, s, momentum=0.5)
assert torch.allclose(s, 0.5 * (0.5 + 0.5 * mean) + 0.5 * unbiased, rtol=1e-15, atol=0)
buffer = torch.ones(2, 2, dtype=d64)
train(x, buffer[0], buffer[1], momentum=0.5)
columns = torch.ones(2, 2, dtype=d64)
train(x, columns[:, 0], columns[:, 1], momentum=0.5)
assert not torch.equal(columns, torch.ones(2, 2, dtype=d64))
train(constant(0, 2), torch.ones(1, dtype=d64).expand(2), torch.ones(1, dtype=d64).expand(2))

# batch_norm_rejects_stale_views_before_writing
# PyTorch replays a view whose base was modified in place, for the input, the weight or the
# bias, and updates the statistics (homura rejects such views before writing them).
for which in range(3):
    def stale_view(values, shape):
        base = torch.tensor(values, dtype=d64, requires_grad=True).reshape(shape).mul(1)
        view = base.view(shape)
        base.mul_(2)
        return view
    x = torch.tensor([[1.0, 10.0], [3.0, 20.0]], dtype=d64)
    inp = stale_view([1.0, 10.0, 3.0, 20.0], (2, 2)) if which == 0 else x
    w = stale_view([1.0, 1.0], (2,)) if which == 1 else None
    b = stale_view([0.0, 0.0], (2,)) if which == 2 else None
    rm, rv = torch.zeros(2, dtype=d64), torch.ones(2, dtype=d64)
    out = train(inp, rm, rv, w, b, momentum=0.5)
    assert not torch.equal(rm, torch.zeros(2, dtype=d64)) and out.grad_fn is not None
    out.sum().backward()

# batch_norm_saves_the_running_statistics_only_in_evaluation_mode
x = var(4, 3, 2)
rm, rv = stats(3)
g = constant(4, 3, 2, seed=3)
first = train(x, rm, rv)
second = train(x, rm, rv)
(both,) = torch.autograd.grad((first * g).sum() + (second * g).sum(), [x])
(single,) = grads(train(x, *stats(3)), [x], g)
assert torch.allclose(both, 2 * single, rtol=1e-14, atol=1e-14)
# PyTorch saves the running statistics in training too, but its update doesn't increment their
# versions: an evaluation graph's backward pass after an update succeeds with the new values
# (homura: a version error).
eval_out = F.batch_norm(x, rm, rv)
versions = (rm._version, rv._version)
old_var = rv.clone()
train(x, rm, rv)
assert (rm._version, rv._version) == versions
(gx,) = grads(eval_out, [x], g)
assert torch.allclose(gx, g / (rv.view(1, 3, 1) + 1e-5).sqrt(), rtol=1e-14, atol=0)
assert not torch.allclose(gx, g / (old_var.view(1, 3, 1) + 1e-5).sqrt())
before = (rm.clone(), rv.clone())
train(constant(0, 3, 2), rm, rv)
assert torch.equal(rm, before[0]) and torch.equal(rv, before[1])

# batch_norm_takes_parameters_by_element_count
x = var(4, 3, 2)
w, b = var(3, 1, seed=1), var(1, 3, seed=2)
wf, bf = var(3, seed=1), var(3, seed=2)
g = constant(4, 3, 2, seed=3)
for training in [True, False]:
    out = F.batch_norm(x, *stats(3), w, b, training=training)
    flat = F.batch_norm(x, *stats(3), wf, bf, training=training)
    assert torch.equal(out, flat)
    # PyTorch's backward pass fails: "TensorAccessor expected 1 dims but tensor has 2" here,
    # "Function NativeBatchNormBackward0 returned an invalid gradient at index 1 - got [3] but
    # expected shape compatible with [3, 1]" for [N, C] inputs in training.
    raises(RuntimeError, "TensorAccessor expected 1 dims but tensor has 2",
           lambda: grads(out, [x, w, b], g))
out = train(var(4, 3), None, None, w, b)
raises(RuntimeError, "Function NativeBatchNormBackward0 returned an invalid gradient at index 1 - "
       "got [3] but expected shape compatible with [3, 1]", lambda: grads(out, [w], constant(4, 3)))

# batch_norm_bias_without_weight_has_second_derivatives
x, b = var(4, 3, 2), var(3, seed=1)
g = var(4, 3, 2, seed=2)
out = train(x, b=b)
gx, gb = grads(out, [x, b], g, create_graph=True)
ggx, ggb = constant(4, 3, 2, seed=3), constant(3, seed=4)
raises(RuntimeError, "Expected a proper Tensor but got None (or an undefined Tensor in C++) for "
       "argument #1 'other'",
       lambda: torch.autograd.grad((gx * ggx).sum() + (gb * ggb).sum(), [x, b, g], allow_unused=True))
plain = train(x)
(px,) = grads(plain, [x], g, create_graph=True)
sx, sg = torch.autograd.grad((px * ggx).sum(), [x, g])
# The bias's gradient alone, the sum of g over all but the channels, doesn't depend on g in
# PyTorch's graph (homura: the broadcast of ggb, added to the plain normalization's derivative).
out = train(x, b=b)
gx, gb = grads(out, [x, b], g, create_graph=True)
assert torch.autograd.grad((gb * ggb).sum(), [g], allow_unused=True) == (None,)
assert torch.equal(gb.detach(), g.detach().sum((0, 2)))

# batch_norm_edge_cases
x = torch.tensor([[-0.0, 1.0], [0.0, -1.0]], dtype=d64)
out = F.batch_norm(x, torch.zeros(2, dtype=d64), torch.ones(2, dtype=d64), eps=0.0)
assert out.tolist() == [[0.0, 1.0], [0.0, -1.0]] and not torch.signbit(out[:, 0]).any()
rm, rv = stats(3)
rm0, rv0 = rm.clone(), rv.clone()
rm.requires_grad_(), rv.requires_grad_()
with torch.no_grad():
    train(constant(4, 3), rm, rv)
assert not torch.equal(rm, rm0) and not torch.equal(rv, rv0)
assert rm.requires_grad and rm.grad_fn is None
x = torch.tensor([[1000.0], [1000.0], [1000.5]], dtype=f16)
assert train(x).tolist() == [[0.0], [0.0], [2.12109375]]
assert train(x, w=torch.ones(1)).tolist() == [[-0.70751953125], [-0.70751953125], [1.4140625]]

# batch_norm_without_elements
w, b = var(3, seed=1), var(3, seed=2)
for shape in [(0, 3, 2), (2, 3, 0)]:
    x = var(*shape)
    rm, rv = stats(3)
    out = train(x, rm, rv, w, b)
    assert out.shape == shape and type(out.grad_fn).__name__ == "AddBackward0"
    assert torch.equal(rm, stats(3)[0]) and torch.equal(rv, stats(3)[1])
    gx, gw, gb = grads(out, [x, w, b], constant(*shape, seed=3))
    assert gx.shape == shape and torch.equal(gw, torch.zeros(3, dtype=d64))
    assert torch.equal(gb, torch.zeros(3, dtype=d64))
x = var(2, 0, 3)
assert train(x, torch.zeros(0, dtype=d64), torch.zeros(0, dtype=d64)).shape == (2, 0, 3)
# PyTorch skips the checks after eps (homura doesn't).
empty = constant(0, 3)
assert F.batch_norm(empty, None, None).shape == (0, 3)
assert train(empty, w=constant(5)).shape == (0, 3)

# batch_norm_on_meta
x = torch.zeros(4, 3, 2, dtype=d64, device="meta", requires_grad=True)
w = torch.zeros(3, dtype=d64, device="meta", requires_grad=True)
for training in [True, False]:
    rm, rv = torch.zeros(3, dtype=d64, device="meta"), torch.zeros(3, dtype=d64, device="meta")
    out = F.batch_norm(x, rm, rv, w, training=training)
    assert (out.shape, out.dtype, out.device.type) == ((4, 3, 2), d64, "meta")
    gx, gw = grads(out, [x, w], torch.zeros(4, 3, 2, dtype=d64, device="meta"))
    assert gx.shape == (4, 3, 2) and gw.shape == (3,) and gw.device.type == "meta"
s = torch.zeros(3, device="meta")
half = train(torch.zeros(4, 3, dtype=f16, device="meta"), s, s.clone())
assert half.dtype == f16
raises(RuntimeError, "mean(): could not infer output dtype. Input dtype must be either a floating "
       "point or complex dtype. Got: torch.int64",
       lambda: train(torch.zeros(4, 3, dtype=torch.long, device="meta")))
# PyTorch's Meta kernel accepts mixed dtypes (homura applies the CPU rules).
assert F.batch_norm(x, s, s).dtype == d64

# batch_norm_gradients_pass_gradcheck
for shape in [(5, 3), (3, 2, 2, 2)]:
    c = shape[1]
    x, w, b = var(*shape), var(c, seed=1), var(c, seed=2)
    for training in [True, False]:
        f = lambda x, w, b: F.batch_norm(x, *stats(c), w, b, training=training)
        assert torch.autograd.gradcheck(f, (x, w, b))
        assert torch.autograd.gradgradcheck(f, (x, w, b))
        assert torch.autograd.gradcheck(lambda x: train(x), (x,))
        assert torch.autograd.gradgradcheck(lambda x: train(x), (x,))
x = var(3, 4, 2).transpose(0, 1)
w = var(3, seed=1)
assert torch.autograd.gradcheck(lambda x, w: train(x, w=w), (x, w))
assert torch.autograd.gradgradcheck(lambda x, w: train(x, w=w), (x, w))

# batch_norm_gradients_of_each_subset
g = constant(4, 3, 2, seed=3)
for training in [True, False]:
    make = lambda: (constant(4, 3, 2), constant(3, seed=1), constant(3, seed=2))
    run = lambda x, w, b: F.batch_norm(x, *stats(3), w, b, training=training)
    inputs = [t.requires_grad_() for t in make()]
    full = grads(run(*inputs), inputs, g)
    for mask in range(1, 8):
        inputs = make()
        chosen = [i for i in range(3) if mask & (1 << i)]
        for i in chosen:
            inputs[i].requires_grad_()
        result = grads(run(*inputs), [inputs[i] for i in chosen], g)
        for i, t in zip(chosen, result):
            assert torch.equal(t, full[i]), (mask, i)

print("check_m2_batch_norm: all reference assertions passed")
