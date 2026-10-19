"""Check the reference behavior that tests/nn_embed_norm.rs and tests/nn_dropout.rs encode; does
not write fixtures.

Each block mirrors a test there, in the same order: the nodes that embedding, layer_norm and
rms_norm record, embedding's argument errors and its padding rows and saved indices,
layer_norm's argument errors (with its mixed dtype rules) and edge cases, the second derivatives
of a layer norm with a bias but no weight (which PyTorch gets wrong), rms_norm's argument
errors and default eps, the layers' defaults; then dropout: its noise values, the branches that
return the input without drawing, the probability checks, integer inputs and its gradient.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

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


def var(*shape, dtype=d64):
    return torch.randn(shape, dtype=dtype, requires_grad=True)


acc = "AccumulateGrad"

# embed_norm_record_the_reference_nodes
w = var(5, 3)
idx = torch.tensor([[1, 4], [0, 1]])
assert chain(F.embedding(idx, w, padding_idx=1)) == [(0, "EmbeddingBackward0"), (1, acc)]
assert chain(F.embedding(torch.tensor(2), w)) == [(0, "EmbeddingBackward0"), (1, acc)]
x, lw, lb = var(2, 3, 4), var(4), var(4)
assert chain(F.layer_norm(x, [4], lw, lb)) == [
    (0, "NativeLayerNormBackward0"), (1, acc), (1, acc), (1, acc)]
assert chain(F.layer_norm(x, [3, 4])) == [(0, "NativeLayerNormBackward0"), (1, acc)]
rw = var(4)
assert chain(F.rms_norm(x, [4], rw)) == [
    (0, "MulBackward0"), (1, "MulBackward0"), (2, acc), (2, "RsqrtBackward0"),
    (3, "AddBackward1"), (4, "MeanBackward1"), (5, "PowBackward0"), (6, acc), (1, acc)]
xh, wh = var(2, 4, dtype=torch.float16), var(4, dtype=torch.float16)
names = [n for _, n in chain(F.rms_norm(xh, [4], wh))]
assert names[0] == "ToCopyBackward0" and names.count("ToCopyBackward0") == 3, names
with torch.no_grad():
    assert F.embedding(idx, w).grad_fn is None
    assert F.layer_norm(x, [4], lw, lb).grad_fn is None
    assert F.rms_norm(x, [4], rw).grad_fn is None

# embedding_checks_its_arguments
idx = torch.tensor([1, 0])
for p in [5, -6]:
    raises(AssertionError, "Padding_idx must be within num_embeddings",
           lambda: F.embedding(idx, w, padding_idx=p))
for shape in [(5,), (5, 3, 2)]:
    raises(RuntimeError, "'weight' must be 2-D", lambda: F.embedding(idx, var(*shape)))
for dtype, name in [(torch.int16, "torch.ShortTensor"), (torch.uint8, "torch.ByteTensor"),
                    (torch.bool, "CPUBoolType"), (torch.float32, "torch.FloatTensor")]:
    raises(RuntimeError, "Expected tensor for argument #1 'indices' to have one of the following "
           f"scalar types: Long, Int; but got {name} instead (while checking arguments for "
           "embedding)", lambda: F.embedding(idx.to(dtype), w))
for bad in [5, -1]:
    raises(IndexError, "index out of range in self",
           lambda: F.embedding(torch.tensor([0, bad]), w))
# Tables without columns or rows reject out of range indices with index_select's messages;
# homura reports "index out of range in self" for both (review round 57).
raises(RuntimeError, "INDICES element is out of DATA bounds, id=3 axis_dim=3",
       lambda: F.embedding(torch.tensor([3]), var(3, 0)))
raises(RuntimeError, "index_select(): self indexing axis dim should be positive",
       lambda: F.embedding(torch.tensor([0]), var(0, 3)))
out = F.embedding(torch.tensor([0, 4]), w, padding_idx=-5)
(grad,) = torch.autograd.grad(out, w, torch.ones(2, 3, dtype=d64))
assert grad[0].tolist() == [0.0] * 3 and grad[4].tolist() == [1.0] * 3
meta_w = torch.zeros(5, 3, device="meta", requires_grad=True)
out = F.embedding(torch.zeros(2, 4, dtype=torch.long, device="meta"), meta_w, padding_idx=0)
assert out.shape == (2, 4, 3)
(grad,) = torch.autograd.grad(out.sum(), meta_w)
assert grad.device.type == "meta" and grad.shape == (5, 3)

# embedding_padding_rows_and_saved_indices
w = torch.tensor([[1.0, 2.0], [3.0, INF], [5.0, 6.0]], dtype=d64, requires_grad=True)
idx = torch.tensor([1, 0, 1])
out = F.embedding(idx, w, padding_idx=1)
assert out.flatten().tolist() == [3.0, INF, 1.0, 2.0, 3.0, INF]
g = torch.tensor([[NAN, 1.0], [2.0, 3.0], [INF, 4.0]], dtype=d64)
(grad,) = torch.autograd.grad(out, w, g)
assert grad.flatten().tolist() == [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
g = torch.ones(3, 2, dtype=d64, requires_grad=True)
(grad,) = torch.autograd.grad(F.embedding(idx, w), w, g, create_graph=True)
assert grad.flatten().tolist() == [1.0, 1.0, 2.0, 2.0, 0.0, 0.0]
assert type(grad.grad_fn).__name__ == "EmbeddingDenseBackwardBackward0"
dw, dg = torch.autograd.grad(grad.sum(), [w, g], allow_unused=True)
assert dw is None and dg.flatten().tolist() == [1.0] * 6
idx = torch.tensor([0, 2])
out = F.embedding(idx, w)
idx.add_(0)
raises(RuntimeError, None, lambda: out.sum().backward())
table = torch.arange(6).reshape(3, 2)
assert F.embedding(torch.tensor([2, 0]), table).flatten().tolist() == [4, 5, 0, 1]
assert F.embedding(idx, torch.ones(3, 2, dtype=torch.bool)).dtype == torch.bool
wt = var(2, 3).t()
rows = F.embedding(torch.tensor([[2, 1]]), wt)
assert rows.is_contiguous() and torch.equal(rows[0], wt[[2, 1]])
assert F.embedding(torch.tensor([2, 2], dtype=torch.int32), w).flatten().tolist() == [5.0, 6.0] * 2
empty = F.embedding(torch.zeros(0, 3, dtype=torch.long), w, padding_idx=0)
assert empty.shape == (0, 3, 2)
(grad,) = torch.autograd.grad(empty, w, torch.zeros(0, 3, 2, dtype=d64))
assert grad.flatten().tolist() == [0.0] * 6

# layer_norm_checks_its_arguments
x = var(2, 3, 4)
raises(RuntimeError, "Expected normalized_shape to be at least 1-dimensional, i.e., containing "
       "at least one element, but got normalized_shape = []", lambda: F.layer_norm(x, []))
raises(RuntimeError, "Expected weight to be of same shape as normalized_shape, but got weight of "
       "shape [3] and normalized_shape = [4]", lambda: F.layer_norm(x, [4], var(3)))
raises(RuntimeError, "Expected bias to be of same shape as normalized_shape, but got bias of "
       "shape [2, 2] and normalized_shape = [4]", lambda: F.layer_norm(x, [4], None, var(2, 2)))
raises(RuntimeError, "Given normalized_shape=[2, 4], expected input with shape [*, 2, 4], but "
       "got input of size[2, 3, 4]", lambda: F.layer_norm(x, [2, 4]))
raises(RuntimeError, "Given normalized_shape=[2, 4], expected input with shape [*, 2, 4], but "
       "got input of size[4]", lambda: F.layer_norm(var(4), [2, 4]))
raises(NotImplementedError, "\"LayerNormKernelImpl\" not implemented for 'Long'",
       lambda: F.layer_norm(torch.ones(2, 4, dtype=torch.long), [4]))


def p(dtype):
    return None if dtype is None else torch.ones(4, dtype=dtype)


def ln(xd, wd, bd):
    return F.layer_norm(torch.ones(2, 4, dtype=xd), [4], p(wd), p(bd))


f16, bf16, f32 = torch.float16, torch.bfloat16, torch.float32
for xd, wd, bd in [(f32, d64, None), (f32, f16, None), (f16, bf16, None), (f16, f32, f16),
                   (f32, None, torch.long)]:
    raises(RuntimeError, "mixed dtype (CPU): expect parameter to have scalar type of Float",
           lambda: ln(xd, wd, bd))
raises(RuntimeError, "mixed dtype (CPU): all inputs must share same datatype.",
       lambda: ln(d64, f32, None))
for xd, bd, names in [(f32, d64, ("Float", "Double")), (f16, f32, ("Half", "Float"))]:
    raises(RuntimeError, "expected scalar type {} but found {}".format(*names),
           lambda: ln(xd, xd, bd))
for xd, wd, bd in [(f16, f32, f32), (bf16, f32, None), (f16, None, f32)]:
    assert ln(xd, wd, bd).dtype == xd

# layer_norm_edge_cases
y = F.layer_norm(torch.tensor([[1.0, 3.0]]), [2], torch.tensor([0.0, 1.0]), None, 0.0)
assert y.tolist() == [[0.0, 1.0]] and math.copysign(1.0, y[0, 0].item()) == 1.0
xs = torch.tensor([[1.0, 2.0]])
assert F.layer_norm(xs, [2], eps=-1.0).isnan().all()
assert F.layer_norm(xs, [2], eps=NAN).isnan().all()
x, w, b = var(2, 0), var(0), var(0)
out = F.layer_norm(x, [0], w, b)
assert out.shape == (2, 0)
assert [t.shape for t in torch.autograd.grad(out, [x, w, b], torch.zeros(2, 0, dtype=d64))] == [
    (2, 0), (0,), (0,)]
x = torch.zeros(2, 3, 4, dtype=f16, device="meta", requires_grad=True)
w = torch.zeros(3, 4, device="meta", requires_grad=True)
b = torch.zeros(3, 4, device="meta", requires_grad=True)
out = F.layer_norm(x, [3, 4], w, b)
assert out.dtype == f16 and out.shape == (2, 3, 4)
grads = torch.autograd.grad(out, [x, w, b], torch.zeros(2, 3, 4, dtype=f16, device="meta"))
assert [(t.dtype, t.shape) for t in grads] == [(f16, (2, 3, 4)), (f32, (3, 4)), (f32, (3, 4))]
x, w, b = torch.randn(3, 4, dtype=d64), torch.randn(4, dtype=d64), var(4)
g = torch.randn(3, 4, dtype=d64)
(grad,) = torch.autograd.grad(F.layer_norm(x, [4], w, b), b, g)
assert torch.equal(grad, g.sum(0))
out = F.layer_norm(x, [4], w, b)
x.add_(1.0)
raises(RuntimeError, None, lambda: torch.autograd.grad(out, b, g))
for changed in [1, 2]:
    x, w, b = var(2, 4), var(4), var(4)
    out = F.layer_norm(x, [4], w, b)
    with torch.no_grad():
        [w, b][changed - 1].add_(1.0)
    raises(RuntimeError, None, lambda: out.sum().backward())

# layer_norm_bias_without_weight_has_second_derivatives: PyTorch's double backward drops the
# bias's gradient without a weight: d(dbias)/dg is None instead of gg, and together with the
# input's gradient it raises.
x, b, g = var(3, 4), var(4), var(3, 4)
gx, gb = torch.autograd.grad(F.layer_norm(x, [4], None, b), [x, b], g, create_graph=True)
gg = torch.randn(4, dtype=d64)
assert torch.autograd.grad((gb * gg).sum(), [x, b, g], allow_unused=True,
                           retain_graph=True) == (None, None, None)
raises(RuntimeError, "Expected a proper Tensor but got None (or an undefined Tensor in C++) for "
       "argument #1 'other'",
       lambda: torch.autograd.grad((gx * torch.randn(3, 4, dtype=d64)).sum() + (gb * gg).sum(),
                                   [x, g]))
# With a weight, the same derivative is gg, as homura computes it without a weight too.
x, w, b, g = var(3, 4), var(4), var(4), var(3, 4)
gb = torch.autograd.grad(F.layer_norm(x, [4], w, b), [x, w, b], g, create_graph=True)[2]
(dg,) = torch.autograd.grad((gb * gg).sum(), g)
assert torch.equal(dg, gg.expand(3, 4))

# rms_norm_checks_its_arguments
x = var(2, 4)
raises(RuntimeError, "Expected normalized_shape to be at least 1-dimensional, i.e., containing "
       "at least one element, but got normalized_shape = []", lambda: F.rms_norm(x, []))
raises(RuntimeError, "Expected weight to be of same shape as normalized_shape, but got weight of "
       "shape [3] and normalized_shape = [4]", lambda: F.rms_norm(x, [4], var(3)))
raises(RuntimeError, "Given normalized_shape=[2, 4], expected input with shape [*2, 4], but got "
       "input of size[2, 3, 4]", lambda: F.rms_norm(var(2, 3, 4), [2, 4]))
raises(ValueError, "Input tensor must have at least 2 dimensions, but got 1",
       lambda: F.rms_norm(var(4), [2, 4]))
raises(NotImplementedError, "\"rms_norm\" not implemented for 'Long'",
       lambda: F.rms_norm(torch.ones(2, 4, dtype=torch.long), [4]))


def expect(v, eps):
    return v / math.sqrt(v * v + eps)


y = F.rms_norm(torch.full((1, 2), 1e-9, dtype=d64), [2])
assert abs(y[0, 0].item() - expect(1e-9, torch.finfo(d64).eps)) < 1e-12
for dtype in [f32, f16]:
    v = torch.full((1, 2), 1e-4, dtype=dtype)
    y = F.rms_norm(v, [2])
    assert abs(y[0, 0].item() - expect(v[0, 0].item(), torch.finfo(f32).eps)) < 1e-3
assert F.rms_norm(torch.zeros(1, 2), [2]).tolist() == [[0.0, 0.0]]
assert F.rms_norm(x.float(), [4], torch.ones(4, dtype=d64)).dtype == f32
xm = torch.zeros(3, 4, dtype=bf16, device="meta", requires_grad=True)
out = F.rms_norm(xm, [4], eps=1e-6)
assert out.dtype == bf16 and out.shape == (3, 4)
assert torch.autograd.grad(out.sum(), xm)[0].shape == (3, 4)

# embedding_layer / norm_layers: the defaults.
torch.manual_seed(0)
table = nn.Embedding(1000, 16)
assert table.weight.shape == (1000, 16) and table.padding_idx is None
assert abs(table.weight.mean().item()) < 0.05 and abs(table.weight.var().item() - 1) < 0.05
table = nn.Embedding(6, 3, padding_idx=-2)
assert table.padding_idx == 4 and table.weight[4].tolist() == [0.0] * 3
raises(AssertionError, "Padding_idx must be within num_embeddings",
       lambda: nn.Embedding(6, 3, padding_idx=6))
assert list(table.state_dict()) == ["weight"]
table(torch.tensor([[4, 0]])).sum().backward()
assert table.weight.grad[4].tolist() == [0.0] * 3 and table.weight.grad[0].tolist() == [1.0] * 3
model = nn.Sequential(nn.Embedding(5, 4), nn.LayerNorm(4))
assert list(model.state_dict()) == ["0.weight", "1.weight", "1.bias"]
norm = nn.LayerNorm([2, 3])
assert norm.eps == 1e-5 and norm.weight.tolist() == [[1.0] * 3] * 2
assert norm.bias.tolist() == [[0.0] * 3] * 2 and list(norm.state_dict()) == ["weight", "bias"]
plain = nn.LayerNorm(4, elementwise_affine=False)
assert plain.weight is None and plain.bias is None and not plain.state_dict()
assert nn.LayerNorm(4, elementwise_affine=False, bias=True).bias is None
assert nn.LayerNorm(4, bias=False).bias is None
raises(RuntimeError, None, lambda: nn.LayerNorm([])(torch.zeros(2, 3)))
rms = nn.RMSNorm(3)
assert rms.eps is None and list(rms.state_dict()) == ["weight"] and rms.weight.tolist() == [1.0] * 3
assert nn.RMSNorm(3, elementwise_affine=False).weight is None

# dropout_matches_pytorch: the noise is 1 / (1 - p) in the dtype (asserted by the generator),
# the product rounds once, and p = 1 multiplies by a zero.
x = torch.tensor([1.5, -2.0, INF, -INF, NAN, 0.0, -0.0], requires_grad=True)
out = F.dropout(x, 1.0)
assert chain(out) == [(0, "MulBackward0"), (1, acc)]
assert out.grad_fn._saved_other.shape == () and out.grad_fn._saved_other.item() == 0
for dtype in [f16, bf16]:
    xh = (torch.randn(400) * 3).to(dtype)
    noise = torch.ones(1, dtype=dtype).div_(0.7)
    o = F.dropout(xh, 0.3)
    kept = o != 0
    assert torch.equal(o[kept], (xh.float() * noise.float()).to(dtype)[kept])

# dropout_returns_the_input_without_drawing
x = torch.randn(3, 4)


def draws(f):
    torch.manual_seed(7)
    reference_draw = torch.rand(4)
    torch.manual_seed(7)
    out = f()
    return out, not torch.equal(torch.rand(4), reference_draw)


for input_, p, training in [(x, 0.5, False), (x, 0.0, True), (torch.zeros(0, 4), 0.5, True),
                            (x, 1.0, False)]:
    out, drew = draws(lambda: F.dropout(input_, p, training))
    assert out is input_ and not drew
out, drew = draws(lambda: F.dropout(x, 1.0))
assert out is not x and not drew and out.tolist() == [[0.0] * 4] * 3
out, drew = draws(lambda: F.dropout(x, 0.5))
assert drew
assert torch.equal(draws(lambda: F.dropout(x, 0.5))[0], out)

# dropout_checks_the_probability_first
x, empty = torch.ones(3), torch.zeros(0)
for p, exc in [(1.5, ValueError), (-0.1, ValueError), (NAN, RuntimeError)]:
    for input_, training in [(x, True), (x, False), (empty, True)]:
        raises(exc, f"dropout probability has to be between 0 and 1, but got {p}",
               lambda: F.dropout(input_, p, training))
raises(ValueError, "dropout probability has to be between 0 and 1, but got 1.5",
       lambda: nn.Dropout(1.5))
layer = nn.Dropout(NAN)
layer.eval()
raises(RuntimeError, "dropout probability has to be between 0 and 1, but got nan", lambda: layer(x))
ints = torch.arange(6)
raises(RuntimeError, "result type Float can't be cast to the desired output type Long",
       lambda: F.dropout(ints, 0.5))
assert F.dropout(ints, 0.0) is ints
zeros = F.dropout(ints, 1.0)
assert zeros.dtype == torch.long and zeros.tolist() == [0] * 6
flags = torch.ones(2, dtype=torch.bool)
raises(RuntimeError, "result type Float can't be cast to the desired output type Bool",
       lambda: F.dropout(flags, 0.5))
assert F.dropout(flags, 1.0).tolist() == [False, False]

# dropout_gradients
torch.manual_seed(3)
x = torch.randn(4, 5, dtype=d64, requires_grad=True)
y = x * 1
out = F.dropout(y, 0.4)
assert type(out.grad_fn).__name__ == "MulBackward0" and out.grad_fn._saved_self is None
with torch.no_grad():
    y.add_(1.0)
g = torch.ones(4, 5, dtype=d64, requires_grad=True)
(grad,) = torch.autograd.grad(out, x, g, create_graph=True)
expected = (out.detach() != 0).to(d64).div(1 - 0.4)
assert torch.equal(grad, expected)
dx, dg = torch.autograd.grad(grad.sum(), [x, g], allow_unused=True)
assert dx is None and torch.equal(dg, expected)
m = torch.zeros(2, 3, dtype=f16, device="meta", requires_grad=True)
out = F.dropout(m, 0.5)
assert out.device.type == "meta" and out.dtype == f16 and out.shape == (2, 3)
assert torch.autograd.grad(out.sum(), m)[0].shape == (2, 3)

# dropout_layer
layer = nn.Dropout()
assert layer.p == 0.5 and layer.training and not layer.state_dict()
assert not list(layer.parameters())
x = torch.ones(200)
out = layer(x)
assert ((out == 0) | (out == 2)).all()
layer.eval()
assert layer(x) is x

print("check_m2_embed_norm: all reference assertions passed")
