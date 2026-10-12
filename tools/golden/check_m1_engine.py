"""Check the reference behavior that tests/autograd_engine.rs encodes; does not write fixtures.

Each block mirrors a test there. Where homura deliberately differs (messages adapted to Rust
names, in-place operations that M1 step 3 doesn't record yet), the PyTorch behavior is asserted
and the difference is noted.
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


def leaf(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype, requires_grad=True)


# ---------------------------------------------------------------- first order
x = leaf(1.0, 2.0, 3.0)
(x * x).sum().backward()
assert x.grad.tolist() == [2.0, 4.0, 6.0]
x = leaf(0.0, 1.0, -1.0)
a = x.exp()
((a * 2.0) + (a * 3.0) - x).sum().backward()
assert torch.allclose(x.grad, 5 * x.detach().exp() - 1)

x = torch.tensor([[1.0, 2, 3], [4, 5, 6]], requires_grad=True)
w = leaf(0.5, -1.0, 2.0)
g = torch.tensor([[1.0, 0, 2], [-1, 3, 0.5]])
(x * w + w).backward(g)
assert x.grad.flatten().tolist() == [0.5, 0, 4, -0.5, -3, 1]
assert w.grad.tolist() == [-3, 18, 11.5]

x = leaf(1.0, 2.0)
(2.0 * x - 1.0).sum().backward()
assert x.grad.tolist() == [2, 2]
x = leaf(1.0, 2.0)
(1.0 - x).sum().backward()
assert x.grad.tolist() == [-1, -1]
x = leaf(1.0, 2.0)
(1.0 + x * 3.0).sum().backward()
assert x.grad.tolist() == [3, 3]
x = leaf(1.0, 2.0)
w = leaf(3.0, 4.0, dtype=torch.float64)
y = (x * w).sum()
assert y.dtype == torch.float64
y.backward()
assert x.grad.dtype == torch.float32 and x.grad.tolist() == [3, 4]
assert w.grad.tolist() == [1, 2]
h = torch.tensor([1.0, 2.0]).half().requires_grad_()
(h * 3.0).sum().backward()
assert h.grad.dtype == torch.float16 and h.grad.tolist() == [3, 3]
x = leaf(1.0, 2.0)
(x * torch.tensor([3, 4])).sum().backward()
assert x.grad.tolist() == [3, 4]

x = leaf(1.0, 2.0)
(x * 5.0).sum().backward(torch.tensor(2.0, dtype=torch.float64))
assert x.grad.dtype == torch.float32 and x.grad.tolist() == [10, 10]

x = leaf(1.0, 2.0)
(x * x).sum().view(1).backward()
assert x.grad.tolist() == [2, 4]
x = leaf(3.0)
(x * 2.0).view(1, 1).backward()
assert x.grad.tolist() == [2]
assert (
    message(lambda: (x * 2.0).expand(2).backward())
    == "grad can be implicitly created only for scalar outputs"
)

x = leaf(1.0, 2.0)
(x * 3.0).sum().backward()
first = x.grad
assert first.is_contiguous()
(x * x).sum().backward()
assert x.grad is first and x.grad.tolist() == [5, 7]

x = leaf(1.0, 2.0)
with torch.no_grad():
    y = x * x
    assert not y.requires_grad and y.is_leaf and y.grad_fn is None
    with torch.enable_grad():
        assert (x * x).requires_grad

x = leaf(1.0, 2.0)
w = leaf(3.0, 4.0)
(g,) = torch.autograd.grad((x * w).sum(), [x])
assert g.tolist() == [3, 4] and x.grad is None and w.grad is None
a = x * 2.0
ga, gx = torch.autograd.grad((a * a).sum(), [a, x])
assert ga.tolist() == [4, 8] and gx.tolist() == [8, 16]
(g,) = torch.autograd.grad([(x * 2.0).sum(), (x * 3.0).sum()], [x])
assert g.tolist() == [5, 5]
y = (x * w).sum()
assert torch.autograd.grad(y, [y])[0].item() == 1
s = leaf(2.0)
assert torch.autograd.grad(s, [s])[0].tolist() == [1] and s.grad is None

x = leaf(1.0, 2.0)
a = x * 2.0
b = a * 3.0
torch.autograd.backward([a.sum(), b.sum()], [None, torch.tensor(2.0)])
assert x.grad.tolist() == [14, 14]

x = torch.zeros(2, 3, device="meta", requires_grad=True)
w = torch.zeros(3, dtype=torch.float64, device="meta", requires_grad=True)
(x * w + 1.0).exp().sum().backward()
assert (x.grad.shape, x.grad.dtype, x.grad.device.type) == ((2, 3), torch.float32, "meta")
assert (w.grad.shape, w.grad.dtype, w.grad.device.type) == ((3,), torch.float64, "meta")

# ---------------------------------------------------------------- second order
cg = {"create_graph": True}
x = leaf(1.0, 2.0, 3.0)
(g,) = torch.autograd.grad((x * x).sum(), [x], **cg)
assert g.requires_grad and g.tolist() == [2, 4, 6]
assert torch.autograd.grad(g.sum(), [x])[0].tolist() == [2, 2, 2]

x, w = leaf(1.0, 2.0), leaf(3.0, -1.0)
(gx,) = torch.autograd.grad((x * x * w).sum(), [x], **cg)
assert gx.tolist() == [6, -4]
hx, hw = torch.autograd.grad(gx.sum(), [x, w])
assert hx.tolist() == [6, -2] and hw.tolist() == [2, 4]

x = leaf(0.5, 1.0)
(g,) = torch.autograd.grad(x.exp().sum(), [x], **cg)
assert type(g.grad_fn).__name__ == "MulBackward0"
assert torch.allclose(torch.autograd.grad(g.sum(), [x])[0], x.detach().exp())

x, w = leaf(1.0, 2.0), leaf(3.0, 4.0)
a = x * w
(g,) = torch.autograd.grad((a * a).sum(), [x], **cg)
del a
assert g.tolist() == [18, 64]
hx, hw = torch.autograd.grad(g.sum(), [x, w])
assert hx.tolist() == [18, 32] and hw.tolist() == [12, 32]

x = leaf(2.0)
accumulator = (x * 1.0).grad_fn.next_functions[0][0]
(g,) = torch.autograd.grad((x * x).sum(), [x], **cg)
found, stack = 0, [g.grad_fn]
while stack:
    f = stack.pop()
    if type(f).__name__ == "AccumulateGrad":
        assert f is accumulator
        found += 1
    stack.extend(n for n, _ in f.next_functions if n is not None)
assert found > 0


def double_backward(x, f, linear):
    y = f(x)
    u = (torch.arange(y.numel(), dtype=torch.float32) - 1.5).reshape(y.shape).to(y.dtype)
    u.requires_grad_()
    (g,) = torch.autograd.grad(y, [x], grad_outputs=u, create_graph=True)
    assert g.requires_grad
    r = (torch.arange(g.numel(), dtype=torch.float32) + 0.5).reshape(g.shape).to(g.dtype)
    (gu,) = torch.autograd.grad((g * r).sum(), [u])
    with torch.no_grad():
        expected = linear(r)
    assert gu.shape == expected.shape and torch.allclose(gu.float(), expected.float())
    return g, u


x = torch.tensor([[1.0, -2, 3], [0.5, 4, -1]], requires_grad=True)
w = torch.tensor([2.0, -1, 0.5])
row = torch.tensor([[1.0, 2, 3]], requires_grad=True)
g, u = double_backward(x, lambda x: x + w, lambda r: r)
assert g is u, "the gradient of add is the upstream gradient itself"
double_backward(x, lambda x: 2.0 + x, lambda r: r)
double_backward(x, lambda x: x - w, lambda r: r)
double_backward(x, lambda x: 1.0 - x, lambda r: -r)
wg = w.clone().requires_grad_()
double_backward(wg, lambda w: x - w, lambda r: -r.expand(2, 3))
double_backward(x, lambda x: x * w, lambda r: r * w)
double_backward(x, lambda x: x * 3.0, lambda r: r * 3.0)
double_backward(x, lambda x: -x, lambda r: -r)
double_backward(x, lambda x: x.sum(), lambda r: r.sum())
double_backward(row, lambda r: r.expand(4, 3), lambda r: r.expand(4, 3))
double_backward(x, lambda x: x.sum_to_size(3), lambda r: r.sum_to_size(3))
double_backward(x, lambda x: x.clone(), lambda r: r)
double_backward(x, lambda x: x.view(3, 2), lambda r: r.view(3, 2))
double_backward(x, lambda x: x.reshape(-1), lambda r: r.reshape(-1))
double_backward(
    row, lambda r: r.expand(2, 3).reshape(6), lambda r: r.expand(2, 3).reshape(6)
)
double_backward(row, lambda r: r.repeat(2, 2), lambda r: r.repeat(2, 2))
double_backward(x, lambda x: x.to(torch.float64), lambda r: r.to(torch.float64))
double_backward(x, lambda x: x.flatten(0, 1), lambda r: r.flatten(0, 1))
x = leaf(0.5, -1.0)
ex = x.detach().exp()
double_backward(x, lambda x: x.exp(), lambda r: r * ex)
double_backward(x, lambda x: x * x, lambda r: r * x.detach() * 2.0)

# ---------------------------------------------------------------- release
x = leaf(1.0, 2.0)
y = (x * x).sum()
y.backward()
# homura names its own API: "backward() or autograd::grad(). Set retain_graph".
assert message(y.backward).startswith(
    "Trying to backward through the graph a second time (or directly access saved tensors "
    "after they have already been freed)."
)
x = leaf(1.0, 2.0)
y = (x * x).sum()
y.backward(retain_graph=True)
y.backward()
assert x.grad.tolist() == [4, 8]
x = leaf(1.0, 2.0)
y = (x * x).sum()
torch.autograd.grad(y, [x], create_graph=True)
assert torch.autograd.grad(y, [x])[0].tolist() == [2, 4]
x = leaf(1.0, 2.0)
y = (x + 1.0).sum()
y.backward()
y.backward()
assert x.grad.tolist() == [2, 2]

# ---------------------------------------------------------------- tensor API
n = torch.tensor([1, 2])
assert (
    message(lambda: n.requires_grad_(True))
    == "only Tensors of floating point dtype can require gradients"
)
assert not n.requires_grad_(False).requires_grad
x = leaf(1.0)
y = x * 2.0
before = y.grad_fn
assert y.requires_grad_(True).grad_fn is before
assert message(lambda: y.requires_grad_(False)) == (
    "you can only change requires_grad flags of leaf variables. If you want to use a computed "
    "variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach()."
)

x = leaf(1.0, 2.0, 3.0)


def assign(g):
    x.grad = g


# Checked in the order dtype, device, size. homura has no grad_dtype attribute, so its dtype
# message names the tensor's dtype instead.
assert message(lambda: assign(torch.zeros(2, dtype=torch.float64, device="meta"))).startswith(
    "attempting to assign a gradient with dtype 'double' to a tensor with grad_dtype 'Float'."
)
assert message(lambda: assign(torch.zeros(2, device="meta"))) == (
    "attempting to assign a gradient with device type 'meta' to a tensor with device type "
    "'cpu'. Please ensure that the gradient and the tensor are on the same device"
)
assert message(lambda: assign(torch.zeros(2))) == (
    "attempting to assign a gradient of size '[2]' to a tensor of size '[3]'. Please ensure "
    "that the gradient and the tensor are the same size"
)
g = torch.zeros(3, 2).select(1, 0)
x.grad = g
assert x.grad is g
(x * 2.0).grad = torch.ones(3)
torch.tensor([1]).grad = torch.tensor([2])

x = leaf(1.0, 2.0)
y = x * 2.0
d = y.detach()
assert not d.requires_grad and d.is_leaf and d.grad_fn is None and not d._is_view()
d.add_(1.0)
assert y.tolist() == [3, 5]

x = leaf(1.0, 2.0)
y = x * 2.0
assert y.detach_().is_leaf and not y.requires_grad
(x * 3.0).sum().backward()
x.detach_()
assert not x.requires_grad and x.grad.tolist() == [3, 3]
for v in [torch.tensor([1.0]).view(1), torch.tensor([1]).view(1)]:
    assert message(v.detach_).startswith("Can't detach views in-place. Use detach() instead.")

x, y = leaf(1.0, 2.0), leaf(3.0, 4.0)


def name(t):
    return type(t.grad_fn).__name__


def next_names(t):
    return [type(n).__name__ if n is not None else None for n, _ in t.grad_fn.next_functions]


acc = "AccumulateGrad"
assert next_names(x + 1.0) == [acc, None] and next_names(1.0 + x) == [acc, None]
assert next_names(x * 2.0) == [acc, None] and next_names(2.0 * x) == [acc, None]
assert next_names(x - y) == [acc, acc] and next_names(1.0 - x) == [acc]
for t, expected in [
    (x + 1.0, "AddBackward0"),
    (x - 1.0, "SubBackward0"),
    (1.0 - x, "RsubBackward1"),
    (x * y, "MulBackward0"),
    (-x, "NegBackward0"),
    (x.exp(), "ExpBackward0"),
    (x.sum(), "SumBackward0"),
    (x.expand(3, 2), "ExpandBackward0"),
    (x.view(2, 1), "ViewBackward0"),
    (x.reshape(1, 2), "ViewBackward0"),
    (x.clone(), "CloneBackward0"),
    (x.to(torch.float64), "ToCopyBackward0"),
    (x.to("meta"), "ToCopyBackward0"),
]:
    assert name(t) == expected, (expected, name(t))
copied = x.expand(2, 2).reshape(4)
assert name(copied) == "UnsafeViewBackward0"
assert next_names(copied) == ["CloneBackward0"]
m = x.expand(3, 2)
# homura records sum_to_size as one SumToSizeBackward0 node.
assert name(m.sum_to_size(1, 2)) == "SumBackward1"
assert m.sum_to_size(3, 2) is m

x = torch.tensor([[1.0, 2, 3]], requires_grad=True)
x.expand(2, 3).reshape(6).backward(torch.tensor([1.0, 2, 3, 4, 5, 6]))
assert x.grad.tolist() == [[5, 7, 9]]
x = leaf(1.0, 2.0)
x.repeat(3).backward(torch.tensor([1.0, 2, 3, 4, 5, 6]))
assert x.grad.tolist() == [9, 12]
x = torch.tensor([[1.0, 2]], requires_grad=True)
x.repeat(2, 2).sum().backward()
assert x.grad.tolist() == [[4, 4]]

# ---------------------------------------------------------------- views
b = leaf(2.0)
v = b.view(1)
b.requires_grad_(False)
assert v.requires_grad and not v.is_leaf and v.grad_fn is not None
b = torch.tensor([2.0])
v = b.view(1).requires_grad_(True)
w = v.view(1)
assert next_names(w) == [acc]
(w * 2).sum().backward()
assert b.grad is None and v.grad.tolist() == [2]

z = leaf(1.0, 2.0, 3.0)
with torch.no_grad():
    v = z.view(3)
assert v.requires_grad and v.is_leaf and v.grad_fn is None
y = v * 2.0
assert y.requires_grad and next_names(y) == [None, None]
y.sum().backward()
assert z.grad is None and v.grad is None
assert message(lambda: v.backward(torch.ones(3))) == (
    "element 0 of tensors does not require grad and does not have a grad_fn"
)
# homura: "Set allow_unused if this is the desired behavior."
assert message(lambda: torch.autograd.grad((v * 2.0).sum(), [v])).startswith(
    "The differentiated Tensor at index 0 appears to not have been used in the graph."
)
b = torch.ones(2)
early = b.view(2)
b.requires_grad_(True)
assert early.requires_grad and early.is_leaf and early.grad_fn is None

for detach in [False, True]:
    b = leaf(1.0, 2.0)
    with torch.no_grad():
        quiet = b.view(2)
    recorded = b.view(2)
    if detach:
        b.detach_()
    else:
        b.requires_grad_(False)
    assert not quiet.requires_grad
    assert recorded.requires_grad and not recorded.is_leaf

for x in [torch.zeros(4, 3), torch.zeros(0, 3)]:
    with torch.no_grad():
        pieces = x.chunk(3, 0)
    assert len(pieces) >= 2
    pieces[0].requires_grad_(True)
    assert not any(p.requires_grad for p in pieces[1:])
    pieces[1].grad = torch.ones(pieces[1].shape)
    assert pieces[0].grad is None and all(p.grad is None for p in pieces[2:])

x = leaf(1.0, -2.0)
for out in [
    x > 0,
    x == x,
    torch.logical_and(x, x),
    torch.logical_not(x),
    x.argmax(),
    x.all(),
    x.to(torch.int64),
    x.detach(),
    torch.zeros_like(x),
    torch.rand_like(x),
]:
    assert not out.requires_grad and out.grad_fn is None

# ---------------------------------------------------------------- in-place operations
leaf_message = "a leaf Variable that requires grad is being used in an in-place operation."
x = leaf(0.5, 0.25)
ones = torch.ones(2)
ops = [
    lambda x: x.add_(1.0),
    lambda x: x.sub_(ones),
    lambda x: x.mul_(2.0),
    lambda x: x.div_(2.0),
    lambda x: x.pow_(2.0),
    lambda x: x.clamp_(0.0, 1.0),
    lambda x: x.masked_fill_(torch.tensor([True, False]), 0.0),
    lambda x: x.copy_(torch.tensor([3.0, 4.0])),
    lambda x: x.fill_(1.0),
    lambda x: x.zero_(),
    lambda x: x.uniform_(0.0, 1.0),
    lambda x: x.normal_(0.0, 1.0),
    lambda x: x.lerp_(ones, 0.5),
    lambda x: x.addcmul_(ones, ones, value=1.0),
    lambda x: x.addcdiv_(ones, ones, value=1.0),
    lambda x: x.__iadd__(1.0),
]
for op in ops:
    version = x._version
    assert message(lambda: op(x)) == leaf_message
    assert x.tolist() == [0.5, 0.25] and x._version == version
# Recorded by PyTorch; homura rejects these until in-place operations record (M1 step 7).
y = x * 2.0
y.add_(1.0)
assert name(y) == "AddBackward0"
o = torch.ones(2)
o.add_(x)
assert name(o) == "AddBackward0"
# Views: PyTorch rejects these two with its own messages; homura with its view message.
assert message(lambda: x.view(2).add_(1.0)) == (
    "a view of a leaf Variable that requires grad is being used in an in-place operation."
)
with torch.no_grad():
    quiet = x.view(2)
assert message(lambda: quiet.mul_(2.0)).startswith(
    "A view was created in no_grad mode and is being modified inplace with grad mode enabled."
)
plain = torch.zeros(2)
plain.view(2).add_(1.0)
assert plain.tolist() == [1, 1]

x = leaf(1.0, 2.0)
y = (x * x).sum()
with torch.no_grad():
    x.sub_(0.5)
    x.view(2).mul_(2.0)
assert x.tolist() == [1, 3] and x.requires_grad and x.is_leaf
# homura writes the tensor as "[f32 [2]]".
assert message(y.backward).startswith(
    "one of the variables needed for gradient computation has been modified by an inplace "
    "operation: [torch.FloatTensor [2]] is at version 2; expected version 0 instead."
)
x = leaf(1.0, 2.0)
y = (x * x).sum()
x.detach().add_(1.0)
assert "is at version 1; expected version 0 instead." in message(y.backward)

x = leaf(1.5, 2.5)
n = torch.zeros(2, dtype=torch.int64)
n.copy_(x)
assert n.tolist() == [1, 2] and not n.requires_grad
# homura names dtypes "f32" and "i64".
assert (
    message(lambda: n.add_(x)) == "result type Float can't be cast to the desired output type Long"
)

# ---------------------------------------------------------------- saved tensors
x = leaf(2.0)
y = (x * x).sum()
x.requires_grad_(False)
y.backward()
assert x.grad is None

x = torch.tensor([2.0])
z = leaf(3.0)
y = (x * z).sum()
x.requires_grad_(True)
(g,) = torch.autograd.grad(y, [z], **cg)
assert g.requires_grad and torch.autograd.grad(g.sum(), [x])[0].tolist() == [1]

detached = (
    "Trying to use a saved tensor that has been detached in-place, i.e. with .detach_().This "
    "is not supported, please use out-of-place `.detach()` instead"
)
x = leaf(2.0)
a = x * 2.0
y = (a * a).sum()
a.detach_()
assert message(y.backward) == detached
x = leaf(2.0)
a = x * 2.0
y = (a * a).sum()
del a
y.backward()
assert x.grad.tolist() == [16]
for detach in [True, False]:
    x = leaf(2.0)
    a = x * x
    b = leaf(3.0)
    u = torch.tensor(1.0, requires_grad=True)
    (g,) = torch.autograd.grad((a * b).sum(), [b], grad_outputs=u, create_graph=True)
    if detach:
        a.detach_()
        assert message(lambda: torch.autograd.grad(g, [u])) == detached
    else:
        del a
        assert torch.autograd.grad(g, [u])[0].item() == 4

# ---------------------------------------------------------------- errors
x = leaf(1.0, 2.0)
plain = torch.tensor([1.0])
not_required = "element {} of tensors does not require grad and does not have a grad_fn"
assert message(lambda: plain.sum().backward()) == not_required.format(0)
assert message(lambda: torch.autograd.backward([x.sum(), plain])) == not_required.format(1)
assert (
    message(lambda: torch.autograd.backward([x.sum(), x.sum()], [torch.tensor(1.0)]))
    == "got 2 tensors and 1 gradients"
)
assert message(lambda: torch.autograd.backward([x.sum(), x.sum()], [torch.tensor([1.0])])) == (
    "Mismatch in shape: grad_output[0] has a shape of torch.Size([1]) and output[0] has a "
    "shape of torch.Size([])."
)
assert message(lambda: (x * 2.0).backward(torch.tensor([1.0, 2, 3]))) == (
    "Mismatch in shape: grad_output[0] has a shape of torch.Size([3]) and output[0] has a "
    "shape of torch.Size([2])."
)
assert (
    message(lambda: torch.autograd.grad(x.sum(), []))
    == "`inputs` argument to `grad()` cannot be empty."
)
assert (
    message(lambda: torch.autograd.grad(x.sum(), [plain]))
    == "One of the differentiated Tensors does not require grad"
)
w = leaf(3.0)
assert message(lambda: torch.autograd.grad(x.sum(), [x, w])) == (
    "The differentiated Tensor at index 1 appears to not have been used in the graph. Set "
    "allow_unused=True if this is the desired behavior."
)
gx, gw = torch.autograd.grad(x.sum(), [x, w], allow_unused=True)
assert gx is not None and gw is None
assert message(lambda: torch.autograd.grad([], [x])).startswith(
    "The differentiated Tensor at index 0 appears to not have been used"
)

x = leaf(1.0, 2.0)
m = x.to("meta")
assert m.requires_grad
assert message(m.sum().backward) == "Cannot copy out of meta tensor; no data!"

assert math.isclose(1.0, 1.0)
print("check_m1_engine: all reference checks passed")
