"""Check the reference behavior that tests/autograd_retain.rs and the autograd suffixes of
`Debug` (src/tensor/fmt.rs) encode; does not write fixtures.

Each block mirrors a test there, in the same order. Where homura deliberately differs (stale views,
a tensor detached in place that joins a graph again, the dtype and size spelling of `Debug`), the
PyTorch behavior is asserted and the difference is noted.
"""

import gc
import weakref

import torch

import reference

reference.check()


def message(f):
    try:
        f()
    except Exception as e:  # noqa: BLE001 - any error type carries the message we compare
        return str(e).split(" Hint: ")[0]
    raise AssertionError("expected an error")


def leaf(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype, requires_grad=True)


NO_GRAD = "can't retain_grad on Tensor that has requires_grad=False"

# retain_grad_is_a_no_op_on_leaves_and_rejects_tensors_without_grad
x = leaf(1.0, 2.0)
x.retain_grad()
assert not x.retains_grad
assert message(lambda: torch.ones(2).retain_grad()) == NO_GRAD
with torch.no_grad():
    z = x * 2
assert message(z.retain_grad) == NO_GRAD
with torch.no_grad():
    v = x.view(2)
assert v.requires_grad and v.is_leaf
v.retain_grad()
assert not v.retains_grad
y = x * 2
y.retain_grad()
y.retain_grad()
assert y.retains_grad and y.grad is None
(y * 3).sum().backward()
assert y.grad.tolist() == [3.0, 3.0]

# the_total_gradient_is_stored_and_accumulated_out_of_place
x = leaf(1.0, 2.0)
y = x + 1
y.retain_grad()
out = (y * 3 + (y * 1) * 4).sum()
out.backward(retain_graph=True)
first, first_x = y.grad, x.grad
assert first.tolist() == [7.0, 7.0] and first.grad_fn is None and not first.requires_grad
out.backward(retain_graph=True)
assert y.grad.tolist() == [14.0, 14.0] and first.tolist() == [7.0, 7.0] and y.grad is not first
# Leaves accumulate in place: the earlier handle follows.
assert x.grad is first_x and first_x.tolist() == [14.0, 14.0]
y.grad = None
out.backward(retain_graph=True)
assert y.grad.tolist() == [7.0, 7.0]
preset = torch.full((2,), 5.0)
y.grad = preset
out.backward()
assert y.grad.tolist() == [12.0, 12.0] and preset.tolist() == [5.0, 5.0]

# autograd_grad_stores_retained_gradients_on_its_paths
x = leaf(1.0, 2.0)
y = x * 2
y.retain_grad()
w = x * 3
w.retain_grad()
out = (y * y + w).sum()
torch.autograd.grad(out, [x], retain_graph=True)
assert y.grad.tolist() == [4.0, 8.0] and w.grad.tolist() == [1.0, 1.0] and x.grad is None
# A requested input that retains its gradient stores it too.
torch.autograd.grad(out, [y], retain_graph=True)
assert y.grad.tolist() == [8.0, 16.0] and w.grad.tolist() == [1.0, 1.0]
# Off the path to the requested input: nothing stored.
torch.autograd.grad(out, [w])
assert y.grad.tolist() == [8.0, 16.0] and w.grad.tolist() == [2.0, 2.0]

# gradients_stored_with_create_graph_can_be_differentiated
x = leaf(1.0, 2.0)
y = x * x
y.retain_grad()
torch.autograd.grad((y * y).sum(), [x], create_graph=True)
assert type(y.grad.grad_fn).__name__ == "CloneBackward0"
assert y.grad.tolist() == [2.0, 8.0]
first = y.grad
(d,) = torch.autograd.grad(y.grad.sum(), [x], retain_graph=True)
assert d.tolist() == [4.0, 8.0]
# That pass reached y's node through the y saved for y.grad, and stored there too.
assert y.grad.tolist() == [4.0, 10.0] and first.tolist() == [2.0, 8.0]
torch.autograd.grad((y * y).sum(), [x], create_graph=True)
assert type(y.grad.grad_fn).__name__ == "AddBackward0" and y.grad.tolist() == [6.0, 18.0]
# A second pass through a saved output stores again (round 33).
x = leaf(0.5, dtype=torch.float64)
y = x.exp()
y.retain_grad()
(g,) = torch.autograd.grad(y, x, create_graph=True)
first = y.grad
assert first.tolist() == [1.0]
torch.autograd.grad(g, x)
assert y.grad.tolist() == [2.0] and first.tolist() == [1.0]

# stored_gradients_are_contiguous_and_keep_the_dtype
x = torch.zeros(3, 2, requires_grad=True)
y = x * 1
y.retain_grad()
g = torch.arange(6.0).view(2, 3).t()
assert not g.is_contiguous()
y.backward(g)
assert y.grad.is_contiguous() and y.grad.stride() == (2, 1) and torch.equal(y.grad, g)
x = leaf(1.0, 2.0, dtype=torch.float16)
y = x * 2
y.retain_grad()
y.sum().backward()
assert y.grad.dtype == torch.float16

# views_and_multiple_outputs_store_their_own_gradient
x = leaf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
b = x * 1
v = b.view(2, 3)
v.retain_grad()
(v * 5).sum().backward()
assert v.grad.tolist() == [[5.0] * 3] * 2 and b.grad is None
# Outputs of one node are told apart by output_nr (homura checks this on a synthetic node in
# src/autograd/retain_tests.rs: it has no single node with several outputs).
x = leaf(1.0, 2.0)
a, c = (x * 1).unbind()
a.retain_grad()
c.retain_grad()
assert a.grad_fn is c.grad_fn
(a * 2 + c * 3).backward(retain_graph=True)
assert a.grad.item() == 2.0 and c.grad.item() == 3.0
a.grad = c.grad = None
(a * 2).backward()
assert a.grad.item() == 2.0 and c.grad is None

# in_place_operations_move_the_registration
x = leaf(1.0, 2.0)
y = x + 1
y.retain_grad()
old = (y * 3).sum()
y.grad = torch.ones(2)
y.add_(1)
assert y.retains_grad and y.grad.tolist() == [1.0, 1.0]
old.backward()
assert y.grad.tolist() == [1.0, 1.0]
(y * 7).sum().backward()
assert y.grad.tolist() == [8.0, 8.0]
# detach_ keeps the registration on the old history, and the flag.
x = leaf(1.0, 2.0)
y = x + 1
y.retain_grad()
old = (y * 3).sum()
y.detach_()
assert y.retains_grad and y.is_leaf and not y.requires_grad
old.backward()
assert y.grad.tolist() == [3.0, 3.0]
# Joining a graph again in place fails an internal assertion in PyTorch, after the history was
# replaced, and the new history never stores. homura registers the tensor with the new node.
x = leaf(1.0, 2.0)
y = x + 1
y.retain_grad()
y.detach_()
assert "INTERNAL ASSERT FAILED" in message(lambda: y.add_(x))
assert type(y.grad_fn).__name__ == "AddBackward0" and y.retains_grad
(y * 2).sum().backward()
assert y.grad is None

# stale_views_can_not_start_retaining
# PyTorch replays the view on the base's new history; homura rejects using the stale history.
x = leaf(1.0, 2.0)
b = x * 1
v = b.view(2)
b.add_(1)
v.retain_grad()
assert type(v.grad_fn).__name__ == "AsStridedBackward0"
(v * 4).sum().backward()
assert v.grad.tolist() == [4.0, 4.0]

# concurrent_backward_passes_lose_no_retained_gradient: not run here. PyTorch serializes the
# accumulation into a leaf but not the retains-grad hook (`grad = grad + new` on the tensor), so two
# threads running backward through the same retained tensor lose updates or crash: with
# x = zeros(1000) leaf, y = x + 1 retained, (y * 1).sum() and (y * 2).sum() each backward-ed 100
# times on its own thread (retain_graph=True), x.grad is always 300 while y.grad was 298 or 299 in
# about half of the runs, and the interpreter segfaulted in some. homura serializes both, so the
# test expects 300 for both.

# Lifetimes (src/autograd/retain_tests.rs).
# registration_does_not_keep_the_tensor_alive
x = leaf(1.0, 2.0)
y = x * 2
y.retain_grad()
z = (y * 3).sum()
ref = weakref.ref(y)
del y
gc.collect()
assert ref() is None
z.backward()
# create_graph without a path back to the tensor: freed with its handle.
x = leaf(1.0, 2.0)
y = x * x
y.retain_grad()
(gx,) = torch.autograd.grad((y * y).sum(), [x], create_graph=True)
ref = weakref.ref(y)
del y, gx
gc.collect()
assert ref() is None
# Round 34: a leaf view of the tensor, used in its gradient's graph, holds it through its base.
x = leaf(1.0)
y = x * 2
y.retain_grad()
with torch.no_grad():
    v = y.view(1)
v.requires_grad_(True)
u = v * 3
torch.autograd.grad(y, x, grad_outputs=u, create_graph=True)
ref = weakref.ref(y)
del y, v, u
gc.collect()
assert ref() is not None
ref().grad = None
gc.collect()
assert ref() is None


# Debug suffixes (src/tensor/fmt.rs). homura spells dtypes and sizes its own way, so only cases
# whose other suffixes are spelled alike are compared whole.
def r(t):
    return repr(t)


x = leaf(1.0, 2.0)
assert r(x) == "tensor([1., 2.], requires_grad=True)"
assert r(x * 2) == "tensor([2., 4.], grad_fn=<MulBackward0>)"
assert r(x.sum()) == "tensor(3., grad_fn=<SumBackward0>)"
b = torch.zeros(2)
b.copy_(x)
assert r(b) == "tensor([1., 2.], grad_fn=<CopyBackwards>)"
with torch.no_grad():
    v = x.view(2)
assert r(v) == "tensor([1., 2.], requires_grad=True)"
assert r(x.detach()) == "tensor([1., 2.])"
assert r(torch.zeros(0, requires_grad=True)) == "tensor([], requires_grad=True)"
assert r(torch.zeros(0, 3, requires_grad=True) * 1) == "tensor([], size=(0, 3), grad_fn=<MulBackward0>)"
m = torch.zeros(2, 3, device="meta", dtype=torch.float16, requires_grad=True)
assert r(m * 1) == (
    "tensor(..., device='meta', size=(2, 3), dtype=torch.float16,\n       grad_fn=<MulBackward0>)"
)
long = torch.arange(2000.0, requires_grad=True) * 1
assert r(long) == "tensor([0.0000e+00, 1.0000e+00, 2.0000e+00,  ..., 1.9970e+03, 1.9980e+03,\n        1.9990e+03], grad_fn=<MulBackward0>)"
# Wrapping: the last line counts two more than its length, on one line and on several.
row = torch.arange(5.0, requires_grad=True) * 1.5
assert r(row) == "tensor([0.0000, 1.5000, 3.0000, 4.5000, 6.0000], grad_fn=<MulBackward0>)"
row = torch.arange(6.0, requires_grad=True) * 1.5
assert r(row) == (
    "tensor([0.0000, 1.5000, 3.0000, 4.5000, 6.0000, 7.5000],\n       grad_fn=<MulBackward0>)"
)
grid = torch.arange(10.0, requires_grad=True).view(2, 5) * 1.25
assert r(grid) == (
    "tensor([[ 0.0000,  1.2500,  2.5000,  3.7500,  5.0000],\n"
    "        [ 6.2500,  7.5000,  8.7500, 10.0000, 11.2500]], grad_fn=<MulBackward0>)"
)
grid = torch.arange(12.0, requires_grad=True).view(2, 6) * 1.25
assert r(grid) == (
    "tensor([[ 0.0000,  1.2500,  2.5000,  3.7500,  5.0000,  6.2500],\n"
    "        [ 7.5000,  8.7500, 10.0000, 11.2500, 12.5000, 13.7500]],\n"
    "       grad_fn=<MulBackward0>)"
)
# A stale view: PyTorch replays it; homura shows grad_fn=<Invalid>, PyTorch's text for a grad_fn
# that can't be computed (no-grad views modified in place).
x = leaf(1.0, 2.0)
b = x * 1
with torch.no_grad():
    nv = b.view(2)
with torch.no_grad():
    nv.add_(1)
assert r(b) == "tensor([2., 3.], grad_fn=<MulBackward0>)"
b2 = x * 1
s = b2.view(2)
b2.add_(1)
assert r(b2) == "tensor([2., 3.], grad_fn=<AddBackward0>)"
assert r(s) == "tensor([2., 3.], grad_fn=<AsStridedBackward0>)"

print("check_m1_retain: all reference assertions passed")
