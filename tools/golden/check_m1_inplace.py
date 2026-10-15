"""Check the reference behavior that tests/autograd_inplace.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order. Where homura deliberately differs (masked_fill
recorded as where_, homura's messages for views it doesn't support yet, mixed scalar and tensor
clamp bounds), the PyTorch behavior is asserted and the difference is noted.
"""

import math

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


def chain(t):
    """The name of the node of t and of the nodes its edges lead to, without namespaces."""
    def name(f):
        return f.name().removeprefix("torch::autograd::") if f is not None else None

    return name(t.grad_fn), [name(f) for f, _ in t.grad_fn.next_functions]


def grads(out, *inputs):
    return [g.tolist() for g in torch.autograd.grad(out.sum(), inputs)]


def close(a, b):
    return all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6) for x, y in zip(a, b, strict=True))


def version_error(shape, output, current, expected):
    return (
        "one of the variables needed for gradient computation has been modified by an inplace "
        f"operation: [torch.FloatTensor {shape}]{output} is at version {current}; expected version "
        f"{expected} instead."
    )


ACC, MUL = "AccumulateGrad", "MulBackward0"
LEAF = "a leaf Variable that requires grad is being used in an in-place operation."
VIEW_OF_LEAF = "a view of a leaf Variable that requires grad is being used in an in-place operation."
NO_GRAD_VIEW = "A view was created in no_grad mode and is being modified inplace with grad mode enabled."

# ---------------------------------------------------------------- nodes, edges and gradients
x, z, w = leaf(1.0, 2.0), leaf(3.0, 5.0), leaf(0.25, 0.75)
lo, hi, s = leaf(0.5, 2.5), leaf(1.5, 3.0), leaf(4.0)
cases = [
    ("add_", lambda y: y.add_(z), "AddBackward0", [ACC], [4, 7], [[1, 1], [1, 1]], [z]),
    ("add_ scalar", lambda y: y.add_(2.0), "AddBackward0", [None], [3, 4], [[1, 1]], []),
    ("sub_", lambda y: y.sub_(z), "SubBackward0", [ACC], [-2, -3], [[1, 1], [-1, -1]], [z]),
    ("mul_", lambda y: y.mul_(z), "MulBackward0", [ACC], [3, 10], [[3, 5], [1, 2]], [z]),
    ("mul_ scalar", lambda y: y.mul_(2.0), "MulBackward0", [None], [2, 4], [[2, 2]], []),
    ("div_", lambda y: y.div_(z), "DivBackward0", [ACC], [1 / 3, 0.4], [[1 / 3, 0.2], [-1 / 9, -2 / 25]], [z]),
    ("pow_ scalar", lambda y: y.pow_(2.0), "PowBackward0", [], [1, 4], [[2, 4]], []),
    ("pow_", lambda y: y.pow_(z), "PowBackward1", [ACC], [1, 32], [[3, 80], [0, 32 * math.log(2)]], [z]),
    ("clamp_ scalar", lambda y: y.clamp_(0.0, 1.5), "ClampBackward1", [], [1, 1.5], [[1, 0]], []),
    ("clamp_", lambda y: y.clamp_(lo, hi), "ClampBackward0", [ACC, ACC], [1, 2.5], [[1, 0], [0, 1], [0, 0]], [lo, hi]),
    ("addcmul_", lambda y: y.addcmul_(z, w, value=2.0), "AddcmulBackward0", [ACC, ACC], [2.5, 9.5], [[1, 1], [0.5, 1.5], [6, 10]], [z, w]),
    ("addcdiv_", lambda y: y.addcdiv_(z, w, value=2.0), "AddcdivBackward0", [ACC, ACC], [25, 2 + 40 / 3], [[1, 1], [8, 8 / 3], [-96, -10 / 0.5625]], [z, w]),
    ("lerp_ scalar", lambda y: y.lerp_(z, 0.25), "LerpBackward0", [ACC], [1.5, 2.75], [[0.75, 0.75], [0.25, 0.25]], [z]),
    ("lerp_", lambda y: y.lerp_(z, w), "LerpBackward1", [ACC, ACC], [1.5, 4.25], [[0.75, 0.25], [0.25, 0.75], [2, 3]], [z, w]),
    ("index_add_", lambda y: y.index_add_(-1, torch.tensor([1]), s), "IndexAddBackward0", [ACC], [1, 6], [[1, 1], [1]], [s]),
    ("scatter_add_", lambda y: y.scatter_add_(-1, torch.tensor([0]), s), "ScatterAddBackward0", [ACC], [5, 2], [[1, 1], [1]], [s]),
]
for name, op, node, edges, value, expected, operands in cases:
    y = x * 1
    old, version = y.grad_fn, y._version
    op(y)
    assert y._version == version + 1, name
    assert chain(y) == (node, [MUL, *edges]), (name, chain(y))
    assert y.grad_fn.next_functions[0][0] is old, name
    assert y.requires_grad and not y.is_leaf
    assert close(y.tolist(), value), name
    for g, e in zip(grads(y, x, *operands), expected, strict=True):
        assert close(g, e), (name, g, e)

# masked_fill_is_recorded_as_where: the reference has its own node, homura records where_ (and a
# ToCopyBackward0 for a tensor value).
mask = torch.tensor([True, False])
y = x * 1
y.masked_fill_(mask, 7.0)
assert y.tolist() == [7, 2] and chain(y) == ("MaskedFillBackward0", [MUL])
assert grads(y, x) == [[0, 1]]
v = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
y = x * 1
y.masked_fill_(mask, v)
assert y.tolist() == [5, 2] and chain(y) == ("MaskedFillBackward1", [MUL, ACC])
gx, gv = torch.autograd.grad(y.sum(), [x, v])
assert gx.tolist() == [0, 1] and gv.dtype == torch.float64 and gv.item() == 1

# tensors_that_do_not_require_grad_join_the_graph
b = torch.zeros(2)
b.add_(z)
assert b.requires_grad and not b.is_leaf and chain(b) == ("AddBackward0", [None, ACC])
assert grads(b, z) == [[1, 1]]
b = torch.tensor([2.0, 4.0])
b.mul_(z)
assert b.tolist() == [6, 20] and chain(b) == ("MulBackward0", [None, ACC]) and grads(b, z) == [[2, 4]]
d = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).t().detach()
o = torch.ones(2, 2, requires_grad=True)
assert not d.is_contiguous() and d._base is None
d.mul_(o)
assert chain(d) == ("MulBackward0", [None, ACC]) and d.flatten().tolist() == [1, 3, 2, 4]
assert torch.autograd.grad(d.sum(), [o])[0].flatten().tolist() == [1, 3, 2, 4]

# mixed_dtypes_keep_the_dtype_of_self_and_return_gradients_in_their_own
z64 = leaf(3.0, 4.0, dtype=torch.float64)
for op, gx, gz in [("mul_", [3, 4], [1, 2]), ("add_", [1, 1], [1, 1])]:
    y = x * 1
    getattr(y, op)(z64)
    assert y.dtype == torch.float32
    a, b = torch.autograd.grad(y.sum(), [x, z64])
    assert (a.dtype, a.tolist(), b.dtype, b.tolist()) == (torch.float32, gx, torch.float64, gz)
h = torch.tensor([1.5, 2.25]).half().requires_grad_()
z32 = leaf(3.1, 4.7)
y = h * 1
y.mul_(z32)
assert y.dtype == torch.float16 and y.float().tolist() == [4.6484375, 10.578125]
gh, gz = torch.autograd.grad(y, [h, z32], torch.tensor([0.3, 0.7]).half())
assert (gh.dtype, gz.dtype) == (torch.float16, torch.float32)
assert gh.float().tolist() == [0.93017578125, 3.291015625]
assert gz.tolist() == [0.4501953125, 1.5751953125]

# successive_in_place_operations_chain_their_nodes
y = x * 1
y.add_(z)
y.mul_(z)
assert chain(y) == ("MulBackward0", ["AddBackward0", ACC]) and grads(y, x, z) == [[3, 5], [7, 12]]
y = x * 1
y += z
y *= 2.0
assert chain(y) == ("MulBackward0", ["AddBackward0", None]) and grads(y, x, z) == [[2, 2], [2, 2]]

# the_value_before_the_write_is_saved_with_its_history (through CloneBackward0 in the reference,
# directly to the old history in homura; the derivatives are the same)
y = x * 1
y.mul_(z)
(gx,) = torch.autograd.grad(y.sum(), [x], create_graph=True)
assert grads(gx, z) == [[1, 1]]
y = x * 1
y.mul_(z)
(gz,) = torch.autograd.grad(y.sum(), [z], create_graph=True)
assert gz.tolist() == [1, 2] and grads(gz, x) == [[1, 1]]
y = x * 1
y.pow_(3.0)
(g,) = torch.autograd.grad(y.sum(), [x], create_graph=True)
assert g.tolist() == [3, 12] and grads(g, x) == [[6, 12]]
b = torch.tensor([2.0, 4.0])
b.div_(z)
assert close(grads(b, z)[0], [-2 / 9, -4 / 25])

# ---------------------------------------------------------------- saved tensors and versions
# self_as_an_operand_is_saved_as_it_was
y = x * 1
y.add_(y)
assert y.tolist() == [2, 4] and chain(y) == ("AddBackward0", [MUL, MUL]) and grads(y, x) == [[2, 2]]
for op, name in [("mul_", "Mul"), ("div_", "Div"), ("pow_", "Pow1")]:
    y = x * 1
    getattr(y, op)(y)
    assert message(lambda: y.sum().backward()) == version_error("[2]", f", which is output 0 of {name},", 1, 0)
y = x * 1
y.mul_(y.detach())
assert chain(y) == ("MulBackward0", [MUL, None])
assert message(lambda: y.sum().backward()) == version_error("[2]", "", 1, 0)

# saved_results_are_the_target_itself
y = x * 1
y.pow_(z)
torch.autograd.grad(y.sum(), [z])
y = x * 1
y.pow_(z)
with torch.no_grad():
    y.add_(1.0)
assert message(lambda: torch.autograd.grad(y.sum(), [z])) == version_error("[2]", ", which is output 0 of Pow1,", 2, 1)

# writing_into_a_tensor_that_an_earlier_node_saved_is_detected
x0 = leaf(0.0, 1.0)
expected = version_error("[2]", ", which is output 0 of Exp,", 1, 0)
y = x0.exp()
y.mul_(2.0)
assert chain(y) == ("MulBackward0", ["ExpBackward0", None])
assert message(lambda: y.sum().backward()) == expected
y = x0.exp()
y.detach().mul_(2.0)
assert message(lambda: y.sum().backward()) == expected
y = x0.exp()
with torch.no_grad():
    y.mul_(2.0)
assert y.grad_fn.name() == "ExpBackward0" and message(lambda: y.sum().backward()) == expected
a = leaf(1.0, 2.0)
y = a * 1
w2 = y * x0
y.add_(1.0)
assert message(lambda: w2.sum().backward()) == version_error("[2]", ", which is output 0 of Add,", 1, 0)

# optimizer_updates_run_in_no_grad_mode
wt = leaf(1.0, 2.0)
for _ in range(3):
    (wt * wt).sum().backward()
    with torch.no_grad():
        wt.sub_(wt.grad * 0.25)
        wt.grad.zero_()
    assert wt.is_leaf and wt.requires_grad and wt.grad_fn is None
assert wt.tolist() == [0.125, 0.25]

# ---------------------------------------------------------------- rejected writes
# leaves_and_views_are_rejected_before_anything_is_written
x = leaf(1.0, 2.0)
assert message(lambda: x.mul_(2.0)) == LEAF
v = x.view(2)
assert message(lambda: v.add_(1.0)) == VIEW_OF_LEAF
b = torch.zeros(2)
early = b.view(2)
b.requires_grad_()
assert message(lambda: early.add_(1.0)) == VIEW_OF_LEAF
with torch.no_grad():
    quiet = x.narrow(0, 0, 1)
# homura: VIEW_OF_LEAF for a no-grad view of such a leaf too.
assert message(lambda: quiet.add_(1.0)).startswith(NO_GRAD_VIEW)
# Recorded by the reference (CopySlices); homura rejects views until D7's second stage.
y = x * 1
yv = y.view(2)
yv.mul_(2.0)
assert chain(yv)[0] == "AsStridedBackward0" and chain(y)[0] == "CopySlices"
y = x * 1
with torch.no_grad():
    quiet = y.view(2)
assert message(lambda: quiet.mul_(2.0)).startswith(NO_GRAD_VIEW)
for no_grad in [False, True]:
    b = torch.zeros(2)
    with torch.set_grad_enabled(not no_grad):
        own = b.view(2)
    own.requires_grad_()
    expected = NO_GRAD_VIEW if no_grad else VIEW_OF_LEAF
    assert message(lambda: own.add_(1.0)).startswith(expected), no_grad
plain = torch.zeros(2)
pv = plain.view(2)
pv.add_(x)
assert chain(plain)[0] == "CopySlices"
plain = torch.zeros(2)
plain.view(2).add_(1.0)
assert plain.tolist() == [1, 1]
with torch.no_grad():
    v.add_(1.0)
assert x.tolist() == [2, 3]

# failed_writes_change_nothing
x = leaf(1.0, 2.0)
for recorded in [True, False]:
    y = x * 1 if recorded else torch.tensor([1.0, 2.0])
    version, grad_fn = y._version, y.grad_fn
    assert message(lambda: y.add_(torch.ones(2, 2))) == "output with shape [2] doesn't match the broadcast shape [2, 2]"
    s1 = torch.ones(1, requires_grad=recorded)
    # homura's index message is its own (M0).
    assert message(lambda: y.index_add_(0, torch.tensor([2]), s1)) == "index out of range in self"
    e = (x * 1).expand(2, 2).detach()
    assert message(lambda: e.add_(torch.ones(2) * z)).startswith(
        "unsupported operation: more than one element of the written-to tensor refers to a single memory location."
    )
    assert y._version == version and y.grad_fn is grad_fn and y.tolist() == [1, 2]
n = torch.zeros(2, dtype=torch.int64)
assert message(lambda: n.add_(x)) == "result type Float can't be cast to the desired output type Long"
assert n._version == 0

# copies_and_fills_are_not_recorded_yet: recorded by the reference (CopyBackwards, FillBackward2,
# ZeroBackward0, UniformBackward0, NormalBackward0), in homura from M1 step 7b on.
for op, node in [
    (lambda y: y.copy_(x), "CopyBackwards"),
    (lambda y: y.fill_(1.0), "FillBackward2"),
    (lambda y: y.zero_(), "ZeroBackward0"),
    (lambda y: y.uniform_(0.0, 1.0), "UniformBackward0"),
    (lambda y: y.normal_(0.0, 1.0), "NormalBackward0"),
]:
    y = x * 1
    op(y)
    assert chain(y)[0] == node, node

# ---------------------------------------------------------------- edge cases
# zero_dimensional_empty_and_meta_targets
s0 = torch.tensor(2.0, requires_grad=True)
y = s0 * 1
y.mul_(s0)
y.mul_(3.0)
assert y.item() == 12 and torch.autograd.grad(y, [s0])[0].item() == 12
e = torch.zeros(0, 3, requires_grad=True)
y = e * 1
y.mul_(e.sum(0, keepdim=True))
assert y.grad_fn.name() == "MulBackward0" and torch.autograd.grad(y.sum(), [e])[0].shape == (0, 3)
m = torch.zeros(2, 3, device="meta", requires_grad=True)
om = torch.ones(3, device="meta", requires_grad=True)
y = m * 1
y.mul_(om)
y.clamp_(0.0, 1.0)
assert chain(y) == ("ClampBackward1", [MUL])
assert [g.shape for g in torch.autograd.grad(y.sum(), [m, om])] == [(2, 3), (3,)]

# ---------------------------------------------------------------- clamp_ and promoted bounds
# clamp_computes_its_gradient_in_the_dtypes_of_the_computation: homura's mixed bounds are an
# extension; the reference gives the same gradient with the scalar as an f64 tensor bound.
x1 = torch.tensor(1.0, requires_grad=True)
lo1 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
y = x1 * 1
y.clamp_(lo1, torch.tensor(1 + 2**-30, dtype=torch.float64))
gx, glo = torch.autograd.grad(y, [x1, lo1])
assert y.dtype == torch.float32 and (gx.dtype, gx.item(), glo.dtype, glo.item()) == (torch.float32, 1, torch.float64, 0)

# ---------------------------------------------------------------- stale views
# views_of_a_rewritten_history_are_stale: the reference replays views on the new history; homura
# rejects their use (D7's second stage).
x = leaf(1.0, 2.0)
y = x * 3
v = y.view(2)
y.mul_(4.0)
assert v.tolist() == [12, 24] and v.grad_fn.name() == "AsStridedBackward0"
assert [g.tolist() for g in torch.autograd.grad(v.sum(), [x], retain_graph=True)] == [[12, 12]]
fresh = y.view(2)
assert fresh.grad_fn.name() == "ViewBackward0" and grads(fresh, x) == [[12, 12]]

# a_leaf_that_joined_the_graph_rejects_old_backward_passes (R16)
w3 = leaf(1.0, 2.0)
out = (w3 * 2).sum()
w3.requires_grad_(False)
w3.add_(z)
assert not w3.is_leaf and w3.requires_grad
assert message(lambda: out.backward()) == "leaf variable has been moved into the graph interior"
assert grads(w3, z) == [[1, 1]]

print("check_m1_inplace: all reference checks passed")
