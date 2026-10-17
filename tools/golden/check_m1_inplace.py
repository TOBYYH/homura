"""Check the reference behavior that tests/autograd_inplace.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order. Where homura deliberately differs (masked_fill
recorded as where_, homura's messages for views it doesn't support yet, mixed scalar and tensor
clamp bounds, namespaces in node names), the PyTorch behavior is asserted and the difference is
noted.
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


def grads_with(out, g, *inputs, **options):
    return [t.tolist() for t in torch.autograd.grad(out, inputs, g, **options)]


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
base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
n = base.t().narrow(0, 1, 2).detach()
w4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
assert n._base is None and n.storage_offset() == 1
n.add_(w4)
n.mul_(w4)
assert chain(n) == ("MulBackward0", ["AddBackward0", ACC])
assert n.flatten().tolist() == [3, 14, 18, 40] and base.flatten().tolist() == [1, 3, 18, 4, 14, 40]
assert torch.autograd.grad(n.sum(), [w4])[0].flatten().tolist() == [4, 9, 9, 14]
v4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
n.copy_(v4.sum(0, keepdim=True))
assert base.flatten().tolist() == [1, 4, 4, 4, 6, 6]
assert torch.autograd.grad(n.sum(), [v4])[0].flatten().tolist() == [2, 2, 2, 2]

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
    # Copies and fills. homura's broadcast message names the tensors the other way round (M0).
    src = torch.tensor([1.0, 2.0, 3.0], requires_grad=recorded)
    assert message(lambda: y.copy_(src)) == "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0"
    assert message(lambda: e.copy_(z if recorded else torch.ones(2))).startswith("unsupported operation: more than one element")
    # The reference fills an expanded tensor; homura rejects every write into one (M0).
    e.fill_(1.0)
    target = y * 1 if recorded else y
    assert message(lambda: target.uniform_(2.0, 1.0)) == "uniform_ expects to return a [from, to) range, but found from=2 > to=1"
    assert y._version == version and y.grad_fn is grad_fn and y.tolist() == [1, 2]
y = x * 1
version, grad_fn = y._version, y.grad_fn
with torch.no_grad():
    assert message(lambda: y.mul_(torch.ones(2, 2))) == "output with shape [2] doesn't match the broadcast shape [2, 2]"
    assert message(lambda: y.index_add_(0, torch.tensor([2]), torch.ones(1))) == "index out of range in self"
    e = (x * 1).expand(2, 2).detach()
    for op in [lambda e: e.add_(1.0), lambda e: e.copy_(z)]:
        assert message(lambda: op(e)).startswith("unsupported operation: more than one element")
    # The reference fills an expanded tensor; homura rejects every write into one (M0).
    e.fill_(1.0)
    assert message(lambda: y.copy_(torch.ones(2, device="meta"))) == "Cannot copy out of meta tensor; no data!"
assert y._version == version and y.grad_fn is grad_fn and y.tolist() == [1, 2]
assert message(lambda: x.uniform_(2.0, 1.0)) == LEAF
n = torch.zeros(2, dtype=torch.int64)
assert message(lambda: n.add_(x)) == "result type Float can't be cast to the desired output type Long"
assert n._version == 0

# promoted_and_broadcast_operands_get_gradients_summed_where_they_were_computed (review round 30)
e30 = 2.0**-30
x30 = leaf(1.0, e30)
for name, g, expected in [
    ("add_", [1, e30], 1.0),
    ("sub_", [1, e30], -1.0),
    ("mul_", [1, 1], 1.0),
    ("copy_", [1, e30], 1 + e30),
]:
    o = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    y = x30 * 1
    getattr(y, name)(o)
    (go,) = torch.autograd.grad(y, [o], torch.tensor(g))
    assert y.dtype == torch.float32 and go.dtype == torch.float64 and go.tolist() == [expected], name

# ---------------------------------------------------------------- copies and fills
# copies_and_fills_give_the_overwritten_value_a_zero_gradient. The reference names CopyBackwards
# with its namespace, in version errors too; homura leaves it out, as for AccumulateGrad.
x, z = leaf(1.0, 2.0), leaf(3.0, 5.0)
overwrites = [
    ("copy_", lambda y: y.copy_(z), "CopyBackwards", [ACC], [3, 5], [[0, 0], [1, 1]], "torch::autograd::CopyBackwards"),
    ("copy_ self", lambda y: y.copy_(y), "CopyBackwards", [MUL], [1, 2], [[1, 1]], "torch::autograd::CopyBackwards"),
    ("fill_", lambda y: y.fill_(7.0), "FillBackward2", [], [7, 7], [[0, 0]], "Fill2"),
    ("zero_", lambda y: y.zero_(), "ZeroBackward0", [], [0, 0], [[0, 0]], "Zero"),
    ("uniform_", lambda y: y.uniform_(0.0, 1.0), "UniformBackward0", [], None, [[0, 0]], "Uniform"),
    ("normal_", lambda y: y.normal_(0.0, 1.0), "NormalBackward0", [], None, [[0, 0]], "Normal"),
]
for name, op, node, edges, value, expected, operation in overwrites:
    y = x * 1
    old, version = y.grad_fn, y._version
    op(y)
    assert y._version == version + 1 and chain(y) == (node, [MUL, *edges]), name
    assert y.grad_fn.next_functions[0][0] is old and y.requires_grad and not y.is_leaf, name
    assert value is None or y.tolist() == value, name
    if name != "copy_ self":
        nan_inf = torch.tensor([math.nan, math.inf])
        assert grads_with(y, nan_inf, x, retain_graph=True) == [[0, 0]], name
    assert grads(y, *[x, z][: len(expected)]) == expected, name
    w = y * y
    with torch.no_grad():
        y.add_(1.0)
    assert message(lambda: w.sum().backward()) == version_error("[2]", f", which is output 0 of {operation},", 2, 1), name
for name, op, *_ in overwrites:
    b = torch.zeros(2)
    z_plain = torch.tensor([3.0, 5.0])
    (b.copy_(z_plain) if name == "copy_" else op(b))
    assert b.is_leaf and not b.requires_grad, name

# copy_converts_the_gradient_of_src_and_sums_it_to_its_shape
s1 = leaf(0.5)
y = x * 1
y.copy_(s1)
assert y.tolist() == [0.5, 0.5] and grads_with(y, torch.tensor([1.0, 2.0]), s1) == [[3]]
b = torch.zeros(3, 2)
d = leaf(1.0, 2.0, dtype=torch.float64)
b.copy_(d)
assert b.requires_grad and not b.is_leaf and chain(b) == ("CopyBackwards", [None, ACC])
(gd,) = torch.autograd.grad(b.sum(), [d])
assert gd.dtype == torch.float64 and gd.tolist() == [3, 3]
y = x * 1
y.copy_(torch.tensor([7, 8]))
assert chain(y) == ("CopyBackwards", [MUL, None]) and y.tolist() == [7, 8] and grads(y, x) == [[0, 0]]
n = torch.zeros(2, dtype=torch.int64)
n.copy_(x)
assert n.tolist() == [1, 2] and n.grad_fn is None and not n.requires_grad
m = torch.zeros(2, device="meta")
m.copy_(x)
assert chain(m) == ("CopyBackwards", [None, ACC])
assert message(lambda: torch.autograd.grad(m.sum(), [x])) == "Cannot copy out of meta tensor; no data!"
mo = torch.ones(2, device="meta", requires_grad=True)
m = torch.zeros(3, 2, device="meta")
m.copy_(mo)
(g,) = torch.autograd.grad(m.sum(), [mo])
assert g.device.type == "meta" and g.shape == (2,)
mm = torch.ones(2, device="meta", requires_grad=True)
m = mm * 1
m.copy_(x)
(g,) = torch.autograd.grad(m.sum(), [mm])
assert g.device.type == "meta" and g.shape == (2,)
y = x * 1
assert message(lambda: y.copy_(mo)) == "Cannot copy out of meta tensor; no data!"
assert y._version == 0 and y.tolist() == [1, 2]
h = torch.tensor([0.5], dtype=torch.float16, requires_grad=True)
y = x * 1
y.copy_(h)
gx, gh = torch.autograd.grad(y, [x, h], torch.tensor([70000.0, -70000.0]))
assert gx.tolist() == [0, 0] and gh.dtype == torch.float16 and math.isnan(gh.item())
s0 = torch.tensor(2.0, requires_grad=True)
y = s0 * 1
y.copy_(x.sum())
assert grads(y, s0, x) == [0, [1, 1]]
e = torch.zeros(0, 2, requires_grad=True)
y = e * 1
y.fill_(1.0)
y.copy_(x)
assert torch.autograd.grad(y.sum(), [e])[0].shape == (0, 2)

# gradients_through_copies_and_fills_can_be_differentiated_again
y = x * x
y.copy_(z * z)
gx, gz = torch.autograd.grad(y.sum(), [x, z], create_graph=True)
assert gx.tolist() == [0, 0] and gz.tolist() == [6, 10]
assert [t.tolist() for t in torch.autograd.grad(gz.sum(), [z], retain_graph=True)] == [[2, 2]]
assert [t.tolist() for t in torch.autograd.grad(gx.sum(), [x], retain_graph=True)] == [[0, 0]]
y = x * x
y.zero_()
assert torch.autograd.grad(y.sum(), [x], create_graph=True)[0].tolist() == [0, 0]
y = x * 1
y.copy_(z)
y.mul_(z)
assert chain(y) == ("MulBackward0", ["CopyBackwards", ACC]) and grads(y, x, z) == [[0, 0], [6, 10]]
y = x * 2
y.copy_(z * x)
gy, gz = torch.autograd.grad((y * y).sum(), [y, z], create_graph=True)
assert gy.tolist() == [6, 20] and gz.tolist() == [6, 40]
assert [t.tolist() for t in torch.autograd.grad(gz.sum(), [y, x, z], retain_graph=True)] == [[2, 4], [12, 40], [2, 8]]
assert [t.tolist() for t in torch.autograd.grad(gy.sum(), [x, z], retain_graph=True)] == [[6, 10], [2, 4]]

# overwritten_values_get_zero_gradients_under_create_graph (R43): with a differentiable upstream
# gradient, the gradient through each overwrite is zero (connected to x), and its own gradient is
# zero for x and unused for u.
for name, op, *_ in overwrites:
    if name == "copy_ self":
        continue
    x = leaf(0.5, -1.25)
    u = leaf(1.0, -0.5)
    y = x * x
    op(y)
    (g,) = torch.autograd.grad(y, [x], u, create_graph=True)
    assert g.tolist() == [0, 0] and g.requires_grad, name
    gx, gu = torch.autograd.grad(g.sum(), [x, u], allow_unused=True)
    assert gx.tolist() == [0, 0] and gu is None, name
x, z = leaf(1.0, 2.0), leaf(3.0, 5.0)

# copies_and_fills_into_leaves_and_views_are_rejected: as for the other in-place operations, the
# reference records views of non-leaves (CopySlices) and has a message of its own for views
# created in no-grad mode; homura rejects them (D7).
for name, op, *_ in overwrites:
    assert message(lambda: op(x)) == LEAF, name
    v = x.view(2)
    assert message(lambda: op(v)) == VIEW_OF_LEAF, name
    b = torch.zeros(2)
    early = b.view(2)
    b.requires_grad_()
    assert message(lambda: op(early)) == VIEW_OF_LEAF, name
    with torch.no_grad():
        quiet = x.view(2)
    assert message(lambda: op(quiet)).startswith(NO_GRAD_VIEW), name
    y = x * 1
    yv = y.view(2)
    op(yv)
    assert chain(y)[0] == "CopySlices", name
    y = x * 1
    with torch.no_grad():
        quiet = y.view(2)
    assert message(lambda: op(quiet)).startswith(NO_GRAD_VIEW), name
    for no_grad in [False, True]:
        b = torch.zeros(2)
        with torch.set_grad_enabled(not no_grad):
            own = b.view(2)
        own.requires_grad_()
        expected = NO_GRAD_VIEW if no_grad else VIEW_OF_LEAF
        assert message(lambda: op(own)).startswith(expected), (name, no_grad)
    y = x * 1
    yv = y.view(2)
    with torch.no_grad():
        op(v)
        op(yv)
    assert chain(y) == (MUL, [ACC, None]), name
x = leaf(1.0, 2.0)
pv = torch.zeros(2).view(2)
pv.copy_(x)
assert chain(pv)[0] == "AsStridedBackward0"
pv = torch.zeros(2).view(2)
pv.copy_(torch.tensor([1.0, 2.0]))
assert pv.tolist() == [1, 2]

# views_of_a_copied_or_filled_tensor_are_stale: replayed by the reference, rejected by homura.
for name, op, node, edges, value, expected, operation in overwrites:
    y = x * 1
    v = y.view(2)
    op(y)
    assert v.grad_fn.name() == "AsStridedBackward0", name
    assert grads(y.view(2), x) == expected[:1], name

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

# tests/random.rs, rejected_fills_leave_the_generator_alone (review round 31, R42): rejected
# random fills leave the generator state unchanged; invalid arguments come before the overlap,
# autograd's errors before both.
for fill, invalid, invalid_message in [
    (lambda t: t.uniform_(0.0, 1.0), lambda t: t.uniform_(2.0, 1.0), "uniform_ expects to return a [from, to) range, but found from=2 > to=1"),
    (lambda t: t.normal_(0.0, 1.0), lambda t: t.normal_(0.0, -1.0), "normal expects std >= 0.0, but found std -1"),
]:
    for no_grad in [False, True]:
        with torch.set_grad_enabled(not no_grad):
            for target in [torch.ones(1).expand(4), torch.ones(1).expand(4).detach()]:
                state = torch.get_rng_state()
                assert message(lambda: fill(target)).startswith("unsupported operation: more than one element")
                assert message(lambda: invalid(target)) == invalid_message
                assert torch.equal(torch.get_rng_state(), state) and target.tolist() == [1, 1, 1, 1]
    xl = torch.ones(4, requires_grad=True)
    state = torch.get_rng_state()
    assert message(lambda: fill(xl)) == LEAF and message(lambda: xl.uniform_(2.0, 1.0)) == LEAF
    assert message(lambda: fill(xl.view(4))) == VIEW_OF_LEAF
    assert torch.equal(torch.get_rng_state(), state)

print("check_m1_inplace: all reference checks passed")
