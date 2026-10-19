"""Check the reference behavior that tests/index_advanced.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: result positions and the parsing of index
expressions, node chains, index errors, index_put errors, aliasing (only in-place index_put
rejects it), layouts, results that are new tensors, saved index tensors, accumulation, the
gradients of overwriting puts, derivatives with respect to the upstream gradient, degenerate
indices, meta tensors and the in-place rules. Where homura differs on purpose, the reference
behavior is asserted and the difference named in a comment. The golden test (whose data
gen_m2_index.py writes) and the test of `bool` items under another default device have no
counterpart here.
"""

import warnings

import torch

import reference

reference.check()
warnings.simplefilter("ignore")
d64 = torch.float64
INF, NAN = float("inf"), float("nan")
passed = 0


def block(f):
    global passed
    f()
    passed += 1
    print("PASS", f.__name__)
    return f


def raises(exc, message, f):
    """Asserts that f raises exc whose first line contains message."""
    try:
        f()
    except exc as e:
        assert message in str(e).splitlines()[0], (str(e), message)
        return
    raise AssertionError(f"no {exc.__name__}: {message}")


def chain(t):
    out = []

    def walk(fn):
        if fn is None:
            return
        out.append(type(fn).__name__)
        for nf, _ in fn.next_functions:
            walk(nf)

    walk(t.grad_fn)
    return out


def ints(*values):
    return torch.tensor(values, dtype=torch.int64)


def mask(*values):
    return torch.tensor(values, dtype=torch.bool)


def counting(*shape):
    n = 1
    for s in shape:
        n *= s
    return torch.arange(n, dtype=d64).reshape(shape).requires_grad_()


def grads(out, inputs, g, create_graph=False):
    return torch.autograd.grad(out, inputs, g, create_graph=create_graph, retain_graph=True,
                               allow_unused=True)


ACC = "AccumulateGrad"


@block
def result_positions_follow_pytorch():
    x = torch.zeros(4, 5, 6, 7)
    i, j = ints(0, 1), ints(2, 3)
    assert x[:, i, j].shape == (4, 2, 7)
    assert x[i, :, j].shape == (2, 5, 7)
    assert x[:, i, :, j].shape == (2, 4, 6)
    assert x[:, i, 0, j].shape == (4, 2)
    assert x[:, i, 0:6, j].shape == (2, 4, 6)
    assert x[:, i, None, j].shape == (2, 4, 1, 7)
    x3 = torch.zeros(3, 4, 5)
    assert x3[:, ints(0, 1), ..., ints(1, 2)].shape == (3, 2)
    assert x[i, ..., j].shape == (2, 5, 6)
    i21 = torch.tensor([[0], [1]])
    assert x[i21, ints(0, 1, 2)].shape == (2, 3, 6, 7)
    m2 = torch.ones(4, 5, dtype=torch.bool)
    m2[0, 0] = False
    assert x[m2, :, ints(1)].shape == (19, 6)
    assert torch.equal(torch.ops.aten.index.Tensor(x, [m2, None, ints(1)]), x[m2, :, ints(1)])
    assert x[m2, 0].shape == (19, 7)
    assert x[m2, None].shape == (19, 1, 6, 7)
    assert x[m2, 1:3].shape == (19, 2, 7)
    m5 = mask(True, False, True, True, False)
    assert x[i21, m5].shape == (2, 3, 6, 7)
    assert x[:, m5].shape == (4, 3, 6, 7)
    assert x[mask(False, False, False, False)].shape == (0, 5, 6, 7)
    assert x[True].shape == (1, 4, 5, 6, 7)
    assert x[False].shape == (0, 4, 5, 6, 7)
    assert x[torch.tensor(True)].shape == (1, 4, 5, 6, 7)
    assert x[i, True].shape == (2, 5, 6, 7)
    assert torch.zeros(())[True].shape == (1,)
    empty = torch.empty(0, dtype=torch.int64)
    assert x[ints(0)[:0], empty].shape == (0, 6, 7)


@block
def index_records_one_node_after_the_views():
    x = counting(3, 4)
    i = ints(2, 0)
    assert chain(x[i]) == ["IndexBackward0", ACC]
    assert chain(x[i, :]) == ["IndexBackward0", ACC]
    assert chain(x[0, i]) == ["IndexBackward0", "SelectBackward0", ACC]
    assert chain(x[i, ints(1, 1)]) == ["IndexBackward0", ACC]
    for spec in (True, (True, slice(None)), torch.tensor(True)):
        assert chain(x[spec]) == ["IndexBackward0", "UnsqueezeBackward0", ACC]
    picked = x[torch.tensor(1)]
    assert chain(picked) == ["SelectBackward0", ACC] and picked.storage_offset() == 4
    v = counting(2, 4)
    assert chain(x.index_put((i,), v)) == ["IndexPutBackward0", ACC, ACC]
    y = (x * 1.0).index_put((i,), v.detach(), True)
    assert chain(y) == ["IndexPutBackward0", "MulBackward0", ACC]
    y = x.detach().index_put((i,), v, True)
    assert [None if f is None else type(f).__name__ for f, _ in y.grad_fn.next_functions] == \
        [None, ACC]


@block
def index_errors_follow_pytorch():
    x = torch.zeros(4, 5, 6, 7)
    i = ints(0, 1)
    msg = "tensors used as indices must be long, int, byte or bool tensors"
    for dtype in (torch.int8, torch.int16, torch.float32):
        raises(IndexError, msg, lambda: x[torch.zeros(1, dtype=dtype)])
    raises(IndexError, msg, lambda: x[torch.zeros((), dtype=torch.float32)])
    # homura rejects these too (PyTorch: a select for 0-dim I8 / I16, a deprecated u8 mask).
    assert x[torch.tensor(1, dtype=torch.int8)].shape == (5, 6, 7)
    assert x[torch.ones(4, dtype=torch.uint8)].shape == (4, 5, 6, 7)
    raises(IndexError, "index 9 is out of bounds for dimension 0 with size 5", lambda: x[:, ints(9)])
    raises(IndexError, "index -6 is out of bounds for dimension 1 with size 5",
           lambda: x[i, ints(0, -6)])
    raises(IndexError, "index 6 is out of bounds for dimension 1 with size 6",
           lambda: x[i, :, ints(0, 6)])
    raises(IndexError, "index is out of bounds for dimension with size 0",
           lambda: torch.zeros(3, 0)[:, ints(0)])
    raises(IndexError, "too many indices for tensor of dimension 4", lambda: x[i, i, i, i, i])
    raises(IndexError, "too many indices for tensor of dimension 0", lambda: torch.zeros(())[i])
    raises(IndexError, "too many indices for tensor of dimension 1",
           lambda: torch.zeros(3)[torch.ones(3, 1, dtype=torch.bool)])
    raises(IndexError, "shape mismatch: indexing tensors could not be broadcast together with "
           "shapes [2], [3]", lambda: x[i, ints(0, 1, 2)])
    raises(IndexError, "could not be broadcast together with shapes [20], [20], [2]",
           lambda: x[torch.ones(4, 5, dtype=torch.bool), i])
    raises(IndexError, "The shape of the mask [3, 5] at index 0 does not match the shape of the "
           "indexed tensor [4, 5, 6, 7] at index 0", lambda: x[torch.ones(3, 5, dtype=torch.bool)])
    raises(IndexError, "The shape of the mask [5, 4] at index 1 does not match the shape of the "
           "indexed tensor [4, 5, 6, 7] at index 2",
           lambda: x[:, torch.ones(5, 4, dtype=torch.bool)])
    # homura's select reports its own message.
    raises(IndexError, "index 4 is out of bounds for dimension 0 with size 4",
           lambda: x[torch.tensor(4)])


@block
def index_put_errors_change_nothing():
    x = counting(4, 3) * 1.0
    i = ints(0, 2)
    v = torch.ones(2, 3, dtype=d64)
    before = x.detach().clone()
    raises(RuntimeError, "Index put requires the source and destination dtypes match, got Double "
           "for the destination and Float for the source.", lambda: x.index_put_((i,), v.float()))
    raises(RuntimeError, "shape mismatch: value tensor of shape [1, 2, 3] cannot be broadcast to "
           "indexing result of shape [2, 3]", lambda: x.index_put_((i,), torch.ones(1, 2, 3, dtype=d64)))
    raises(RuntimeError, "value tensor of shape [3, 3] cannot be broadcast to indexing result of "
           "shape [2, 3]", lambda: x.index_put_((i,), torch.ones(3, 3, dtype=d64)))
    raises(IndexError, "too many indices for tensor of dimension 2 (got 3)",
           lambda: x.index_put_((i, i, i), v))
    assert torch.equal(x.detach(), before) and x.grad_fn.name() == "MulBackward0"
    # Out-of-range positions: PyTorch writes the rows before the bad one, without bumping the
    # version (homura checks every position before it writes anything).
    version = x._version
    raises(IndexError, "index 4 is out of bounds for dimension 0 with size 4",
           lambda: x.index_put_((ints(0, 4),), v))
    assert not torch.equal(x.detach(), before) and x._version == version
    assert x.detach()[0].tolist() == [1, 1, 1]
    # The values may be a meta tensor in PyTorch (homura: same device as the target), and an
    # index on the meta device is not checked for a CPU target (homura rejects it).
    # Python's index_put_ takes tensors only; ATen's list takes None for a full slice and
    # fails internally without a tensor.
    raises(TypeError, "", lambda: x.index_put_((0, i), v))
    raises(RuntimeError, "", lambda: torch.ops.aten.index_put(x, [None], v))
    # A 0-dim bool mask takes the masked_fill path (homura rejects it).
    assert torch.ops.aten.index_put(torch.zeros(2), [torch.tensor(True)], torch.tensor(1.)).sum() == 2
    raises(RuntimeError, "Index put requires the source and destination dtypes match",
           lambda: x.index_put((i,), v.float()))


@block
def only_in_place_index_put_rejects_aliasing():
    overlap = "unsupported operation: some elements of the input tensor and the written-to tensor " \
              "refer to a single memory location."
    x = torch.arange(4.0)
    reverse = ints(3, 2, 1, 0)
    assert x.index_put((reverse,), x).tolist() == [3, 2, 1, 0] and x.tolist() == [0, 1, 2, 3]
    raises(RuntimeError, overlap, lambda: x.index_put_((reverse,), x))
    part = x[:2]
    i = ints(2, 3)
    assert x.index_put((i,), part, True).tolist() == [0, 1, 2, 4]
    raises(RuntimeError, overlap, lambda: x.index_put_((i,), part, True))
    n = ints(0, 1, 2)
    assert n.index_put((n,), torch.tensor(9)).tolist() == [9, 9, 9]
    raises(RuntimeError, overlap, lambda: n.index_put_((n,), torch.tensor(9)))
    e = torch.ones(1).expand(3)
    assert e.index_put((ints(1),), torch.tensor(5.0)).tolist() == [1, 5, 1]
    # PyTorch warns and writes (homura rejects the expanded target in place).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.ones(1).expand(3).index_put_((ints(1),), torch.tensor(5.0))
        assert any("expanded tensors" in str(m.message) for m in w)
    base = torch.arange(6.0)
    base[:3].index_put_((ints(0, 2),), base[3:5])
    assert base.tolist() == [3, 1, 4, 3, 4, 5]
    # Interleaving views: PyTorch's check sees no shared element (homura rejects them).
    even, odd = base[0::2], base[1::2]
    even.index_put_((ints(0, 1, 2),), odd)
    x = counting(3, 2)
    v = counting(2)
    y = x.index_put((ints(1),), v)
    for g in (torch.tensor(1.0, dtype=d64).expand(3, 2),
              torch.arange(1.0, 7.0, dtype=d64).reshape(2, 3).t()):
        gx, gv = grads(y, (x, v), g)
        gd = g.reshape(-1)
        assert gx.reshape(-1).tolist() == [gd[0], gd[1], 0, 0, gd[4], gd[5]]
        assert gv.tolist() == [gd[2], gd[3]]


@block
def index_put_handles_any_target_layout():
    i, j = ints(3, 0, 2), ints(1, 3, 0)
    v = -1.0 - torch.arange(9.0).reshape(3, 3)
    targets = [
        torch.arange(48.0).reshape(4, 3, 4),
        torch.arange(48.0).reshape(4, 4, 3).transpose(1, 2),
        torch.arange(60.0).reshape(5, 3, 4)[1:],
        torch.arange(48.0).reshape(3, 4, 4).permute(1, 0, 2),
    ]
    for x in targets:
        for accumulate in (False, True):
            expected = x.clone()
            for n, (r, c) in enumerate(zip(i.tolist(), j.tolist())):
                expected[r, :, c] = (expected[r, :, c] if accumulate else 0) + v[n]
            y = torch.ops.aten.index_put(x, [i, None, j], v, accumulate)
            assert torch.equal(y, expected)
            yt = torch.ops.aten.index_put(x.transpose(0, 1), [None, i, j], v.t(), accumulate)
            assert torch.equal(yt.transpose(0, 1), expected)
            big = torch.zeros(6, 3, 4)
            view = big[1:5]
            view.copy_(x)
            torch.ops.aten.index_put_(view, [i, None, j], v, accumulate)
            assert torch.equal(view, expected)
            assert big[0].abs().sum() == 0 and big[5].abs().sum() == 0


@block
def in_place_index_put_writes_through_non_contiguous_targets():
    for adjacent in (False, True):
        for accumulate in (False, True):
            base = torch.arange(120, dtype=d64).reshape(6, 4, 5)
            target = base[1:6:2].permute(2, 0, 1)
            assert target.shape == (5, 3, 4) and not target.is_contiguous()
            assert target.storage_offset() == 20
            if adjacent:
                entries = [None, ints(2, 0), ints(1, 3)]
                v = torch.arange(10, dtype=d64).reshape(2, 5).t() + 100
            else:
                entries = [ints(4, 1), None, ints(1, 3)]
                v = torch.arange(6, dtype=d64).reshape(3, 2).t() + 100
            expected = target.clone()
            torch.ops.aten.index_put_(expected, entries, v, accumulate)
            whole = base.clone()
            whole[1:6:2].permute(2, 0, 1).copy_(expected)
            torch.ops.aten.index_put_(target, entries, v, accumulate)
            assert torch.equal(base, whole)


@block
def advanced_indexing_results_are_new_tensors():
    # PyTorch's out-of-place index_put keeps the strides of its input (homura's result is
    # contiguous in the order the indexing works in).
    t = torch.arange(12.0).reshape(3, 4).t()
    assert t.index_put((ints(0),), torch.tensor(9.0)).stride() == (1, 4)
    x = torch.arange(24.0).reshape(2, 3, 4)
    i = ints(2, 0)
    for r in (x[:, i], x[ints(1), :, ints(3)], x[x > 5], x[True],
              torch.ops.aten.index_put(x, [None, i], torch.tensor(1.0)),
              torch.ops.aten.index_put(x, [ints(1), None, ints(3)], torch.tensor(1.0), True)):
        assert r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        r.add_(100)
        assert torch.equal(x, torch.arange(24.0).reshape(2, 3, 4))
    leaf = counting(2, 3)
    y = leaf[:, i[:1]]
    y.mul_(2.0)
    gx, = grads(y, (leaf,), torch.ones(2, 1, dtype=d64))
    assert gx.reshape(-1).tolist() == [0, 0, 2, 0, 0, 2]


@block
def saved_indices_are_version_checked_unless_consumed_by_value():
    modified = "one of the variables needed for gradient computation has been modified"
    x = counting(3)
    m = mask(True, False, True)
    y = x[m]
    m.fill_(False)
    raises(RuntimeError, modified, lambda: grads(y, (x,), torch.ones(2, dtype=d64)))
    i = ints(2, 0)
    y = x[i]
    i.fill_(1)
    raises(RuntimeError, modified, lambda: grads(y, (x,), torch.ones(2, dtype=d64)))
    for t in (torch.tensor(1), torch.tensor(True), torch.tensor(1, dtype=torch.int32)):
        y = x[t]
        t.fill_(0)
        gx, = grads(y, (x,), torch.ones(y.shape, dtype=d64))
        assert gx.tolist() == ([1, 1, 1] if t.dtype == torch.bool else [0, 1, 0])
    for in_place in (False, True):
        for accumulate in (False, True):
            for dtype in (torch.int64, torch.int32):
                x = counting(3)
                v = counting()
                i = torch.tensor(1, dtype=dtype)
                y = x * 1.0
                y = y.index_put_((i,), v, accumulate) if in_place else \
                    y.index_put((i,), v, accumulate)
                assert y.shape == (3,)
                i.fill_(2)
                for inputs in ((x, v), (x,), (v,)):
                    raises(RuntimeError, modified,
                           lambda: grads(y, inputs, torch.ones(3, dtype=d64)))
    x = counting(3)
    v = counting()
    i = torch.tensor(1)
    y = x.index_put((i,), v)
    g = torch.ones(3, dtype=d64, requires_grad=True)
    gv, = grads(y, (v,), g, True)
    assert chain(gv)[0] == "IndexBackward0"
    i.fill_(0)
    raises(RuntimeError, modified, lambda: grads(gv, (g,), torch.tensor(1.0, dtype=d64)))


@block
def graphs_free_their_saved_indices():
    x = counting(4)
    y = x[mask(True, False, True, True)]
    g = torch.ones(3, dtype=d64)
    y.backward(g, retain_graph=True)
    y.backward(g)
    assert x.grad.tolist() == [2, 0, 2, 2]
    raises(RuntimeError, "Trying to backward through the graph a second time",
           lambda: y.backward(g))


@block
def accumulation_sums_in_f64_and_rounds_once():
    # PyTorch rounds after each addition: 0 where homura gets 1.
    for dtype, start in ((torch.float16, 2048), (torch.bfloat16, 256), (torch.float32, 16777216)):
        x = torch.tensor([start], dtype=dtype)
        y = x.index_put((ints(0, 0),), torch.tensor([1.0, -start], dtype=dtype), True)
        assert y.item() == 0
        assert float(start) + 1.0 - start == 1.0
    x = torch.tensor([INF, 1.0, NAN, 0.0], dtype=d64)
    v = torch.tensor([-INF, 2.0, 3.0, INF, INF], dtype=d64)
    r = x.index_put((ints(0, 1, 2, 3, 3),), v, True).tolist()
    assert r[0] != r[0] and r[2] != r[2] and r[1] == 3 and r[3] == INF
    n = torch.tensor([127], dtype=torch.int8)
    assert n.index_put((ints(0, 0),), torch.tensor([1, 1], dtype=torch.int8), True).tolist() == \
        [-127]
    b = mask(False, True, False)
    assert b.index_put((ints(0, 0, 1),), mask(True, True, False), True).tolist() == \
        [True, True, False]


@block
def overwriting_puts_zero_the_gradient_where_they_write():
    x = counting(3)
    v = torch.tensor([5.0, 6.0], dtype=d64, requires_grad=True)
    y = x.index_put((ints(1, 1),), v)
    assert y[1].item() in (5.0, 6.0)
    gx, gv = grads(y, (x, v), torch.tensor([1.0, 2.0, 3.0], dtype=d64))
    assert gx.tolist() == [1, 0, 3] and gv.tolist() == [2, 2]
    v = counting(2)
    y = x.index_put((ints(1, 2),), v)
    gx, gv = grads(y, (x, v), torch.tensor([1.0, INF, NAN], dtype=d64))
    assert gx.tolist() == [1, 0, 0]
    assert gv[0] == INF and gv[1] != gv[1]


@block
def gradients_pass_gradcheck():
    from torch.autograd import gradcheck, gradgradcheck

    def check(f, inputs):
        assert gradcheck(f, inputs) and gradgradcheck(f, inputs)

    x = counting(4, 3, 5)
    i, j = ints(3, 0, 3), ints(1, -1, 2)
    m = mask(True, False, True)
    i21 = torch.tensor([[0], [2]])
    check(lambda t: t[:, ints(2, 0, 1), j], (x,))
    check(lambda t: t[i, :, j], (x,))
    check(lambda t: t[:, m], (x,))
    check(lambda t: t[i21, m], (x,))
    check(lambda t: t[0, True, j], (x,))
    v = counting(3, 3)
    k = ints(3, 0, 1)
    for accumulate in (False, True):
        check(lambda t, w: torch.ops.aten.index_put(t, [k, None, j], w, accumulate), (x, v))
        check(lambda t, w: torch.ops.aten.index_put(t, [k, None, j], w, accumulate),
              (x, counting(3)))
        check(lambda t, w: torch.ops.aten.index_put(t, [None, m], w, accumulate), (x, counting()))
    check(lambda t, w: t.index_put((ints(1, 1, 3),), w, True), (x, counting(3, 3, 5)))
    y = counting(2, 3)
    reversed_rows = ints(1, 0)
    for accumulate in (False, True):
        check(lambda t: t.index_put((reversed_rows,), t.pow(2), accumulate)[:, ints(2, 0)].pow(2),
              (y,))


@block
def derivatives_connect_to_the_upstream_gradient():
    x = counting(4)
    y = x[ints(1, 3, 1)]
    g = torch.tensor([1.0, 2.0, 4.0], dtype=d64, requires_grad=True)
    gx, = grads(y, (x,), g, True)
    assert gx.tolist() == [0, 5, 0, 2]
    # homura names this node IndexPutBackward0.
    assert gx.grad_fn.name() == "IndexPutImplBackward0"
    gg = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=d64, requires_grad=True)
    dx, dg = grads(gx, (x, g), gg, True)
    assert dx is None and dg.tolist() == [10, 1000, 10]
    third, = grads(dg, (gg,), torch.tensor([1.0, 2.0, 3.0], dtype=d64))
    assert third.tolist() == [0, 4, 0, 2]
    v = counting(2)
    y = x.index_put((ints(0, 2),), v)
    g = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=d64, requires_grad=True)
    gx, gv = grads(y, (x, v), g, True)
    total = (gx * torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=d64)).sum() + \
        (gv * torch.tensor([7.0, 9.0], dtype=d64)).sum()
    r = torch.autograd.grad(total, (x, v, g), allow_unused=True)
    assert r[0] is None and r[1] is None and r[2].tolist() == [7, 10, 9, 1000]


@block
def degenerate_indices_keep_their_gradients():
    empty = torch.empty(0, dtype=torch.int64)
    cases = [
        (counting(3, 4), lambda x: x[:, empty]),
        (counting(3, 4), lambda x: x[mask(False, False, False)]),
        (counting(0, 4), lambda x: x[:, ints(1, 3)]),
        (counting(3), lambda x: x[ints(2, 1, 0)]),
        (counting(3), lambda x: x[False]),
    ]
    for x, f in cases:
        y = f(x)
        g = torch.ones(y.shape, dtype=d64, requires_grad=True)
        gx, = grads(y, (x,), g, True)
        assert gx.shape == x.shape and gx.sum() == y.numel()
        dg, = grads(gx, (g,), torch.ones(x.shape, dtype=d64))
        assert dg.shape == y.shape
    x = counting(3)
    s = counting()
    y = x.index_put((ints(0, 1, 2),), s)
    gx, gs = grads(y, (x, s), torch.tensor([1.0, 2.0, 3.0], dtype=d64))
    assert gx.tolist() == [0, 0, 0] and gs.item() == 6


@block
def meta_targets_index_by_shape():
    x = torch.zeros(4, 5, 6, device="meta", requires_grad=True)
    meta_i = torch.zeros(2, dtype=torch.int64, device="meta")
    cpu_j = torch.tensor([[0], [1], [2]])
    assert x[meta_i, cpu_j].shape == (3, 2, 6)
    # PyTorch doesn't check CPU positions for a meta target (homura does).
    assert x[:, ints(5)].shape == (4, 1, 6)
    m = mask(True, False, True, True, False)
    assert x[:, m].shape == (4, 3, 6)
    v = torch.zeros(3, 6, device="meta", requires_grad=True)
    for accumulate in (False, True):
        y = torch.ops.aten.index_put(x, [None, m], v, accumulate)
        gx, gv = grads(y, (x, v), torch.zeros(4, 5, 6, device="meta"))
        assert gx.shape == (4, 5, 6) and gv.shape == (3, 6)
    meta_mask = torch.zeros(5, dtype=torch.bool, device="meta")
    raises(NotImplementedError, "nonzero", lambda: x[:, meta_mask])
    # PyTorch takes a meta mask in index_put's forward pass but can't compute the values'
    # gradient (homura rejects it in all three).
    y = torch.ops.aten.index_put(x, [None, meta_mask], v)
    raises(NotImplementedError, "nonzero", lambda: grads(y, (v,), torch.zeros(4, 5, 6, device="meta")))
    raises(RuntimeError, "Tensor.item() cannot be called on meta tensors",
           lambda: x[torch.zeros((), dtype=torch.bool, device="meta")])
    meta_scalar = torch.zeros((), dtype=torch.int64, device="meta")
    raises(RuntimeError, "Tensor.item() cannot be called on meta tensors", lambda: x[meta_scalar])
    x1 = torch.zeros(3, device="meta", requires_grad=True)
    s = torch.zeros((), device="meta", requires_grad=True)
    for accumulate in (False, True):
        y = (x1 * 1.0).index_put((meta_scalar,), s, accumulate)
        assert y.shape == (3,)
        g = torch.zeros(3, device="meta", requires_grad=True)
        gx, gs = grads(y, (x1, s), g, True)
        assert gx.shape == (3,) and gs.shape == ()
        dg, = grads(gs, (g,), torch.zeros((), device="meta"))
        assert dg.shape == (3,)
    huge = torch.zeros(1 << 20, 1 << 20, device="meta")
    column = torch.zeros(1 << 16, 1, dtype=torch.int64)
    row = torch.zeros(1 << 16, dtype=torch.int64)
    assert huge[column, row].shape == (1 << 16, 1 << 16)
    # PyTorch takes a CPU scalar as values for a meta target (homura: same device).
    assert x.index_put((ints(0),), torch.tensor(1.0)).shape == (4, 5, 6)


@block
def index_put_follows_the_in_place_rules():
    i = ints(0, 2)
    leaf = counting(3)
    raises(RuntimeError, "a leaf Variable that requires grad is being used in an in-place operation.",
           lambda: leaf.index_put_((i,), torch.tensor([1.0, 2.0], dtype=d64)))
    # PyTorch records a write into a view through CopySlices (homura rejects it, D7).
    view = (leaf * 1.0)[0:3]
    view.index_put_((i,), torch.tensor([1.0, 2.0], dtype=d64))
    with torch.no_grad():
        leaf.index_put_((i,), torch.tensor([10.0, 20.0], dtype=d64), True)
        leaf[0:2].index_put_((ints(1),), torch.tensor(-1.0, dtype=d64))
    assert leaf.tolist() == [10, -1, 22]
    buffer = torch.zeros(3, dtype=d64)
    v = counting(2)
    buffer.index_put_((i,), v)
    assert chain(buffer) == ["IndexPutBackward0", ACC]
    gv, = grads(buffer, (v,), torch.tensor([1.0, 2.0, 3.0], dtype=d64))
    assert gv.tolist() == [1, 3]
    x = counting(3)
    y = x * 2.0
    y.index_put_((ints(1),), v[0:1])
    assert chain(y)[:2] == ["IndexPutBackward0", "MulBackward0"]
    gx, gv = grads(y, (x, v), torch.ones(3, dtype=d64))
    assert gx.tolist() == [2, 0, 2] and gv.tolist() == [1, 0]


print("SUMMARY", passed, "blocks passed")
