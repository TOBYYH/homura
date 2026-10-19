"""Check the reference behavior that tests/autograd_function.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: node names, higher-order derivatives through
saved inputs, outputs and hidden intermediates, mixed and marked outputs, outputs built under
enable_grad, in-place writes into outputs, when a call is recorded, repeated and partially
requested inputs, meta and mixed dtypes, the number, shape, dtype and device of returned
gradients (and of root gradients), what forward may not do, marking and saving twice, missing
gradients, calls without differentiable outputs, unrecorded calls, saved tensors and lifetimes,
nested passes, exceptions and threads. Where homura differs on purpose, the reference behavior is
asserted and the difference named in a comment. The crate-internal tests (names of generic types,
Weak checks of saved slots) and the test of stale input views (PyTorch replays views on their
base's new history instead) have no counterpart here.
"""

import threading

import torch
from torch.autograd import Function

import reference

reference.check()
d64 = torch.float64
passed = 0


def block(f):
    global passed
    f()
    passed += 1
    print("PASS", f.__name__)
    return f


def raises(message, f, exc=RuntimeError):
    """Asserts that f raises exc whose first line contains message."""
    try:
        f()
    except exc as e:
        assert message in str(e).splitlines()[0], (str(e), message)
        return
    raise AssertionError(f"no {exc.__name__}: {message}")


def leaf(*values):
    return torch.tensor(values, dtype=d64, requires_grad=True)


def grad1(y, x, create_graph=False):
    return torch.autograd.grad(y, x, create_graph=create_graph, allow_unused=True)[0]


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * x * 2


class Product(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        return g * b, g * a


class SaveOut(Function):
    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g * y


def ret(fn):
    class Ret(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        def backward(ctx, g):
            return fn(g)

    return Ret


@block
def nodes_are_named_after_the_type():
    x = leaf(-1.0, 2.0)
    y = Square.apply(x)
    assert y.grad_fn.name() == "SquareBackward"
    assert len(y.grad_fn.next_functions) == 1
    assert type(y.grad_fn.next_functions[0][0]).__name__ == "AccumulateGrad"
    assert "grad_fn=<SquareBackward>" in repr(y)

    # Nested classes are named by their own name; homura leaves out generic arguments.
    class Outer:
        class Inner(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 3

            @staticmethod
            def backward(ctx, g):
                return g * 3

    y = Outer.Inner.apply(x)
    assert y.grad_fn.name() == "InnerBackward"
    y.sum().backward()
    assert x.grad.tolist() == [3.0, 3.0]


@block
def saved_inputs_give_second_derivatives():
    x = leaf(0.5, -1.5, 2.0)
    y = leaf(1.25, 0.75, -0.5)
    assert torch.autograd.gradcheck(Square.apply, (x,))
    assert torch.autograd.gradgradcheck(Square.apply, (x,))
    assert torch.autograd.gradcheck(Product.apply, (x, y))
    assert torch.autograd.gradgradcheck(Product.apply, (x, y))
    dx = grad1(Square.apply(x).sum(), x, True)
    assert dx.tolist() == [1.0, -3.0, 4.0]
    assert grad1(dx.sum(), x).tolist() == [2.0, 2.0, 2.0]


@block
def saved_outputs_give_second_derivatives():
    x = leaf(0.0, 1.0, -0.5)
    assert torch.autograd.gradgradcheck(SaveOut.apply, (x,))
    dx = grad1(SaveOut.apply(x).sum(), x, True)
    torch.testing.assert_close(grad1(dx.sum(), x), x.detach().exp())


def hidden_exp(recompute):
    class HiddenExp(Function):
        @staticmethod
        def forward(ctx, x):
            e = x.exp()
            ctx.save_for_backward(x if recompute else e)
            return e * 1

        @staticmethod
        def backward(ctx, g):
            (saved,) = ctx.saved_tensors
            return g * (saved.exp() if recompute else saved)

    return HiddenExp


@block
def hidden_intermediates_are_constants():
    x = leaf(0.0, 1.0)
    dx = grad1(hidden_exp(False).apply(x).sum(), x, True)
    torch.testing.assert_close(dx, x.detach().exp())
    assert not dx.requires_grad and dx.grad_fn is None
    dx = grad1(hidden_exp(True).apply(x).sum(), x, True)
    torch.testing.assert_close(grad1(dx.sum(), x), x.detach().exp())


@block
def saved_tensors_with_a_history_keep_it():
    seen = []
    w = leaf(3.0)

    class HiddenWithHistory(Function):
        @staticmethod
        def forward(ctx, x):
            with torch.enable_grad():
                h = x.detach() * w
            ctx.save_for_backward(h)
            return x * 2

        @staticmethod
        def backward(ctx, g):
            (h,) = ctx.saved_tensors
            seen.append(h.requires_grad)
            return g * h

    x = leaf(1.0)
    dx = grad1(HiddenWithHistory.apply(x).sum(), x, True)
    assert dx.tolist() == [3.0] and seen == [True]
    assert dx.grad_fn.name() == "MulBackward0"
    assert grad1(dx.sum(), w).tolist() == [1.0]


@block
def outputs_mixed_float_integer_and_marked():
    seen = []

    class Mixed(Function):
        @staticmethod
        def forward(ctx, x):
            marked = x * 3
            ctx.mark_non_differentiable(marked)
            return x * 2, marked, x.long(), x.exp()

        @staticmethod
        def backward(ctx, *grads):
            seen.append([g is not None and bool(g.any()) for g in grads])
            return grads[0] * 2

    x = leaf(1.0, 2.0)
    outputs = Mixed.apply(x)
    assert [t.requires_grad for t in outputs] == [True, False, False, True]
    assert outputs[1].grad_fn is None and outputs[2].grad_fn is None
    assert outputs[0].grad_fn is outputs[3].grad_fn
    outputs[0].sum().backward()
    assert x.grad.tolist() == [2.0, 2.0]
    # Difference: PyTorch materializes the missing gradients as zeros (I64 zeros for the integer
    # output); homura passes None.
    assert seen == [[True, False, False, False]]
    raises(
        "element 0 of tensors does not require grad and does not have a grad_fn",
        lambda: outputs[1].sum().backward(),
    )


@block
def saved_non_differentiable_outputs_have_no_history():
    seen = []

    class SaveNonDifferentiable(Function):
        @staticmethod
        def forward(ctx, x):
            y, marked, index = x * 2, x * 3, x.argmax()
            ctx.mark_non_differentiable(marked)
            ctx.save_for_backward(marked, index, y)
            return y, marked, index

        @staticmethod
        def backward(ctx, g, *_):
            saved = ctx.saved_tensors
            seen.extend(t.requires_grad for t in saved)
            seen.extend(t.grad_fn is not None for t in saved)
            return g * 2

    x = leaf(1.0, 2.0)
    outputs = SaveNonDifferentiable.apply(x)
    assert grad1(outputs[0].sum(), x, True).tolist() == [2.0, 2.0]
    assert seen == [False, False, True, False, False, True]


@block
def no_outputs_and_no_inputs():
    class Nothing(Function):
        @staticmethod
        def forward(ctx, *inputs):
            return torch.ones(2, dtype=d64) if not inputs else ()

        @staticmethod
        def backward(ctx, *grads):
            raise AssertionError("unreachable")

    assert Nothing.apply(leaf(1.0)) == ()
    y = Nothing.apply()
    assert not y.requires_grad and y.grad_fn is None


@block
def history_built_in_forward_is_replaced():
    stash = []

    class EnableGrad(Function):
        @staticmethod
        def forward(ctx, x):
            with torch.enable_grad():
                w = torch.ones(3, dtype=d64, requires_grad=True)
                y = (x.clone() * w).view(3, 1).transpose(0, 1)
            stash.append(y)
            return y

        @staticmethod
        def backward(ctx, g):
            return g.reshape(3) * 4

    x = leaf(1.0, 2.0, 3.0)
    y = EnableGrad.apply(x)
    assert y.grad_fn.name() == "EnableGradBackward"
    assert len(y.grad_fn.next_functions) == 1
    # Difference: PyTorch returns the tensor forward returned with its history replaced; homura
    # returns a new tensor with the same data and layout, leaving the returned one as it was.
    assert stash[0] is y
    y.sum().backward()
    assert x.grad.tolist() == [4.0, 4.0, 4.0]


@block
def in_place_operations_on_outputs():
    class ViewHidden(Function):
        @staticmethod
        def forward(ctx, x):
            return x.exp().view(2, 1)

        @staticmethod
        def backward(ctx, g):
            return g.view(2)

    x = leaf(1.0, 2.0)
    y = ViewHidden.apply(x)
    # Difference: PyTorch keeps the output a view and rejects writing into it; homura's output is
    # an ordinary tensor, the write is recorded after the node.
    raises("Output 0 of ViewHidden is a view and is being modified inplace", lambda: y.mul_(2))
    y = Square.apply(x)
    y.mul_(2)
    assert y.grad_fn.next_functions[0][0].name() == "SquareBackward"
    y = SaveOut.apply(x)
    with torch.no_grad():
        y.add_(1)
    raises(
        "[torch.DoubleTensor [2]], which is output 0 of SaveOut, is at version 1; expected "
        "version 0 instead.",
        lambda: y.sum().backward(),
    )


def flags(seen):
    class Flags(Function):
        @staticmethod
        def forward(ctx, *inputs):
            assert not torch.is_grad_enabled()
            seen.append(list(ctx.needs_input_grad))
            return sum(inputs, torch.zeros(2, dtype=d64))

        @staticmethod
        def backward(ctx, g):
            seen.append(list(ctx.needs_input_grad))
            return (g,) * len(ctx.needs_input_grad)

    return Flags


@block
def recording_depends_on_grad_mode_and_inputs():
    seen = []
    x = leaf(1.0, 2.0)
    c = torch.tensor([1.0, 1.0], dtype=d64)
    with torch.no_grad():
        y = flags(seen).apply(x, c)
    assert not y.requires_grad and y.grad_fn is None
    assert seen == [[True, False]]
    y = Square.apply(c)
    assert not y.requires_grad and y.grad_fn is None
    # A view of a leaf created in no-grad mode: it requires grad through its base, but has no
    # edge, so the gradient is dropped.
    with torch.no_grad():
        v = x.view(2)
    y = Square.apply(v)
    assert y.grad_fn.next_functions[0][0] is None
    y.sum().backward()
    assert x.grad is None


@block
def repeated_and_partially_requested_inputs():
    x = leaf(3.0)
    y = Product.apply(x, x)
    (a0, _), (a1, _) = y.grad_fn.next_functions
    assert a0 is a1
    y.sum().backward()
    assert x.grad.tolist() == [6.0]
    a, b = leaf(2.0), torch.tensor([5.0], dtype=d64)
    Product.apply(a, b).sum().backward()
    assert a.grad.tolist() == [5.0] and b.grad is None
    seen = []
    a, b = leaf(1.0, 2.0), leaf(3.0, 4.0)
    y = flags(seen).apply(a, b)
    assert grad1(y.sum(), a).tolist() == [1.0, 1.0]
    assert seen == [[True, True], [True, True]]


@block
def meta_inputs_and_mixed_dtypes():
    x = torch.ones(2, device="meta", requires_grad=True)
    y = SaveOut.apply(x)
    assert y.device.type == "meta"
    g = grad1(y.sum(), x)
    assert g.device.type == "meta" and g.shape == (2,)
    seen = []

    class ToHalf(Function):
        @staticmethod
        def forward(ctx, x):
            return x.half()

        @staticmethod
        def backward(ctx, g):
            seen.append(g.dtype)
            return g

    x = torch.ones(2, requires_grad=True)
    ToHalf.apply(x).sum().backward()
    assert seen == [torch.float16] and x.grad.dtype == torch.float32


@block
def returned_gradient_counts():
    x = leaf(1.0, 2.0)
    for f in (lambda g: g, lambda g: (g, None), lambda g: (g, None, None)):
        assert grad1(ret(f).apply(x).sum(), x).tolist() == [1.0, 1.0]
    raises(
        "function RetBackward returned an incorrect number of gradients (expected 1, got 3)",
        lambda: grad1(ret(lambda g: (g, None, g)).apply(x).sum(), x),
    )
    raises(
        "function RetBackward returned an incorrect number of gradients (expected 1, got 0)",
        lambda: grad1(ret(lambda g: ()).apply(x).sum(), x),
    )

    class Short(Function):
        @staticmethod
        def forward(ctx, a, b):
            return a * b

        @staticmethod
        def backward(ctx, g):
            return (g,)

    a, b = leaf(1.0), torch.tensor([2.0], dtype=d64)
    raises(
        "function ShortBackward returned an incorrect number of gradients (expected 2, got 1)",
        lambda: Short.apply(a, b).sum().backward(),
    )


@block
def returned_gradients_are_reduced_and_converted():
    x = torch.ones(2, 3, dtype=d64, requires_grad=True)
    g = grad1(ret(lambda _: torch.ones(4, 2, 3)).apply(x).sum(), x)
    assert g.shape == (2, 3) and g.tolist() == [[4.0] * 3] * 2
    for shape in ((3,), (1, 3), (5,), ()):
        raises(
            "Function RetBackward returned an invalid gradient at index 0 - got [",
            lambda shape=shape: grad1(ret(lambda _: torch.ones(shape)).apply(x).sum(), x),
        )
    raises(
        "Function RetBackward returned an invalid gradient at index 0 - got [3] but expected "
        "shape compatible with [2, 3]",
        lambda: grad1(ret(lambda _: torch.ones(3)).apply(x).sum(), x),
    )
    e = torch.ones(0, dtype=d64, requires_grad=True)
    assert grad1(ret(lambda _: torch.ones(3, 0)).apply(e).sum(), e).shape == (0,)
    r = torch.ones(1, 3, dtype=d64, requires_grad=True)
    g = grad1(ret(lambda _: torch.arange(6.0).reshape(2, 3)).apply(r).sum(), r)
    assert g.tolist() == [[3.0, 5.0, 7.0]]
    x = torch.ones(2, requires_grad=True)
    for dtype in (d64, torch.int64, torch.bool):
        g = grad1(ret(lambda g, dtype=dtype: g.to(dtype)).apply(x).sum(), x)
        assert g.dtype == torch.float32 and g.tolist() == [1.0, 1.0]


@block
def reduced_gradients_stay_differentiable():
    class Larger(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return (g * x * 2).expand(4, 2, 3)

    x = torch.ones(2, 3, dtype=d64, requires_grad=True)
    dx = grad1(Larger.apply(x).sum(), x, True)
    assert dx.tolist() == [[8.0] * 3] * 2
    assert grad1(dx.sum(), x).tolist() == [[8.0] * 3] * 2


@block
def returned_gradients_on_another_device():
    for shape in ((), (4,), (2, 3)):
        x = torch.ones((), device="meta", requires_grad=True)
        g = grad1(ret(lambda _, shape=shape: torch.ones(shape)).apply(x), x)
        assert g.device.type == "meta" and g.shape == ()
    for shape in ((), (4,)):
        x = torch.ones((), requires_grad=True)
        y = ret(lambda _, shape=shape: torch.ones(shape, device="meta")).apply(x)
        raises("Cannot copy out of meta tensor", lambda: grad1(y, x), NotImplementedError)
    x = torch.ones(2, 3, device="meta", requires_grad=True)
    raises(
        "Function RetBackward returned an invalid gradient at index 0 - expected device meta but "
        "got cpu",
        lambda: grad1(ret(lambda _: torch.ones(4, 2, 3)).apply(x).sum(), x),
    )
    x = torch.ones(2, 3, requires_grad=True)
    raises(
        "Function RetBackward returned an invalid gradient at index 0 - expected device cpu but "
        "got meta",
        lambda: grad1(ret(lambda _: torch.ones(2, 3, device="meta")).apply(x).sum(), x),
    )
    raises(
        "returned an invalid gradient at index 0 - got [5]",
        lambda: grad1(ret(lambda _: torch.ones(5, device="meta")).apply(x).sum(), x),
    )


@block
def root_gradients_on_another_device():
    x = torch.ones(2, requires_grad=True)
    raises(
        "invalid gradient at index 0 - expected device cpu but got meta",
        lambda: (x * 2).backward(torch.ones(2, device="meta")),
    )
    x = torch.ones(2, device="meta", requires_grad=True)
    raises(
        "invalid gradient at index 0 - expected device meta but got cpu",
        lambda: torch.autograd.grad(x * 2, x, torch.ones(2)),
    )
    x = torch.ones((), device="meta", requires_grad=True)
    assert torch.autograd.grad(x * 2, x, torch.ones(()))[0].device.type == "meta"


@block
def forward_must_not_modify_inputs():
    class Modify(Function):
        @staticmethod
        def forward(ctx, x):
            x.add_(1)
            return x * 2

        @staticmethod
        def backward(ctx, g):
            return g * 2

    # Difference: PyTorch accepts the write without mark_dirty (the gradients may be wrong);
    # homura rejects it after forward returns, in every grad mode.
    x = leaf(1.0, 2.0)
    for t in (x, x * 1, torch.tensor([1.0, 2.0], dtype=d64)):
        Modify.apply(t)
        assert t.tolist() == [2.0, 3.0]
        with torch.no_grad():
            t.sub_(1)
    with torch.no_grad():
        Modify.apply(x)


def alias(fn):
    class Alias(Function):
        @staticmethod
        def forward(ctx, x):
            return fn(x)

        @staticmethod
        def backward(ctx, *grads):
            return grads[0]

    return Alias


@block
def outputs_must_not_share_memory():
    # Difference: PyTorch accepts outputs that share memory with an input or another output;
    # homura rejects them.
    x = leaf(1.0, 2.0, 3.0, 4.0)
    for fn in (
        lambda x: x,
        lambda x: x.view(2, 2),
        lambda x: (x * 2, x.narrow(0, 1, 2)),
        lambda x: (lambda y: (y, y))(x * 2),
        lambda x: (x * 2).chunk(2),
    ):
        outputs = alias(fn).apply(x)
        assert all(t.requires_grad for t in (outputs if isinstance(outputs, tuple) else (outputs,)))


def mark(fn):
    class Mark(Function):
        @staticmethod
        def forward(ctx, x):
            outputs = (x * 2, x * 3)
            fn(ctx, x, outputs)
            return outputs

        @staticmethod
        def backward(ctx, *grads):
            return sum(g for g in grads if g is not None)

    return Mark


@block
def marking_non_differentiable_outputs():
    x = leaf(1.0, 2.0)

    def flags_of(fn):
        return [t.requires_grad for t in mark(fn).apply(x)]

    def twice(first, second):
        def fn(ctx, x, o):
            ctx.mark_non_differentiable(*first(x, o))
            ctx.mark_non_differentiable(*second(x, o))

        return fn

    assert flags_of(twice(lambda x, o: (o[0],), lambda x, o: (o[1],))) == [True, False]
    assert flags_of(lambda ctx, x, o: ctx.mark_non_differentiable(o[0], o[0])) == [False, True]
    assert flags_of(twice(lambda x, o: (o[0],), lambda x, o: ())) == [True, True]
    assert flags_of(twice(lambda x, o: (x,), lambda x, o: (o[1],))) == [True, False]
    # Difference: PyTorch ignores marked tensors that aren't outputs; homura rejects them.
    for fn in (
        lambda ctx, x, o: ctx.mark_non_differentiable(x),
        lambda ctx, x, o: ctx.mark_non_differentiable(x * 1),
    ):
        assert flags_of(fn) == [True, True]
        with torch.no_grad():
            mark(fn).apply(x)


@block
def saving_again_replaces():
    class SaveTwice(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x, x)
            ctx.save_for_backward(x * 10)
            return x * 2

        @staticmethod
        def backward(ctx, g):
            saved = ctx.saved_tensors
            assert len(saved) == 1
            return g * saved[0]

    x = leaf(1.0)
    SaveTwice.apply(x).sum().backward()
    assert x.grad.tolist() == [10.0]


class NoneGrad(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 1

    @staticmethod
    def backward(ctx, g):
        return None


@block
def backward_is_skipped_when_no_gradient_arrives():
    calls = []

    class Counted(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, g):
            calls.append(g)
            (x,) = ctx.saved_tensors
            return g * x * 2

    x = leaf(1.0, 2.0)
    y = Counted.apply(x)
    z = NoneGrad.apply(y)
    # Difference: PyTorch calls backward with materialized zeros and returns a zero gradient;
    # homura skips the call and returns None.
    assert grad1(z.sum(), x).tolist() == [0.0, 0.0]
    assert len(calls) == 1 and calls[0].tolist() == [0.0, 0.0]
    raises("Trying to backward through the graph a second time", lambda: (y * 1).sum().backward())


@block
def calls_without_differentiable_outputs_record_nothing():
    class NoDifferentiable(Function):
        @staticmethod
        def forward(ctx, x):
            marked = x * 2
            ctx.mark_non_differentiable(marked)
            ctx.save_for_backward(x, marked)
            return x.long(), x > 0, marked

        @staticmethod
        def backward(ctx, *grads):
            raise AssertionError("unreachable")

    outputs = NoDifferentiable.apply(leaf(1.0, -2.0))
    assert all(not t.requires_grad and t.grad_fn is None for t in outputs)


@block
def unrecorded_calls_return_new_tensors():
    values = torch.arange(6, dtype=d64).reshape(2, 3)
    x = values.clone().requires_grad_(True)
    c = values.clone()
    for t, outer_no_grad, enable_grad in ((x, True, False), (c, False, False), (x, True, True)):
        stash = []

        class Stash(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.set_grad_enabled(enable_grad):
                    w = torch.ones(2, 3, dtype=d64, requires_grad=enable_grad)
                    y = (x.clone() * w).transpose(0, 1)
                stash.append(y)
                return y

            @staticmethod
            def backward(ctx, g):
                return g.transpose(0, 1)

        with torch.set_grad_enabled(not outer_no_grad):
            y = Stash.apply(t)
        assert y.shape == (3, 2) and y.stride() == (1, 3)
        # Difference: PyTorch returns the tensor forward returned, with any history it was given
        # in forward (an output that requires grad after an unrecorded call); homura returns a
        # new tensor without history and leaves the returned one as it was.
        assert y is stash[0]
        assert y.requires_grad == enable_grad and (y.grad_fn is not None) == enable_grad


@block
def saved_tensors_are_checked_and_freed():
    x = leaf(1.0, 2.0)
    y = Square.apply(x)
    with torch.no_grad():
        x.add_(1)
    raises("[torch.DoubleTensor [2]] is at version 1; expected version 0 instead.", lambda: y.sum().backward())
    x = leaf(1.0, 2.0)
    s = Square.apply(x).sum()
    s.backward(retain_graph=True)
    s.backward(retain_graph=True)
    assert x.grad.tolist() == [4.0, 8.0]
    s.backward()
    raises("Trying to backward through the graph a second time", s.backward)


@block
def backward_can_run_nested_passes():
    class Nested(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.sin()

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            with torch.enable_grad():
                t = x.detach().requires_grad_(True)
                inner = torch.autograd.grad(t.sin().sum(), t)[0]
            return g * inner

    x = leaf(0.5, -1.0)
    torch.testing.assert_close(grad1(Nested.apply(x).sum(), x), x.detach().cos())


@block
def panics_leave_grad_mode_and_the_graph_usable():
    armed = [True]

    def panicky(in_forward):
        class Panicky(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                if in_forward:
                    raise RuntimeError("forward failed")
                return x * 2

            @staticmethod
            def backward(ctx, g):
                if armed[0]:
                    armed[0] = False
                    raise RuntimeError("backward failed")
                assert len(ctx.saved_tensors) == 1
                return g * 2

        return Panicky

    x = leaf(1.0)
    raises("forward failed", lambda: panicky(True).apply(x))
    assert torch.is_grad_enabled()
    s = panicky(False).apply(x).sum()
    raises("backward failed", lambda: s.backward(retain_graph=True))
    assert torch.is_grad_enabled()
    s.backward(retain_graph=True)
    assert x.grad.tolist() == [2.0]


@block
def independent_calls_run_concurrently():
    results = {}

    def run(i):
        x = leaf(float(i), 1.0)
        Square.apply(x).sum().backward()
        results[i] = x.grad.tolist()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {i: [2.0 * i, 2.0] for i in range(4)}


def concurrent_passes(retain_graph):
    entered, released = threading.Barrier(2), threading.Barrier(2)

    class Shared(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, g):
            if threading.current_thread().name == "slow":
                entered.wait(10)
                released.wait(10)
            (x,) = ctx.saved_tensors
            return g * x * 2

    x = leaf(1.0, 2.0)
    y = Shared.apply(x)
    results = {}

    def run(name):
        try:
            results[name] = torch.autograd.grad(y.sum(), x, retain_graph=retain_graph)[0].tolist()
        except RuntimeError as e:
            results[name] = str(e)

    slow = threading.Thread(target=run, args=("slow",), name="slow")
    slow.start()
    entered.wait(10)
    run("fast")
    released.wait(10)
    slow.join(10)
    return results, x, y


@block
def shared_graphs_run_concurrently_with_retain_graph():
    results, x, y = concurrent_passes(True)
    assert results == {"slow": [2.0, 4.0], "fast": [2.0, 4.0]}
    y.sum().backward()
    assert x.grad.tolist() == [2.0, 4.0]
    raises("Trying to backward through the graph a second time", lambda: y.sum().backward())


@block
def shared_graphs_without_retain_graph_free_the_saved_tensors():
    results, _, _ = concurrent_passes(False)
    assert results["fast"] == [2.0, 4.0]
    assert "Trying to backward through the graph a second time" in results["slow"]


print(f"{passed} blocks passed")
