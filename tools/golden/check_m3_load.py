"""Check the reference behavior that tests/nn_load.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: `load_state_dict(sd, assign=True)` and models
built on the meta device. Where homura differs on purpose, the block asserts PyTorch's behavior
and says what homura does instead: PyTorch replaces the parameters with new ones (homura keeps
them and gives them the entry's data), and writes the entries that match before it reports
errors (homura checks every entry first).
"""

import torch
import torch.nn as nn

import reference

reference.check()
passed = 0


def block(f):
    global passed
    f()
    passed += 1


def message(f):
    try:
        f()
    except RuntimeError as e:
        return str(e)
    raise AssertionError("expected an error")


@block
def assign_takes_the_data_of_the_entries():
    m = nn.Linear(3, 2)
    m.weight.grad = torch.ones(2, 3)
    m.bias.requires_grad_(False)
    # A non-contiguous F64 entry: the parameter takes its dtype, layout and storage.
    weight = torch.arange(6, dtype=torch.float64).reshape(3, 2).t()
    bias = torch.zeros(2, dtype=torch.float64)
    report = m.load_state_dict({"weight": weight, "bias": bias}, assign=True)
    assert report.missing_keys == [] and report.unexpected_keys == []
    w = m.weight
    assert w.dtype == torch.float64 and w.shape == (2, 3) and w.stride() == (1, 2)
    assert w.data_ptr() == weight.data_ptr() and w.is_leaf
    # requires_grad is the module's, the gradient is gone (a new parameter in PyTorch; homura
    # keeps the parameter and clears its `.grad`).
    assert w.requires_grad and w.grad is None and not m.bias.requires_grad
    # The parameter and the entry share the storage: writing one changes the other.
    with torch.no_grad():
        w.add_(1)
    assert weight[0, 0].item() == 1.0 and weight._version == 1
    # A buffer takes the entry too (in PyTorch the entry itself; in homura its data).
    bn = nn.BatchNorm2d(2)
    mean = torch.full((2,), 3.0)
    bn.load_state_dict({**bn.state_dict(), "running_mean": mean}, assign=True)
    assert bn.running_mean is mean
    with torch.no_grad():
        bn.running_mean.add_(1)
    assert mean.tolist() == [4.0, 4.0]


@block
def assign_combines_with_strict():
    sd = {"weight": torch.ones(2, 3), "extra": torch.ones(1)}
    m = nn.Linear(3, 2)
    text = message(lambda: m.load_state_dict(sd, assign=True))
    assert 'Missing key(s) in state_dict: "bias"' in text, text
    assert 'Unexpected key(s) in state_dict: "extra"' in text, text
    # PyTorch has assigned the weight anyway; homura writes nothing when it fails.
    assert m.weight.tolist() == [[1.0] * 3] * 2
    m = nn.Linear(3, 2)
    bias = m.bias
    report = m.load_state_dict(sd, strict=False, assign=True)
    assert report.missing_keys == ["bias"] and report.unexpected_keys == ["extra"]
    assert m.weight.tolist() == [[1.0] * 3] * 2 and m.bias is bias


@block
def assign_checks_the_entries():
    m = nn.Linear(3, 2)
    text = message(
        lambda: m.load_state_dict(
            {"weight": torch.zeros(2, 3, dtype=torch.int64), "bias": torch.zeros(2)}, assign=True
        )
    )
    assert "Only Tensors of floating point and complex dtype can require gradients" in text, text
    assert 'While copying the parameter named "weight"' in text, text
    # A parameter that doesn't require grad takes an integer entry.
    m = nn.Linear(3, 2)
    m.weight.requires_grad_(False)
    m.load_state_dict(
        {"weight": torch.zeros(2, 3, dtype=torch.int64), "bias": torch.zeros(2)}, assign=True
    )
    assert m.weight.dtype == torch.int64
    text = message(
        lambda: nn.Linear(3, 2).load_state_dict(
            {"weight": torch.zeros(3, 3), "bias": torch.zeros(2)}, assign=True
        )
    )
    assert (
        "size mismatch for weight: copying a param with shape torch.Size([3, 3]) from checkpoint, "
        "the shape in current model is torch.Size([2, 3])." in text
    ), text


def layers():
    return [
        lambda: nn.Linear(4, 3),
        lambda: nn.Conv2d(2, 3, 3, padding=1),
        lambda: nn.BatchNorm2d(2),
        lambda: nn.LayerNorm(4),
        lambda: nn.RMSNorm(4),
        lambda: nn.Embedding(5, 4),
    ]


def inputs():
    torch.manual_seed(1)
    return [
        torch.randn(2, 4),
        torch.randn(1, 2, 4, 4),
        torch.randn(2, 2, 3, 3),
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.tensor([0, 4, 2]),
    ]


@block
def models_built_on_meta_load_by_assignment():
    for make, x in zip(layers(), inputs(), strict=True):
        with torch.device("meta"):
            empty = make()
        assert all(t.device.type == "meta" for t in [*empty.parameters(), *empty.buffers()])
        trained = make()
        copied = make()
        copied.load_state_dict(trained.state_dict())
        empty.load_state_dict(trained.state_dict(), assign=True)
        assert all(t.device.type == "cpu" for t in [*empty.parameters(), *empty.buffers()])
        for m in (trained, copied, empty):
            m.eval()
        assert torch.equal(empty(x), copied(x))
    # Without strict, what isn't loaded stays on the meta device.
    with torch.device("meta"):
        m = nn.Linear(3, 2)
    report = m.load_state_dict({"weight": torch.ones(2, 3)}, strict=False, assign=True)
    assert report.missing_keys == ["bias"]
    assert m.weight.device.type == "cpu" and m.bias.device.type == "meta"


@block
def entries_on_the_meta_device_are_assigned():
    m = nn.Linear(3, 2)
    m.load_state_dict(
        {"weight": torch.zeros(2, 3, device="meta"), "bias": torch.zeros(2, device="meta")},
        assign=True,
    )
    assert m.weight.device.type == "meta" and m.bias.device.type == "meta"
    # Copying from them fails.
    text = message(
        lambda: nn.Linear(3, 2).load_state_dict(
            {"weight": torch.zeros(2, 3, device="meta"), "bias": torch.zeros(2)}
        )
    )
    assert "Cannot copy out of meta tensor; no data!" in text, text


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = self.a


@block
def shared_parameters_take_the_last_entry():
    t = Tied()
    assert list(t.state_dict()) == ["a.weight", "a.bias", "b.weight", "b.bias"]
    t.load_state_dict(
        {
            "a.weight": torch.ones(2, 2),
            "a.bias": torch.zeros(2),
            "b.weight": torch.full((2, 2), 2.0),
            "b.bias": torch.zeros(2),
        },
        assign=True,
    )
    assert t.a is t.b and t.a.weight[0, 0].item() == 2.0


@block
def graphs_of_the_entries_still_run():
    # An entry that requires grad, saved by its own graph: assigning it leaves that graph alone.
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    y = (source * source).sum()
    m = nn.Module()
    m.w = nn.Parameter(torch.zeros(2))
    m.load_state_dict({"w": source}, assign=True)
    y.backward()
    assert source.grad.tolist() == [2.0, 4.0]
    assert m.w.data_ptr() == source.data_ptr() and m.w.is_leaf


@block
def entries_are_read_before_any_is_assigned():
    # The entry of `b` is the parameter `a` itself: `b` gets what `a` held before the load.
    m = nn.Module()
    m.a = nn.Parameter(torch.tensor([1.0, 2.0]))
    m.b = nn.Parameter(torch.zeros(2))
    m.load_state_dict({"a": torch.tensor([5.0, 6.0]), "b": m.a}, assign=True)
    assert m.a.tolist() == [5.0, 6.0] and m.b.tolist() == [1.0, 2.0]


print(f"check_m3_load: {passed} blocks passed")
