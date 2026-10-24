"""Check the reference behavior that tests/nn_load.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: `load_state_dict(sd, assign=True)`, models
built on the meta device, the versions of modules in `state_dict._metadata`, the prefixes of keys
and old `BatchNorm2d` state dicts without `num_batches_tracked`. Where homura differs on purpose, the block asserts PyTorch's behavior
and says what homura does instead: PyTorch replaces the parameters with new ones (homura keeps
them and gives them the entry's data), and writes the entries that match before it reports
errors (homura checks every entry first).
"""

import collections
import warnings

import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.blocks = nn.ModuleList([nn.BatchNorm2d(2), nn.BatchNorm2d(2)])
        self.empty = nn.ModuleList()
        self.heads = nn.ModuleDict({"a": nn.Linear(2, 2)})
        self.register_module("skip", None)
        # The same module under a second name.
        self.tied = self.bn


def versions(sd):
    return [(path, metadata["version"]) for path, metadata in sd._metadata.items()]


NET_VERSIONS = [
    ("", 1),
    ("conv", 1),
    ("bn", 2),
    ("blocks", 1),
    ("blocks.0", 2),
    ("blocks.1", 2),
    ("empty", 1),
    ("heads", 1),
    ("heads.a", 1),
    ("tied", 2),
]


@block
def state_dicts_record_module_versions():
    sd = Net().state_dict()
    # Every module by its path, before its submodules; a shared module under each name, a
    # `None` one not at all. Only the version is recorded.
    assert versions(sd) == NET_VERSIONS
    assert all(list(metadata) == ["version"] for metadata in sd._metadata.values())
    assert versions(nn.Sequential(nn.Linear(2, 2), nn.BatchNorm2d(2)).state_dict()) == [
        ("", 1),
        ("0", 1),
        ("1", 2),
    ]
    assert versions(nn.BatchNorm2d(2).state_dict()) == [("", 2)]
    # Only the batch normalization layers have another version than 1.
    for module in [nn.Linear, nn.Conv2d, nn.LayerNorm, nn.RMSNorm, nn.Embedding, nn.Dropout,
                   nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.MaxPool2d,
                   nn.AdaptiveAvgPool2d, nn.Flatten, nn.Sequential, nn.ModuleList, nn.ModuleDict]:
        assert module._version == 1, module
    assert nn.BatchNorm2d._version == 2


def prefixed(keys, paths, prefix):
    sd = collections.OrderedDict((key, torch.zeros(1)) for key in keys)
    sd._metadata = collections.OrderedDict((path, {"version": v}) for path, v in paths)
    consume_prefix_in_state_dict_if_present(sd, prefix)
    return list(sd), versions(sd)


@block
def prefixes_move_keys_and_versions():
    # Renamed entries go last; the module named by the prefix becomes the root.
    assert prefixed(
        ["module.a", "b", "module.c.d"], [("", 1), ("module", 3), ("module.c", 2), ("x", 1)],
        "module.",
    ) == (["b", "a", "c.d"], [("", 3), ("x", 1), ("c", 2)])
    # A new key that exists replaces that entry in its position.
    assert prefixed(["a", "module.a"], [("", 7), ("module", 3)], "module.") == (["a"], [("", 3)])
    # The prefix is removed once.
    assert prefixed(["module.module.a"], [("module.module", 3)], "module.") == (
        ["module.a"],
        [("module", 3)],
    )
    # PyTorch compares the paths with the prefix without any ".": for "a.b." it renames the
    # version of "ab" (homura that of "a.b").
    assert prefixed(
        ["a.b.w", "a.bw"], [("", 1), ("a", 1), ("a.b", 2), ("ab", 4), ("a.b.c", 3)], "a.b."
    ) == (["a.bw", "w"], [("", 4), ("a", 1), ("a.b", 2), ("c", 3)])
    # Adding a prefix: `state_dict(prefix=...)`.
    sd = Net().state_dict(prefix="model.")
    assert list(sd)[:2] == ["model.conv.weight", "model.conv.bias"]
    assert versions(sd) == [("model", 1)] + [("model." + p, v) for p, v in NET_VERSIONS[1:]]
    # The state dicts of submodules, which homura's `sub` takes out of the whole one.
    net = Net()
    child = net.blocks[1].state_dict()
    assert list(child) == ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
    assert versions(child) == [("", 2)]
    heads = net.heads.state_dict()
    assert list(heads) == ["a.weight", "a.bias"] and versions(heads) == [("", 1), ("a", 1)]


def load_old(version, device, assign, strict=True):
    """Loads a state dict without the count into a `BatchNorm2d` whose count is 5 (on the CPU)."""
    with torch.device(device):
        bn = nn.BatchNorm2d(2)
    if device == "cpu":
        bn.num_batches_tracked.fill_(5)
    count = bn.num_batches_tracked
    sd = collections.OrderedDict(
        (k, v) for k, v in nn.BatchNorm2d(2).state_dict().items() if k != "num_batches_tracked"
    )
    if version != "none":
        sd._metadata = {"": {"version": version}} if version is not None else {"x": {"version": 1}}
    try:
        # Copying into the meta device warns that it does nothing; homura doesn't warn.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = bn.load_state_dict(sd, strict=strict, assign=assign)
    except RuntimeError as e:
        return str(e)
    n = bn.num_batches_tracked
    value = None if n.device.type == "meta" else n.item()
    return report.missing_keys, n.device.type, n.dtype, value, n is count


@block
def old_batch_norm_state_dicts_keep_the_count():
    missing = 'Missing key(s) in state_dict: "num_batches_tracked".'
    # Without versions, without one for the module, or of version 1: the count stays; on the
    # meta device, copying leaves it there and assigning gives it an I64 0 on the CPU.
    for version in ["none", None, 1]:
        for assign in [False, True]:
            assert load_old(version, "cpu", assign) == ([], "cpu", torch.int64, 5, True)
        assert load_old(version, "meta", False) == ([], "meta", torch.int64, None, True)
        assert load_old(version, "meta", True) == ([], "cpu", torch.int64, 0, False)
    # Version 2 has the count: it is missing.
    for device in ["cpu", "meta"]:
        for assign in [False, True]:
            assert missing in load_old(2, device, assign)
    assert load_old(2, "cpu", False, strict=False) == (
        ["num_batches_tracked"], "cpu", torch.int64, 5, True
    )
    # An old state dict that has the count loads it.
    sd = nn.BatchNorm2d(2).state_dict()
    sd["num_batches_tracked"].fill_(3)
    sd._metadata = {"": {"version": 1}}
    for assign in [False, True]:
        bn = nn.BatchNorm2d(2)
        bn.load_state_dict(sd, assign=assign)
        assert bn.num_batches_tracked.item() == 3
    # The version of the module's own path counts, in a whole model.
    sd = Net().state_dict()
    del sd["blocks.1.num_batches_tracked"]
    text = message(lambda: Net().load_state_dict(sd))
    assert 'Missing key(s) in state_dict: "blocks.1.num_batches_tracked".' in text, text
    sd._metadata["blocks.1"]["version"] = 1
    assert Net().load_state_dict(sd).missing_keys == []
    del sd._metadata["blocks.1"]
    assert Net().load_state_dict(sd).missing_keys == []
    # Without running statistics there is no count to keep.
    plain = nn.BatchNorm2d(2, track_running_stats=False)
    assert plain.num_batches_tracked is None
    assert plain.load_state_dict({"weight": torch.ones(2), "bias": torch.zeros(2)}).missing_keys == []


print(f"check_m3_load: {passed} blocks passed")
