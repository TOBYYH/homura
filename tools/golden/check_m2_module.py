"""Check the reference behavior that tests/nn_module.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order. The Rust tests build the same module trees by
hand (`Net`, `Tied`, ...) and assert the key lists asserted here. Where homura deliberately
differs (loading and `to` check everything first, old graphs after `to`, `to` replacing `.grad`
by a converted copy and rejecting one with a history, the spelling of shapes and dtypes in
messages), the PyTorch behavior is asserted and the difference is noted. The tests of method
resolution on wrapped tensors and of hyperparameters in containers have no PyTorch counterpart.
"""

import torch
from torch import nn

import reference

reference.check()


def message(f, kind=Exception):
    try:
        f()
    except kind as e:
        # KeyError quotes its message.
        return e.args[0] if isinstance(e, KeyError) else str(e)
    raise AssertionError("expected an error")


def keys(pairs):
    return [k for k, _ in pairs]


class Stats(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(3))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.stats = Stats()


class Net(nn.Module):
    """Registers its members in the order of the fields of the Rust `Net`."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(4, 3))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.relu = nn.ReLU()
        self.head = nn.Linear(3, 2, bias=False)
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("cache", torch.zeros(2), persistent=False)
        # Inserted in key order: homura's BTreeMap sorts, nn.ModuleDict keeps insertion order.
        self.table = nn.ModuleDict({"a": nn.Linear(3, 1), "b": nn.Linear(3, 1)})
        self.empty = nn.ModuleList()
        # Hyperparameters are plain attributes, not members.
        self.sizes = [1, 2]
        self.name = "net"


BLOCK = ["lin.weight", "lin.bias", "stats.running_mean", "stats.num_batches_tracked"]
NET_STATE = (
    ["embed", "scale"]
    + [f"blocks.{i}.{k}" for i in range(2) for k in BLOCK]
    + ["head.weight", "table.a.weight", "table.a.bias", "table.b.weight", "table.b.bias"]
)

# state_dict_and_traversals_follow_pytorch_order
net = Net()
assert list(net.state_dict()) == NET_STATE
assert keys(net.named_parameters()) == [
    "embed",
    "blocks.0.lin.weight",
    "blocks.0.lin.bias",
    "blocks.1.lin.weight",
    "blocks.1.lin.bias",
    "head.weight",
    "table.a.weight",
    "table.a.bias",
    "table.b.weight",
    "table.b.bias",
]
assert keys(net.named_buffers()) == [
    "scale",
    "cache",
    "blocks.0.stats.running_mean",
    "blocks.0.stats.num_batches_tracked",
    "blocks.1.stats.running_mean",
    "blocks.1.stats.num_batches_tracked",
]
assert keys(net.named_modules()) == [
    "",
    "blocks",
    "blocks.0",
    "blocks.0.lin",
    "blocks.0.stats",
    "blocks.1",
    "blocks.1.lin",
    "blocks.1.stats",
    "relu",
    "head",
    "table",
    "table.a",
    "table.b",
    "empty",
]
assert keys(net.named_children()) == ["blocks", "relu", "head", "table", "empty"]
assert len(list(net.children())) == 5


# shared_modules_and_parameters_are_listed_once_but_saved_under_every_key
class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = self.a
        self.c = nn.Linear(2, 2, bias=False)
        self.c.weight = self.a.weight


tied = Tied()
assert list(tied.state_dict()) == ["a.weight", "b.weight", "c.weight"]
assert keys(tied.named_parameters()) == ["a.weight"]
assert keys(tied.named_modules()) == ["", "a", "c"]
assert keys(tied.named_children()) == ["a", "c"]
assert len(list(tied.parameters())) == 1


# modules_are_identified_by_arc_not_by_address: Python objects are always distinct, so a wrapper
# and its first submodule (same address in Rust) are two modules, and two ReLU are two modules.
class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()


assert keys(Wrapper().named_modules()) == ["", "inner"]
pair = nn.Module()
pair.r1, pair.r2 = nn.ReLU(), nn.ReLU()
assert keys(pair.named_modules()) == ["", "r1", "r2"]
pair.r2 = pair.r1
assert keys(pair.named_modules()) == ["", "r1"]

# invalid_names_panic_with_pytorch_messages
m = nn.Module()
assert message(lambda: m.add_module("a.b", nn.ReLU()), KeyError) == (
    'module name can\'t contain ".", got: a.b'
)
assert message(lambda: m.add_module("", nn.ReLU()), KeyError) == (
    'module name can\'t be empty string ""'
)
assert message(lambda: nn.ModuleDict({"a.b": nn.ReLU()}), KeyError) == (
    'module name can\'t contain ".", got: a.b'
)
zero = lambda: nn.Parameter(torch.zeros(1))  # noqa: E731
assert message(lambda: m.register_parameter("a.b", zero()), KeyError) == (
    'parameter name can\'t contain "."'
)
assert message(lambda: m.register_parameter("", zero()), KeyError) == (
    'parameter name can\'t be empty string ""'
)
assert message(lambda: m.register_buffer("a.b", torch.zeros(1)), KeyError) == (
    'buffer name can\'t contain "."'
)
assert message(lambda: m.register_buffer("", torch.zeros(1)), KeyError) == (
    'buffer name can\'t be empty string ""'
)
m.register_buffer("x", torch.zeros(1))
assert message(lambda: m.register_parameter("x", zero()), KeyError) == (
    "attribute 'x' already exists"
)

# train_and_eval_reach_every_module
net = Net()
assert all(mod.training for mod in net.modules())
net.eval()
assert not any(mod.training for mod in net.modules())
net.train()
net.blocks[0].eval()
assert not net.blocks[0].stats.training and net.blocks[1].stats.training
net.train()
assert net.blocks[0].stats.training

# zero_grad_and_requires_grad_cover_every_parameter
tied = Tied()
tied.a.weight.grad = torch.ones(2, 2)
tied.zero_grad()
assert tied.a.weight.grad is None
tied.requires_grad_(False)
assert not any(p.requires_grad for p in tied.parameters())

# parameters_share_data_without_history
x = torch.ones(2, requires_grad=True)
y = x * 2
p = nn.Parameter(y)
assert p.is_leaf and p.requires_grad and p.grad_fn is None and p is not y
assert p.data_ptr() == y.data_ptr()
p.sum().backward()
assert x.grad is None
base = torch.zeros(4)
q = nn.Parameter(base[1:3])
assert q._base is None
with torch.no_grad():
    q.fill_(5.0)
assert base.tolist() == [0.0, 5.0, 5.0, 0.0]
assert message(lambda: nn.Parameter(torch.zeros(2, dtype=torch.long))) == (
    "Only Tensors of floating point and complex dtype can require gradients"
)
assert not nn.Parameter(torch.zeros(2, dtype=torch.long), requires_grad=False).requires_grad
assert repr(nn.Parameter(torch.ones(2))) == (
    "Parameter containing:\ntensor([1., 1.], requires_grad=True)"
)
assert type(p.to(torch.float64)) is torch.Tensor and p.dtype == torch.float32

# state_dict_values_share_storage_without_history
lin = nn.Linear(2, 2)
value = lin.state_dict()["weight"]
assert not value.requires_grad and value.grad_fn is None and value is not lin.weight
with torch.no_grad():
    lin.weight.fill_(3.0)
assert value.flatten().tolist() == [3.0] * 4

# load_state_dict_converts_into_the_same_tensors
tied = Tied()
weight = tied.a.weight
report = tied.load_state_dict(
    {"a.weight": torch.ones(2, 2, dtype=torch.float64), "b.weight": torch.full((2, 2), 2.0),
     "c.weight": torch.full((2, 2), 3.0)}
)
assert tied.a.weight is weight and weight.dtype == torch.float32
assert weight.flatten().tolist() == [3.0] * 4  # the last key of a shared tensor wins
assert report.missing_keys == [] and report.unexpected_keys == []
net = Net()
sd = net.state_dict()
del sd["head.weight"]
sd["extra"] = torch.zeros(1)
report = net.load_state_dict(sd, strict=False)
assert report.missing_keys == ["head.weight"] and report.unexpected_keys == ["extra"]
sd["blocks.1.lin.bias"] = torch.zeros(5)
assert message(lambda: net.load_state_dict(sd)) == (
    "Error(s) in loading state_dict for Net:\n"
    '\tMissing key(s) in state_dict: "head.weight". \n'
    '\tUnexpected key(s) in state_dict: "extra". \n'
    "\tsize mismatch for blocks.1.lin.bias: copying a param with shape torch.Size([5]) from "
    "checkpoint, the shape in current model is torch.Size([3])."
)
assert message(lambda: net.load_state_dict(sd, strict=False)) == (
    "Error(s) in loading state_dict for Net:\n"
    "\tsize mismatch for blocks.1.lin.bias: copying a param with shape torch.Size([5]) from "
    "checkpoint, the shape in current model is torch.Size([3])."
)
stats = Stats()
stats.load_state_dict({"running_mean": torch.ones(3), "num_batches_tracked": torch.tensor(3.7)})
assert stats.num_batches_tracked.dtype == torch.long and stats.num_batches_tracked.item() == 3

# load_errors_leave_the_module_unchanged: PyTorch writes the entries before the failing one
# (homura checks every entry first and writes nothing).
lin = nn.Linear(2, 2)
sd = {"weight": torch.ones(2, 2), "bias": torch.zeros(2, device="meta")}
assert message(lambda: lin.load_state_dict(sd)) == (
    "Error(s) in loading state_dict for Linear:\n"
    '\tWhile copying the parameter named "bias", whose dimensions in the model are '
    "torch.Size([2]) and whose dimensions in the checkpoint are torch.Size([2]), an exception "
    "occurred : ('Cannot copy out of meta tensor; no data!',)."
)
assert lin.weight.flatten().tolist() == [1.0] * 4

# to_converts_in_place_and_keeps_identity
net = Net()
embed, count = net.embed, net.blocks[0].stats.num_batches_tracked
embed.grad = torch.ones(4, 3)
grad = embed.grad
net.to(torch.float64)
assert net.embed is embed and embed.dtype == torch.float64 and embed.is_leaf and embed.requires_grad
# PyTorch converts the gradient tensor itself (homura replaces it by a converted copy).
assert embed.grad is grad and grad.dtype == torch.float64
assert net.cache.dtype == torch.float64
# Buffers are replaced by new tensors in PyTorch (homura converts them in place, like parameters).
assert net.blocks[0].stats.num_batches_tracked.dtype == torch.long
assert message(lambda: net.to(torch.int64)) == (
    "nn.Module.to only accepts floating point or complex dtypes, but got desired dtype=torch.int64"
)
assert message(lambda: net.to(torch.bool)) == (
    "nn.Module.to only accepts floating point or complex dtypes, but got desired dtype=torch.bool"
)
net.to("meta")
assert net.embed.is_meta and net.blocks[0].stats.num_batches_tracked.is_meta
assert message(lambda: net.to("cpu"), NotImplementedError).startswith(
    "Cannot copy out of meta tensor; no data!"
)

# old_graphs_fail_after_to: PyTorch runs old graphs, whether they saved the parameter or not, and
# writes a gradient of the old dtype into the converted parameter (homura panics instead).
lin = nn.Linear(2, 1, bias=False)
loss = (lin.weight * lin.weight).sum()
lin.to(torch.float64)
loss.backward()
assert lin.weight.dtype == torch.float64 and lin.weight.grad.dtype == torch.float32
lin = nn.Linear(2, 1, bias=False)
loss = (lin.weight + 1).sum()
lin.to(torch.float64)
loss.backward()
assert lin.weight.dtype == torch.float64 and lin.weight.grad.dtype == torch.float32

# views_and_detached_aliases_keep_the_old_data
lin = nn.Linear(2, 1, bias=False)
with torch.no_grad():
    lin.weight.fill_(1.0)
view = lin.weight[0]
alias = lin.weight.detach()
lin.to(torch.float64)
assert view.dtype == torch.float32 and alias.dtype == torch.float32
with torch.no_grad():
    alias.fill_(7.0)
assert lin.weight.flatten().tolist() == [1.0, 1.0]

# to_checks_every_tensor_before_converting_any / converted_gradients_are_new_tensors: PyTorch
# converts a gradient that is a view in place, and keeps the grad_fn of one with a history (homura
# converts the view into a new contiguous tensor and rejects the one with a history).
lin = nn.Linear(2, 2, bias=False)
lin.weight.grad = torch.zeros(4, 2)[::2]
gview = lin.weight.grad
lin.to(torch.float64)
assert lin.weight.grad is gview and gview.dtype == torch.float64
lin = nn.Linear(2, 2, bias=False)
lin.weight.grad = torch.ones(2, 2, requires_grad=True) * 1
lin.to(torch.float64)
assert lin.weight.grad.grad_fn is not None and lin.weight.grad.dtype == torch.float64

# converted_gradients_are_new_tensors: PyTorch keeps a gradient shared by two leaves as one object,
# so converting one leaf leaves the other with a gradient of another dtype, which backward passes
# keep accumulating into (homura replaces the converted leaf's gradient, and refuses to accumulate
# into a gradient that doesn't match its leaf).
a, b = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
g = torch.zeros(2)
a.grad = g
b.grad = g
holder = nn.Module()
holder.register_parameter("a", a)
holder.to(torch.float64)
assert a.grad is g and b.grad is g
assert b.dtype == torch.float32 and g.dtype == torch.float64
(b * 2).sum().backward()
assert b.grad.dtype == torch.float64 and b.grad.tolist() == [2.0, 2.0]

print("check_m2_module: all reference assertions passed")
