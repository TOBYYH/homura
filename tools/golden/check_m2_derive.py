"""Check the reference behavior that tests/nn_derive.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order, with the PyTorch modules that the derived
structs and `Sequential`s stand for: a field is an attribute assigned in `__init__` in field
order, `#[module(rename = "k")]` is the attribute name `k`, `#[module(non_persistent)]` a buffer
registered with `persistent=False`, `#[module(skip)]` a plain attribute, `Vec` an
`nn.ModuleList` and an `Arc` held twice the same module object. The traversals of the derived
`Net` are those of check_m2_module.py; only its train / eval part is repeated here. The tests of
generic structs with plain values, raw identifiers and the crate path have no PyTorch
counterpart beyond the keys asserted.
"""

import torch
from torch import nn

import reference

reference.check()


def keys(pairs):
    return [k for k, _ in pairs]


def state_keys(m):
    return list(m.state_dict().keys())


def linear(weight, outputs):
    lin = nn.Linear(len(weight) // outputs, outputs)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(weight).reshape(outputs, -1))
        lin.bias.fill_(1.0)
    return lin


# derived_modules_list_their_members_like_hand_written_ones: train / eval.
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
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


net = Net()
net.blocks[1].stats.training = False
net.train()
assert net.training and net.blocks[1].stats.training
net.blocks[0].eval()
assert net.training and not net.blocks[0].stats.training


# attributes_rename_skip_and_keep_buffers_out_of_the_state_dict
class Norm(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.register_buffer("cache", torch.zeros(4), persistent=False)
        self.register_buffer("mask", mask, persistent=False)
        self.log = []
        self.scratch = torch.zeros(4)
        self.eps = 1e-5


norm = Norm()
assert state_keys(norm) == ["weight"]
assert keys(norm.named_parameters()) == ["weight"]
assert keys(norm.named_buffers()) == ["cache"]
assert keys(norm.named_modules()) == [""]
norm.to(torch.float64)
assert norm.weight.dtype == torch.float64 and norm.cache.dtype == torch.float64
assert norm.scratch.dtype == torch.float32
norm.load_state_dict({"weight": torch.full((4,), 2.0)}, strict=True)
assert norm.weight.tolist() == [2.0] * 4
result = norm.load_state_dict({"weight": torch.full((4,), 2.0), "cache": torch.zeros(4)}, strict=False)
assert result.unexpected_keys == ["cache"] and result.missing_keys == []
try:
    norm.load_state_dict({"weight": torch.full((4,), 2.0), "cache": torch.zeros(4)}, strict=True)
    raise AssertionError("expected an error")
except RuntimeError as e:
    assert 'Unexpected key(s) in state_dict: "cache".' in str(e)
with_mask = Norm(mask=torch.ones(1))
assert keys(with_mask.named_buffers()) == ["cache", "mask"]
assert state_keys(with_mask) == ["weight"]


# generic_structs_are_modules
class Wrap(nn.Module):
    def __init__(self, inner, extra=None, scale=1.0):
        super().__init__()
        self.inner = inner
        self.extra = extra
        self.scale = scale

    def forward(self, x):
        return self.inner(x) * self.scale


assert state_keys(Wrap(nn.Linear(2, 1))) == ["inner.weight", "inner.bias"]
wrap = Wrap(nn.ModuleList([nn.Linear(2, 1, bias=False)]), extra=nn.ModuleList())
assert state_keys(wrap) == ["inner.0.weight"]
assert keys(wrap.named_modules()) == ["", "inner", "inner.0", "extra"]
assert keys(Wrap(3).named_modules()) == [""]


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)])


assert state_keys(Stack()) == ["layers.0.weight", "layers.1.weight"]


# shared_unit_and_raw_named_members
class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = self.a


tied = Tied()
assert state_keys(tied) == ["a.weight", "b.weight"]
assert keys(tied.named_parameters()) == ["a.weight"]
assert keys(tied.named_modules()) == ["", "a"]


class Keywords(nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, "for", nn.Parameter(torch.zeros(1)))
        setattr(self, "in", nn.Identity())


keywords = Keywords()
assert state_keys(keywords) == ["for"]
assert keys(keywords.named_modules()) == ["", "in"]
assert keys(nn.ReLU().named_modules()) == [""]


# the_crate_path_can_be_given: the parameter comes before the submodule's.
class Scaled(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(1, 1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))


assert state_keys(Scaled()) == ["scale", "inner.weight"]


# sequential_names_layers_by_position
model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Sequential(nn.Linear(2, 1, bias=False)))
assert state_keys(model) == ["0.weight", "0.bias", "2.0.weight"]
assert keys(model.named_modules()) == ["", "0", "1", "2", "2.0"]
assert keys(model.named_children()) == ["0", "1", "2"]
assert len(model) == 3
assert state_keys(model[2]) == ["0.weight"]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        self.features.append(nn.Linear(2, 2, bias=False))
        self.features.append(nn.ReLU())
        self.head = nn.Linear(2, 1, bias=False)


classifier = Classifier()
assert state_keys(classifier) == ["features.0.weight", "head.weight"]
assert keys(classifier.named_modules()) == ["", "features", "features.0", "features.1", "head"]


# sequential_applies_its_layers_in_order
model = nn.Sequential(
    linear([1.0, -3.0, 3.0, 4.0], 2), nn.ReLU(), Wrap(linear([1.0, 1.0], 1), scale=0.5)
)
x = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
# No pre-activation of the relu is zero.
assert (model[0](x) != 0).all()
y = model(x)
assert y.flatten().tolist() == [4.5, 5.5]
outer = nn.Sequential(model)
assert outer(x).flatten().tolist() == [4.5, 5.5]
y.sum().backward()
assert next(outer.parameters()).grad.flatten().tolist() == [1.0, 0.0, 1.5, 0.5]
empty = nn.Sequential()
assert empty(x) is x
assert len(empty) == 0 and state_keys(empty) == []


# sequential_layers_can_be_shared_and_converted
shared = linear([1.0, 0.0, 0.0, 1.0], 2)
model = nn.Sequential(shared, nn.ReLU(), shared)
assert state_keys(model) == ["0.weight", "0.bias", "2.weight", "2.bias"]
assert keys(model.named_parameters()) == ["0.weight", "0.bias"]
assert keys(model.named_modules()) == ["", "0", "1"]
assert model(torch.tensor([[1.0, -3.0]])).flatten().tolist() == [3.0, 1.0]
model.to(torch.float64)
assert shared.weight.dtype == torch.float64

print("check_m2_derive: all reference checks passed")
