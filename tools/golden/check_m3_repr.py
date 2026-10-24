"""Check the reference behavior that tests/nn_display.rs encodes; does not write fixtures.

Each block mirrors a test there, in the same order: PyTorch's `repr` of modules (`print(model)`),
which homura's `Module::display` follows, and `sum(p.numel() for p in model.parameters())`, which
`Module::num_parameters` follows. The expected texts are homura's: PyTorch's text with homura's
type names (`to_homura`), checked against `repr` here and compared verbatim by the Rust tests.
Where homura prints something else on purpose (the sizes of pooling layers given as equal pairs),
the block asserts PyTorch's text and the difference.
"""

import math

import torch
import torch.nn as nn

import reference

reference.check()
passed = 0


def block(f):
    global passed
    f()
    passed += 1


# PyTorch's class names that homura spells differently.
NAMES = [("ReLU", "Relu"), ("GELU", "Gelu"), ("SiLU", "Silu"), ("RMSNorm", "RmsNorm")]


def to_homura(text):
    for torch_name, homura_name in NAMES:
        text = text.replace(torch_name + "(", homura_name + "(")
    return text


def check(module, expected):
    found = to_homura(repr(module))
    assert found == expected, (found, expected)


@block
def layers_print_their_settings():
    cases = [
        (nn.Linear(8, 4), "Linear(in_features=8, out_features=4, bias=True)"),
        (nn.Linear(8, 4, bias=False), "Linear(in_features=8, out_features=4, bias=False)"),
        (nn.Linear(0, 0), "Linear(in_features=0, out_features=0, bias=True)"),
        (nn.Conv2d(3, 8, 3), "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))"),
        (
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))",
        ),
        (
            nn.Conv2d(3, 8, (3, 2), stride=(2, 1), padding=(1, 0), dilation=(2, 1), bias=False),
            "Conv2d(3, 8, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=(2, 1), "
            "bias=False)",
        ),
        (
            nn.Conv2d(4, 8, 3, groups=2, dilation=2),
            "Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), groups=2)",
        ),
        (
            nn.Conv2d(3, 8, 3, padding="same"),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)",
        ),
        (
            nn.Conv2d(3, 8, 3, padding="valid"),
            "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=valid)",
        ),
        (nn.Conv2d(3, 8, 3, padding=(0, 0)), "Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))"),
        (
            nn.BatchNorm2d(8),
            "BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, bias=True, "
            "track_running_stats=True)",
        ),
        (
            nn.BatchNorm2d(8, eps=1e-3, momentum=None, affine=False, track_running_stats=False),
            "BatchNorm2d(8, eps=0.001, momentum=None, affine=False, bias=False, "
            "track_running_stats=False)",
        ),
        (
            nn.BatchNorm2d(8, momentum=0.01, bias=False),
            "BatchNorm2d(8, eps=1e-05, momentum=0.01, affine=True, bias=False, "
            "track_running_stats=True)",
        ),
        (
            nn.MaxPool2d(2),
            "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
        ),
        (
            nn.MaxPool2d(3, 2, 1, dilation=2, ceil_mode=True),
            "MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True)",
        ),
        (
            nn.MaxPool2d((3, 2), (2, 1), (1, 0)),
            "MaxPool2d(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=1, "
            "ceil_mode=False)",
        ),
        (nn.AdaptiveAvgPool2d(1), "AdaptiveAvgPool2d(output_size=1)"),
        (nn.AdaptiveAvgPool2d((2, 3)), "AdaptiveAvgPool2d(output_size=(2, 3))"),
        (nn.Flatten(), "Flatten(start_dim=1, end_dim=-1)"),
        (nn.Flatten(0, -2), "Flatten(start_dim=0, end_dim=-2)"),
        (nn.Dropout(), "Dropout(p=0.5, inplace=False)"),
        (nn.Dropout(0.1), "Dropout(p=0.1, inplace=False)"),
        (nn.Dropout(0.0), "Dropout(p=0.0, inplace=False)"),
        (nn.Dropout(1.0), "Dropout(p=1.0, inplace=False)"),
        (nn.LayerNorm(4), "LayerNorm((4,), eps=1e-05, elementwise_affine=True, bias=True)"),
        (
            nn.LayerNorm([2, 3], eps=1e-6, elementwise_affine=False),
            "LayerNorm((2, 3), eps=1e-06, elementwise_affine=False, bias=False)",
        ),
        (
            nn.LayerNorm(4, bias=False),
            "LayerNorm((4,), eps=1e-05, elementwise_affine=True, bias=False)",
        ),
        (nn.RMSNorm(4), "RmsNorm((4,), eps=None, elementwise_affine=True)"),
        (nn.RMSNorm([2, 3], eps=1e-6), "RmsNorm((2, 3), eps=1e-06, elementwise_affine=True)"),
        (
            nn.RMSNorm(4, elementwise_affine=False),
            "RmsNorm((4,), eps=None, elementwise_affine=False)",
        ),
        (nn.Embedding(10, 4), "Embedding(10, 4)"),
        (nn.Embedding(10, 4, padding_idx=0), "Embedding(10, 4, padding_idx=0)"),
        (nn.Embedding(10, 4, padding_idx=-1), "Embedding(10, 4, padding_idx=9)"),
        (nn.ReLU(), "Relu()"),
        (nn.GELU(), "Gelu(approximate='none')"),
        (nn.GELU(approximate="tanh"), "Gelu(approximate='tanh')"),
        (nn.SiLU(), "Silu()"),
        (nn.Tanh(), "Tanh()"),
        (nn.Sigmoid(), "Sigmoid()"),
        (nn.Softmax(-1), "Softmax(dim=-1)"),
        (nn.Softmax(1), "Softmax(dim=1)"),
        (nn.Sequential(), "Sequential()"),
        (nn.ModuleList(), "ModuleList()"),
        (nn.ModuleDict(), "ModuleDict()"),
    ]
    for module, expected in cases:
        check(module, expected)
    # The names homura spells differently.
    assert [repr(m) for m in [nn.ReLU(), nn.SiLU()]] == ["ReLU()", "SiLU()"]
    assert repr(nn.GELU()).startswith("GELU(") and repr(nn.RMSNorm(4)).startswith("RMSNorm(")


@block
def floats_are_printed_as_python_prints_them():
    check(
        nn.BatchNorm2d(3, eps=1e-20, momentum=1e16),
        "BatchNorm2d(3, eps=1e-20, momentum=1e+16, affine=True, bias=True, "
        "track_running_stats=True)",
    )
    check(
        nn.BatchNorm2d(3, eps=123456.0, momentum=1 / 3),
        "BatchNorm2d(3, eps=123456.0, momentum=0.3333333333333333, affine=True, bias=True, "
        "track_running_stats=True)",
    )
    check(nn.Dropout(0.25), "Dropout(p=0.25, inplace=False)")
    tie = 1_000_000_000_000_000.25
    check(
        nn.LayerNorm(1, eps=tie),
        "LayerNorm((1,), eps=1000000000000000.2, elementwise_affine=True, bias=True)",
    )
    check(nn.RMSNorm(1, eps=-tie), "RmsNorm((1,), eps=-1000000000000000.2, elementwise_affine=True)")
    check(
        nn.BatchNorm2d(1, eps=2.0 ** -25, momentum=1_000_000_000_000_000.75),
        "BatchNorm2d(1, eps=2.9802322387695312e-08, momentum=1000000000000000.8, affine=True, "
        "bias=True, track_running_stats=True)",
    )
    check(
        nn.LayerNorm(4, eps=1.5e-4),
        "LayerNorm((4,), eps=0.00015, elementwise_affine=True, bias=True)",
    )
    # The values of the unit test of homura's float formatting (src/nn/display.rs).
    floats = [
        (0.0, "0.0"), (-0.0, "-0.0"), (1.0, "1.0"), (0.5, "0.5"), (0.1, "0.1"),
        (1e-5, "1e-05"), (1e-4, "0.0001"), (1.5e-4, "0.00015"), (0.001, "0.001"),
        (1e-20, "1e-20"), (1.25e-7, "1.25e-07"), (123456.0, "123456.0"),
        (1e15, "1000000000000000.0"), (1e16, "1e+16"), (1.5e300, "1.5e+300"),
        (1 / 3, "0.3333333333333333"), (-2.5, "-2.5"),
        (1.7976931348623157e308, "1.7976931348623157e+308"), (5e-324, "5e-324"),
        (float("inf"), "inf"), (float("-inf"), "-inf"), (float("nan"), "nan"),
        # Two shortest candidates equally close: the even last digit, up or down (R111).
        (1_000_000_000_000_000.25, "1000000000000000.2"),
        (-1_000_000_000_000_000.25, "-1000000000000000.2"),
        (1_000_000_000_000_000.75, "1000000000000000.8"),
        (252_158_283_539_364.625, "252158283539364.62"),
        (2.0 ** -25, "2.9802322387695312e-08"),
        (3 * 2.0 ** -24, "1.7881393432617188e-07"),
        (math.nextafter(1_000_000_000_000_000.25, math.inf), "1000000000000000.4"),
        (math.nextafter(1_000_000_000_000_000.25, -math.inf), "1000000000000000.1"),
        (2.0 ** -1007, "7.291122019556398e-304"),
    ]
    # The ties are exact: both candidates read back as the value.
    assert (1_000_000_000_000_000.25).as_integer_ratio() == (4000000000000001, 4)
    assert float("1000000000000000.2") == float("1000000000000000.3") == 1_000_000_000_000_000.25
    assert float("2.9802322387695312e-08") == float("2.9802322387695313e-08") == 2.0 ** -25
    for x, text in floats:
        assert repr(x) == text, (x, repr(x))


@block
def pooling_sizes_given_as_equal_pairs():
    # PyTorch prints what it was given; homura keeps pairs, and prints equal ones as one number.
    assert repr(nn.MaxPool2d((2, 2))) == (
        "MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)"
    )
    assert repr(nn.AdaptiveAvgPool2d((1, 1))) == "AdaptiveAvgPool2d(output_size=(1, 1))"


@block
def containers_list_their_children():
    check(
        nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Sequential(nn.Tanh())),
        "Sequential(\n"
        "  (0): Linear(in_features=2, out_features=3, bias=True)\n"
        "  (1): Relu()\n"
        "  (2): Sequential(\n"
        "    (0): Tanh()\n"
        "  )\n"
        ")",
    )
    # Insertion order, which homura's `BTreeMap` has when the keys are inserted in key order.
    check(
        nn.ModuleDict({"a": nn.Linear(1, 1), "b": nn.ReLU()}),
        "ModuleDict(\n"
        "  (a): Linear(in_features=1, out_features=1, bias=True)\n"
        "  (b): Relu()\n"
        ")",
    )
    # A `Sequential` lists each child; a `ModuleList` groups runs of the same text.
    check(
        nn.Sequential(nn.ReLU(), nn.ReLU()),
        "Sequential(\n  (0): Relu()\n  (1): Relu()\n)",
    )
    check(
        nn.ModuleList([nn.Linear(2, 2) for _ in range(3)] + [nn.ReLU(), nn.ReLU(), nn.Linear(2, 2)]),
        "ModuleList(\n"
        "  (0-2): 3 x Linear(in_features=2, out_features=2, bias=True)\n"
        "  (3-4): 2 x Relu()\n"
        "  (5): Linear(in_features=2, out_features=2, bias=True)\n"
        ")",
    )
    check(
        nn.ModuleList([nn.Sequential(nn.ReLU()), nn.Sequential(nn.ReLU())]),
        "ModuleList(\n  (0-1): 2 x Sequential(\n    (0): Relu()\n  )\n)",
    )
    check(nn.ModuleList([nn.ReLU()]), "ModuleList(\n  (0): Relu()\n)")
    check(
        nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 2), nn.Linear(1, 1)]),
        "ModuleList(\n"
        "  (0): Linear(in_features=1, out_features=1, bias=True)\n"
        "  (1): Linear(in_features=1, out_features=2, bias=True)\n"
        "  (2): Linear(in_features=1, out_features=1, bias=True)\n"
        ")",
    )


class Multi(nn.Module):
    """A module with a multi-line `extra_repr` and a child."""

    def __init__(self, text, child):
        super().__init__()
        self.text = text
        self.inner = child

    def extra_repr(self):
        return self.text


class Leaf(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def extra_repr(self):
        return self.text


@block
def extra_repr_lines_are_indented():
    check(Leaf("x"), "Leaf(x)")
    check(Leaf(""), "Leaf()")
    check(Leaf("x\ny"), "Leaf(\n  x\n  y\n)")
    check(Multi("x", nn.ReLU()), "Multi(\n  x\n  (inner): Relu()\n)")
    # Blank lines: at the top they keep the two spaces of the join; nested, they become empty.
    multi = Multi("a\n\nb\n", nn.Sequential(Leaf("x\ny")))
    text = (
        "Multi(\n"
        "  a\n"
        "  \n"
        "  b\n"
        "  \n"
        "  (inner): Sequential(\n"
        "    (0): Leaf(\n"
        "      x\n"
        "      y\n"
        "    )\n"
        "  )\n"
        ")"
    )
    check(multi, text)
    check(
        nn.Sequential(multi),
        "Sequential(\n"
        "  (0): Multi(\n"
        "    a\n"
        "\n"
        "    b\n"
        "\n"
        "    (inner): Sequential(\n"
        "      (0): Leaf(\n"
        "        x\n"
        "        y\n"
        "      )\n"
        "    )\n"
        "  )\n"
        ")",
    )
    check(
        nn.ModuleList([Leaf("x\ny"), Leaf("x\ny")]),
        "ModuleList(\n  (0-1): 2 x Leaf(\n    x\n    y\n  )\n)",
    )


class Net(nn.Module):
    """A shared `Linear` under two names, a parameter (not printed) and nested containers."""

    def __init__(self):
        super().__init__()
        shared = nn.Linear(2, 2)
        self.p = nn.Parameter(torch.zeros(2))
        self.a = shared
        self.b = shared
        self.empty = nn.ModuleList()
        self.blocks = nn.ModuleList([nn.ReLU(), nn.Sequential(nn.Linear(2, 2))])


@block
def shared_modules_appear_under_each_name():
    net = Net()
    check(
        net,
        "Net(\n"
        "  (a): Linear(in_features=2, out_features=2, bias=True)\n"
        "  (b): Linear(in_features=2, out_features=2, bias=True)\n"
        "  (empty): ModuleList()\n"
        "  (blocks): ModuleList(\n"
        "    (0): Relu()\n"
        "    (1): Sequential(\n"
        "      (0): Linear(in_features=2, out_features=2, bias=True)\n"
        "    )\n"
        "  )\n"
        ")",
    )
    assert [name for name, _ in net.named_children()] == ["a", "empty", "blocks"]


def num_parameters(module):
    return sum(p.numel() for p in module.parameters())


@block
def parameters_are_counted_once_by_elements():
    assert num_parameters(nn.Linear(2, 3)) == 9
    assert num_parameters(nn.Linear(2, 3, bias=False)) == 6
    # Buffers don't count.
    assert num_parameters(nn.BatchNorm2d(8)) == 16
    assert num_parameters(nn.BatchNorm2d(8, affine=False)) == 0
    # A shared module once, frozen parameters too.
    assert num_parameters(Net()) == 2 + 6 + 6
    frozen = nn.Linear(2, 3)
    frozen.weight.requires_grad_(False)
    assert num_parameters(frozen) == 9
    assert num_parameters(nn.Sequential()) == 0


print(f"check_m3_repr: {passed} blocks passed")
