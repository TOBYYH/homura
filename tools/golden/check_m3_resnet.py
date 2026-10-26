"""Check the reference behavior that homura-models/tests/resnet_{config,api}.rs encode; does not
write fixtures (gen_m3_resnet.py writes those, and the structure tests compare with them).

Each block mirrors a test there: how transformers 5.20.2 reads `config.json` (defaults, labels,
dtype names), what it does with the configurations homura rejects, its activations, its
initialization and the error for pixel values with other channels.

Usage: python tools/golden/check_m3_resnet.py (needs torch and transformers).
"""

import json
import tempfile
from pathlib import Path

import reference

reference.check_transformers()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import ResNetConfig, ResNetForImageClassification, ResNetModel  # noqa: E402
from transformers.activations import ACT2FN  # noqa: E402

passed = 0


def block(f):
    global passed
    f()
    passed += 1


def read(config):
    """The configuration transformers reads from a config.json with `config`."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps({"model_type": "resnet", **config}))
        return ResNetConfig.from_pretrained(d)


@block
def missing_fields_have_transformers_defaults():
    c = read({})
    assert (c.num_channels, c.embedding_size, c.layer_type, c.hidden_act) == (3, 64, "bottleneck", "relu")
    assert list(c.hidden_sizes) == [256, 512, 1024, 2048] and list(c.depths) == [3, 4, 6, 3]
    assert not c.downsample_in_first_stage and not c.downsample_in_bottleneck
    assert c.num_labels == 2 and c.id2label == {0: "LABEL_0", 1: "LABEL_1"}
    assert c.label2id == {"LABEL_0": 0, "LABEL_1": 1} and c.dtype is None
    # Fields that homura ignores are accepted.
    read({"stage_names": ["stem"], "out_indices": [4], "out_features": None})


@block
def labels_follow_transformers():
    named = lambda n: {i: f"LABEL_{i}" for i in range(n)}
    assert read({"num_labels": 4}).id2label == named(4)
    assert read({"num_labels": 0}).id2label == {}
    abc = {"0": "a", "1": "b", "2": "c"}
    assert read({"id2label": abc}).id2label == {0: "a", 1: "b", 2: "c"}
    assert read({"id2label": abc, "num_labels": 3}).id2label == {0: "a", 1: "b", 2: "c"}
    # A num_labels that disagrees wins (after a warning) and renames the labels.
    assert read({"id2label": abc, "num_labels": 5}).id2label == named(5)
    assert read({"id2label": abc, "num_labels": 0}).id2label == {}
    c = read({"id2label": {"2": "c", "0": "a"}})
    assert c.num_labels == 2 and c.id2label == {2: "c", 0: "a"}
    # Without id2label, label2id is replaced too.
    assert read({"label2id": {"a": "x"}}).label2id == {"LABEL_0": 0, "LABEL_1": 1}
    # homura rejects a negative num_labels; transformers reads it as no labels.
    assert read({"num_labels": -1}).id2label == {}
    try:
        read({"id2label": {"x": "a"}})
        raise AssertionError("non-integer key accepted")
    except ValueError:
        pass


@block
def dtype_and_torch_dtype():
    assert read({"torch_dtype": "float32"}).dtype == torch.float32
    assert read({"dtype": "float64", "torch_dtype": "float16"}).dtype == torch.float64
    assert read({"dtype": None, "torch_dtype": "bfloat16"}).dtype == torch.bfloat16
    for name, dtype in [("float", torch.float32), ("double", torch.float64), ("half", torch.float16),
                        ("float16", torch.float16)]:
        assert read({"dtype": name}).dtype == dtype
    # Names of dtypes that aren't floating point are read too; homura rejects them.
    assert read({"dtype": "int8"}).dtype == torch.int8


@block
def invalid_configurations():
    def builds(**kwargs):
        try:
            ResNetForImageClassification(ResNetConfig(**kwargs))
            return True
        except Exception:  # noqa: BLE001
            return False

    assert not builds(hidden_sizes=[], depths=[])
    assert not builds(hidden_act="nope", embedding_size=8, hidden_sizes=[8], depths=[1])
    # transformers has `mish`; homura doesn't, and rejects it.
    assert builds(hidden_act="mish", embedding_size=8, hidden_sizes=[8], depths=[1])
    assert not builds(layer_type="wide")
    # transformers builds these, which homura rejects.
    m = ResNetModel(ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1]))
    assert len(m.encoder.stages) == 1
    m = ResNetModel(ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 0]))
    assert len(m.encoder.stages[1].layers) == 1
    assert builds(embedding_size=8, hidden_sizes=[2], depths=[1])
    # Basic layers accept small hidden sizes.
    assert builds(layer_type="basic", embedding_size=8, hidden_sizes=[2, 3], depths=[1, 1])


@block
def activations_are_the_core_layers():
    x = torch.linspace(-4, 4, 101, dtype=torch.float64)
    core = {
        "relu": (torch.relu(x), "ReLU()"),
        "sigmoid": (torch.sigmoid(x), "Sigmoid()"),
        "tanh": (torch.tanh(x), "Tanh()"),
        "silu": (nn.functional.silu(x), "SiLUActivation()"),
        "swish": (nn.functional.silu(x), "SiLU()"),
        "gelu": (nn.functional.gelu(x), "GELUActivation()"),
        "gelu_pytorch_tanh": (nn.functional.gelu(x, approximate="tanh"), "GELUTanh()"),
    }
    for name, (expected, text) in core.items():
        act = ACT2FN[name]
        assert torch.equal(act(x), expected), name
        assert repr(act) == text, (name, repr(act))
    for name in ["gelu_new", "mish"]:
        assert name in ACT2FN  # transformers has them; homura doesn't
    # hidden_act is used by the stem and each layer's output; the inner convolutions keep ReLU.
    m = ResNetModel(ResNetConfig(layer_type="basic", embedding_size=8, hidden_sizes=[8, 16],
                                 depths=[1, 2], hidden_act="gelu"))
    text = repr(m)
    assert text.count("GELUActivation()") == 1 + 3 and text.count("ReLU()") == 3


@block
def initialization_is_transformers():
    torch.manual_seed(0)
    m = ResNetForImageClassification(ResNetConfig(layer_type="basic", embedding_size=64,
                                                  hidden_sizes=[64], depths=[1], num_labels=100))
    w = m.resnet.embedder.embedder.convolution.weight
    expected = (2 / (64 * 49)) ** 0.5
    assert abs(w.std().item() / expected - 1) < 0.05
    for k, v in m.state_dict().items():
        if k.endswith("normalization.weight") or k.endswith("running_var"):
            assert torch.all(v == 1), k
        elif k.endswith(("normalization.bias", "running_mean", "num_batches_tracked")):
            assert torch.all(v == 0), k
    for k in ["classifier.1.weight", "classifier.1.bias"]:
        v = m.state_dict()[k]
        assert v.abs().max() <= 0.125 and v.abs().max() > 0.1


@block
def outputs_and_errors():
    config = ResNetConfig(layer_type="basic", embedding_size=8, hidden_sizes=[8, 16], depths=[1, 2],
                          num_labels=5)
    m = ResNetModel(config).eval()
    out = m(torch.zeros(2, 3, 32, 32))
    assert out.last_hidden_state.shape == (2, 16, 4, 4) and out.pooler_output.shape == (2, 16, 1, 1)
    try:
        m(torch.zeros(1, 4, 32, 32))
        raise AssertionError("other channels accepted")
    except ValueError as e:
        assert str(e) == ("Make sure that the channel dimension of the pixel values match with "
                          "the one set in the configuration.")
    clf = ResNetForImageClassification(config.__class__(**{**config.to_dict(), "num_labels": 0}))
    assert isinstance(clf.classifier[1], nn.Identity)
    assert clf(torch.zeros(2, 3, 32, 32)).logits.shape == (2, 16)
    # 1 + 3 + 5 batch normalizations with a projection in the first stage.
    m = ResNetModel(ResNetConfig(layer_type="basic", embedding_size=8, hidden_sizes=[8, 16],
                                 depths=[1, 2], downsample_in_first_stage=True))
    assert sum(isinstance(x, nn.BatchNorm2d) for x in m.modules()) == 9


print(f"check_m3_resnet: {passed} blocks passed")
