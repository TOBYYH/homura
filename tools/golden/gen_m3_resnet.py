"""Generates homura-models/tests/golden/: small Hugging Face repositories of transformers' ResNet and
what the reference computes with them (M3 step 4a1), and the module trees of the configurations
the tests build.

- repos/<name>/: what `save_pretrained` writes for a small `ResNetForImageClassification` (seeded;
  the batch normalization parameters and statistics are then set to random values, as after
  training): `config.json` and `model.safetensors`, plus the same weights saved with `torch.save`
  as `pytorch_model.bin`, as older repositories have them.
- resnet/inputs.safetensors: `pixel_values`, 2 × 3 × 32 × 32 values `((i·7919 + 13) mod 1024) / 1024
  − 0.5` (exact in F32 and F64), and `target`, the classes [3, 7] for the loss.
- resnet/<name>.safetensors: for the repository's model, in eval mode, `eval_f32.*` and
  `eval_f64.*` (the model and the input converted to F64): `logits`, `last_hidden_state` and
  `pooler_output`. In train mode, in F64: `train.logits` of one forward computation, `train.<key>`
  for every buffer of the state dict after it (the running statistics and counts); and in eval
  mode, in F64, `grad.<key>` for every parameter after `cross_entropy(logits, target).backward()`
  (in train mode the batch statistics of the 1 x 1 last stages make the gradients
  ill-conditioned).
- resnet/structure.json: for the configurations of microsoft/resnet-18 and microsoft/resnet-50
  (their structural fields; the label names are left out) and of every repository, the config
  dict, and for `ResNetForImageClassification` the state dict's keys with
  shapes and dtypes, its `_metadata` versions, the number of parameters and `repr(model)`; for
  `ResNetModel` only the number of parameters, the rest being the classification model's under
  `resnet` (see `base_tree`).

`torch.save` writes a random serialization id and the safetensors writer orders its header
differently from run to run, so a file is only rewritten when its content changes: regenerating
leaves the files as they are.

Usage: python tools/golden/gen_m3_resnet.py (needs torch and transformers).
"""

import io
import json
import struct
import zipfile
from pathlib import Path

import reference

reference.check_transformers()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load as load_safetensors  # noqa: E402
from safetensors.torch import save as save_safetensors  # noqa: E402
from transformers import ResNetConfig, ResNetForImageClassification, ResNetModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "homura-models" / "tests" / "golden"

# The structural fields of the two repositories of the M3 acceptance (config.json at the commits
# of M3.3; written by transformers 4.18).
REAL = {
    "resnet-18": {
        "architectures": ["ResNetForImageClassification"], "depths": [2, 2, 2, 2],
        "downsample_in_first_stage": False, "embedding_size": 64, "hidden_act": "relu",
        "hidden_sizes": [64, 128, 256, 512], "layer_type": "basic", "model_type": "resnet",
        "num_channels": 3, "num_labels": 1000, "torch_dtype": "float32",
    },
    "resnet-50": {
        "architectures": ["ResNetForImageClassification"], "depths": [3, 4, 6, 3],
        "downsample_in_first_stage": False, "embedding_size": 64, "hidden_act": "relu",
        "hidden_sizes": [256, 512, 1024, 2048], "layer_type": "bottleneck", "model_type": "resnet",
        "num_channels": 3, "num_labels": 1000, "torch_dtype": "float32",
    },
}

# The repositories: both layer types, one and two layers per stage, identity and projection
# shortcuts, both downsampling options, 10 labels and none.
REPOS = {
    "basic": dict(layer_type="basic", embedding_size=8, hidden_sizes=[8, 8, 12, 12],
                  depths=[2, 1, 2, 1], num_labels=10),
    "basic_no_labels": dict(layer_type="basic", embedding_size=4, hidden_sizes=[8, 8, 12, 12],
                            depths=[1, 1, 1, 1], downsample_in_first_stage=True, num_labels=0),
    "bottleneck": dict(layer_type="bottleneck", embedding_size=8, hidden_sizes=[8, 8, 16, 16],
                       depths=[2, 1, 2, 1], num_labels=10),
    "bottleneck_downsample": dict(layer_type="bottleneck", embedding_size=8,
                                  hidden_sizes=[8, 12, 16, 16], depths=[1, 1, 1, 1],
                                  downsample_in_first_stage=True, downsample_in_bottleneck=True,
                                  num_labels=10),
}


def records(data):
    """The records of a zip archive but the random serialization id, or None for other data."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    return [(n, z.read(n)) for n in z.namelist() if not n.endswith("/.data/serialization_id")]


def safetensors_content(data):
    """The metadata and tensors of a safetensors file, whatever the order of its header."""
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    body = data[8 + n:]
    tensors = {k: (v["dtype"], v["shape"], body[v["data_offsets"][0]:v["data_offsets"][1]])
               for k, v in header.items() if k != "__metadata__"}
    return header.get("__metadata__"), tensors


def write_file(path, data):
    """Writes `data` to `path` unless the file holds the same content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_bytes()
        if old == data or (records(old) is not None and records(old) == records(data)) \
                or (path.suffix == ".safetensors"
                    and safetensors_content(old) == safetensors_content(data)):
            return
    path.write_bytes(data)


def pixel_values():
    i = torch.arange(2 * 3 * 32 * 32, dtype=torch.int64)
    return (((i * 7919 + 13) % 1024).to(torch.float64) / 1024 - 0.5).to(torch.float32).view(2, 3, 32, 32)


def build(cls, config, seed):
    torch.manual_seed(seed)
    model = cls(config)
    # Trained-looking batch normalization, so that the statistics matter in eval mode.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
                module.running_mean.uniform_(-0.2, 0.2)
                module.running_var.uniform_(0.5, 1.5)
                module.num_batches_tracked.fill_(7)
    return model


def contiguous(tensors):
    return {k: v.detach().contiguous() for k, v in tensors.items()}


def outputs(model, x):
    out = {}
    for dtype, prefix in ((torch.float32, "eval_f32"), (torch.float64, "eval_f64")):
        m = build_copy(model, dtype)
        m.eval()
        with torch.no_grad():
            base = m.resnet(x.to(dtype))
            out[f"{prefix}.logits"] = m(x.to(dtype)).logits
            out[f"{prefix}.last_hidden_state"] = base.last_hidden_state
            out[f"{prefix}.pooler_output"] = base.pooler_output
    m = build_copy(model, torch.float64)
    m.train()
    logits = m(x.to(torch.float64)).logits
    out["train.logits"] = logits
    for k, v in m.state_dict().items():
        if k not in dict(m.named_parameters()):
            out[f"train.{k}"] = v
    # The gradients in eval mode: in train mode the batch statistics of 2 values (the last
    # stages are 1 x 1) make them ill-conditioned, so that a change of a few ulps in the weights
    # changes them by up to 1e-3.
    m = build_copy(model, torch.float64)
    m.eval()
    F.cross_entropy(m(x.to(torch.float64)).logits, torch.tensor([3, 7])).backward()
    for k, p in m.named_parameters():
        out[f"grad.{k}"] = p.grad
    return contiguous(out)


def build_copy(model, dtype):
    m = ResNetForImageClassification(model.config)
    m.load_state_dict(model.state_dict())
    return m.to(dtype)


def tree(model, with_repr):
    sd = model.state_dict()
    out = {
        "keys": [[k, list(v.shape), str(v.dtype).removeprefix("torch.")] for k, v in sd.items()],
        "versions": {k: v["version"] for k, v in sd._metadata.items()},
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    if with_repr:
        out["repr"] = repr(model)
    return out


def structure(config_dict):
    with torch.device("meta"):
        config = ResNetConfig.from_dict(config_dict)
        return {
            "config": config_dict,
            "classification": tree(ResNetForImageClassification(config), True),
            "base": base_tree(ResNetModel(config), ResNetForImageClassification(config)),
        }


def base_tree(base, clf):
    """The number of parameters of `ResNetModel`. The rest of its tree is the classification
    model's under `resnet` (asserted; the tests derive it the same way): the `repr` is the block
    `(resnet): ...`, one level less indented, the keys and versions those under `resnet.`."""
    lines = repr(clf).splitlines()
    start = lines.index("  (resnet): ResNetModel(")
    end = lines.index("  )", start)
    block = ["ResNetModel("] + [line[2:] for line in lines[start + 1:end]] + [")"]
    assert "\n".join(block) == repr(base)
    # Its keys and versions are the classification model's under `resnet`, without the prefix.
    full, own = tree(clf, False), tree(base, False)
    assert own["keys"] == [[k.removeprefix("resnet."), s, d] for k, s, d in full["keys"]
                           if k.startswith("resnet.")]
    assert own["versions"] == {("" if k == "resnet" else k.removeprefix("resnet.")): v
                               for k, v in full["versions"].items()
                               if k == "resnet" or k.startswith("resnet.")} | {"": 1}
    return {"num_parameters": own["num_parameters"]}


def main():
    x = pixel_values()
    write_file(OUT / "resnet" / "inputs.safetensors",
               save_safetensors({"pixel_values": x, "target": torch.tensor([3, 7])},
                                metadata=reference.metadata()))
    structures = {}
    for name, config in REAL.items():
        local = ROOT / "target" / "review-m3-plan" / "microsoft" / name / "config.json"
        if local.exists():
            raw = json.loads(local.read_text())
            assert {k: raw[k] for k in config if k != "num_labels"} == \
                {k: v for k, v in config.items() if k != "num_labels"}, name
            assert len(raw["id2label"]) == config["num_labels"]
        structures[name] = structure(config)

    for seed, (name, kwargs) in enumerate(REPOS.items()):
        config = ResNetConfig(**kwargs)
        model = build(ResNetForImageClassification, config, seed)
        repo = OUT / "repos" / name
        tmp = OUT / "repos" / f".{name}.tmp"
        model.save_pretrained(tmp)
        files = sorted(p.name for p in tmp.iterdir())
        assert files == ["config.json", "model.safetensors"], files
        config_text = (tmp / "config.json").read_text()
        weights = (tmp / "model.safetensors").read_bytes()
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
        write_file(repo / "config.json", config_text.encode())
        write_file(repo / "model.safetensors", weights)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        write_file(repo / "pytorch_model.bin", buf.getvalue())
        # Both files hold the state dict, and the reference reads the repository back.
        sd = model.state_dict()
        st = load_safetensors(weights)
        assert sorted(st) == sorted(sd)
        assert all(torch.equal(st[k], sd[k]) for k in sd)
        loaded = ResNetForImageClassification.from_pretrained(repo)
        assert all(torch.equal(loaded.state_dict()[k], sd[k]) for k in sd)
        write_file(OUT / "resnet" / f"{name}.safetensors",
                   save_safetensors(outputs(model, x), metadata=reference.metadata()))
        structures[name] = structure(json.loads(config_text))

    write_file(OUT / "resnet" / "structure.json",
               (json.dumps(structures, separators=(",", ":")) + "\n").encode())
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"homura-models/tests/golden: {total} bytes")
    assert total < 1_000_000


main()
