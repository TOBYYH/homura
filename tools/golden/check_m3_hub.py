"""Check the reference behavior that homura-models/tests/{hub,hub_env,pretrained}.rs encode; does
not write fixtures (the tests build their repositories from tests/golden/repos, as this script
does).

Each block mirrors tests there: where huggingface_hub 2.3.0 puts its cache, which repository ids,
revisions and file names it accepts, how an offline lookup resolves a revision, which weight files
transformers 5.20.2 picks (in a local directory and in the cache), `transformers_weights`, the
`resnet.` prefix of the two models, the dtype `from_pretrained` gives the model, and that loaded
models are in eval mode with parameters that require grad, without `_metadata` being read.

Usage: python tools/golden/check_m3_hub.py (needs torch, transformers and huggingface_hub).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import reference

reference.check_hub()

import torch  # noqa: E402
from huggingface_hub.utils import HFValidationError, validate_repo_id  # noqa: E402
from huggingface_hub._local_folder import _validate_relative_filename  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import ResNetForImageClassification, ResNetModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT / "homura-models" / "tests" / "golden" / "repos"
EXPECTED = ROOT / "homura-models" / "tests" / "golden" / "resnet"
HASH = "0123456789abcdef0123456789abcdef01234567"
WEIGHT_NAMES = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                "pytorch_model.bin.index.json"]

passed = 0


def block(f):
    global passed
    f()
    passed += 1


def fails(f, *exceptions):
    try:
        f()
    except exceptions or (Exception,):
        return True
    return False


# --- the cache root: constants.py evaluated in fresh interpreters with controlled environments.

def hub_cache(env):
    base = {"PATH": os.environ["PATH"], "HOME": "/home/h"}
    base.update(env)
    code = "from huggingface_hub import constants as c; print(c.HF_HUB_CACHE)"
    return subprocess.run([sys.executable, "-I", "-c", code], env=base, capture_output=True, text=True,
                          check=True).stdout.rstrip("\n")


# The table of hub_env.rs (HOME is /home/h in every case).
CACHE_DIRS = [
    ({}, "/home/h/.cache/huggingface/hub"),
    ({"XDG_CACHE_HOME": "/x"}, "/x/huggingface/hub"),
    ({"XDG_CACHE_HOME": "/x", "HF_HOME": "/hf"}, "/hf/hub"),
    ({"HF_HOME": "/hf", "HUGGINGFACE_HUB_CACHE": "/legacy"}, "/legacy"),
    ({"HF_HOME": "/hf", "HUGGINGFACE_HUB_CACHE": "/legacy", "HF_HUB_CACHE": "/new"}, "/new"),
    ({"HF_HUB_CACHE": "~/c"}, "/home/h/c"),
    ({"HF_HUB_CACHE": "$V/c", "V": "/v"}, "/v/c"),
    ({"HF_HUB_CACHE": "${V}/c", "V": "/v"}, "/v/c"),
    ({"HF_HUB_CACHE": "$UNSET/c"}, "$UNSET/c"),
    ({"HUGGINGFACE_HUB_CACHE": "~/legacy"}, "/home/h/legacy"),
    ({"HUGGINGFACE_HUB_CACHE": "$V/legacy", "V": "/v"}, "/v/legacy"),
    ({"HF_HOME": "~/hf"}, "/home/h/hf/hub"),
    ({"XDG_CACHE_HOME": "~/x"}, "/home/h/x/huggingface/hub"),
    ({"XDG_CACHE_HOME": "$V/x", "V": "/v"}, "/v/x/huggingface/hub"),
    # HF_HOME is expanded, then the hub cache path made from it is expanded again.
    ({"HF_HOME": "$A", "A": "$B", "B": "/b"}, "/b/hub"),
    # A value from a variable isn't expanded again within the same expansion.
    ({"HF_HUB_CACHE": "$A", "A": "$B", "B": "/b"}, "$B"),
    ({"HF_HUB_CACHE": ""}, ""),
    ({"HF_HOME": ""}, "hub"),
    ({"XDG_CACHE_HOME": ""}, "huggingface/hub"),
    ({"HF_HUB_CACHE": "~"}, "/home/h"),
    ({"HF_HUB_CACHE": "~user/c"}, None),  # the user database: homura leaves it as it is
    ({"HF_HUB_CACHE": "a~/b"}, "a~/b"),
    ({"HF_HUB_CACHE": "a/~"}, "a/~"),
    ({"HF_HUB_CACHE": "$1/x", "1": "/one"}, "/one/x"),
    ({"HF_HUB_CACHE": "${}/x"}, "${}/x"),
    ({"HF_HUB_CACHE": "${V/x", "V": "/v"}, "${V/x"),
    ({"HF_HUB_CACHE": "${V}/c}", "V": "/v"}, "/v/c}"),
    ({"HF_HUB_CACHE": "$V$V", "V": "/v"}, "/v/v"),
    ({"HF_HUB_CACHE": "$V_x/c", "V": "/v"}, "$V_x/c"),
    ({"HF_HUB_CACHE": "$V-x/c", "V": "/v"}, "/v-x/c"),
    ({"HF_HUB_CACHE": "${A B}/c", "A B": "/ab"}, "/ab/c"),
    ({"HF_HUB_CACHE": "/c$"}, "/c$"),
    ({"HF_HUB_CACHE": "~/c", "HOME": "/home/h/"}, "/home/h/c"),
    ({"HF_HUB_CACHE": "~", "HOME": "/"}, "/"),
    ({"HF_HUB_CACHE": "~/c", "HOME": "/"}, "/c"),
    ({"HF_HUB_CACHE": "~/c", "HOME": "/h//"}, "/h/c"),
    ({"HF_HUB_CACHE": "~/c", "HOME": ""}, "/c"),
    ({"HF_HOME": "/hf/"}, "/hf/hub"),
    ({"HF_HOME": "/hf", "HUGGINGFACE_HUB_CACHE": ""}, ""),
]


@block
def cache_root():
    for env, expected in CACHE_DIRS:
        found = hub_cache(env)
        if expected is None:
            assert found == "~user/c" or not found.startswith("~"), found
        else:
            assert found == expected, (env, found, expected)


# --- names: repository ids and file names.

REPO_IDS = {
    "microsoft/resnet-50": True, "gpt2": True, "a/b": True, "A_b.c-d/x_Y.z-9": True, "_a/b_": True,
    "a" * 96 + "/" + "b" * 96: True, "a" * 97: False, "org/" + "b" * 97: False,
    "": False, "/": False, "a/": False, "/a": False, "a/b/c": False, "a b": False,
    "-a/b": False, "a/b-": False, ".a/b": False, "a/b.": False, "a--b/c": False, "a/b..c": False,
    "a/b.git": False, "a/b.gitx": True, "a/b:c": False, "a\\b": False, "a/b\n": False,
}


@block
def repository_ids():
    for repo_id, valid in REPO_IDS.items():
        assert fails(lambda: validate_repo_id(repo_id), HFValidationError) != valid, repo_id
    # Letters and digits are ASCII only.
    assert fails(lambda: validate_repo_id("é/b"), HFValidationError)


FILENAMES = {
    "config.json": True, "sub/model.safetensors": True, "./model.safetensors": True,
    "a//b": True, "..": False, "../x": False, "a/../b": False, "a\\..\\b": False, "/etc/x": False,
    "\\x": False, "C:x": False, "a/C:x": False, "\\\\host\\share": False, "a/\\b": False,
    "1:x": False, "a:": False, "..x": True, "x..": True,
}


@block
def file_names():
    for name, valid in FILENAMES.items():
        assert fails(lambda: _validate_relative_filename(name), ValueError) != valid, name


# --- repositories on disk.

def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_cache(cache, repo_id, files, refs=None, commit=HASH):
    """A cache in huggingface_hub's layout: blobs by content, snapshot entries linking to them."""
    repo = Path(cache) / ("models--" + repo_id.replace("/", "--"))
    for name, data in files.items():
        blob = repo / "blobs" / sha(data)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = repo / "snapshots" / commit / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    for ref, content in (refs if refs is not None else {"main": commit}).items():
        path = repo / "refs" / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return repo


def write_dir(directory, files):
    for name, data in files.items():
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def repo_files(name="basic"):
    return {p.name: p.read_bytes() for p in (REPOS / name).iterdir()}


def basic_state_dict():
    return load_file(REPOS / "basic" / "model.safetensors")


def marked(mark):
    """The basic repository's weights with the classifier bias set to `mark`."""
    sd = basic_state_dict()
    sd["classifier.1.bias"] = torch.full_like(sd["classifier.1.bias"], float(mark))
    return sd


def safetensors_bytes(sd):
    with tempfile.TemporaryDirectory() as d:
        save_file(sd, f"{d}/x.safetensors", metadata={"format": "pt"})
        return Path(f"{d}/x.safetensors").read_bytes()


def pt_bytes(sd):
    with tempfile.TemporaryDirectory() as d:
        torch.save(sd, f"{d}/x.bin")
        return Path(f"{d}/x.bin").read_bytes()


def index_bytes(sd, shard, metadata=None):
    return json.dumps({"metadata": metadata or {"total_size": 0},
                       "weight_map": {k: shard for k in sd}}).encode()


def candidates(present):
    """Weight files of the basic repository, each with its own mark (1 to 4)."""
    files = {}
    if 1 in present:
        files["model.safetensors"] = safetensors_bytes(marked(1))
    if 2 in present:
        files["model.safetensors.index.json"] = index_bytes(marked(2), "model-00001-of-00001.safetensors")
        files["model-00001-of-00001.safetensors"] = safetensors_bytes(marked(2))
    if 3 in present:
        files["pytorch_model.bin"] = pt_bytes(marked(3))
    if 4 in present:
        files["pytorch_model.bin.index.json"] = index_bytes(marked(4), "pytorch_model-00001-of-00001.bin")
        files["pytorch_model-00001-of-00001.bin"] = pt_bytes(marked(4))
    return files


def config_bytes(**changes):
    config = json.loads((REPOS / "basic" / "config.json").read_text())
    for key, value in changes.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return json.dumps(config).encode()


def mark_of(model):
    return model.classifier[1].bias[0].item()


def load_local(files, **kwargs):
    with tempfile.TemporaryDirectory() as d:
        write_dir(d, files)
        return ResNetForImageClassification.from_pretrained(d, **kwargs)


def load_cached(files, revision=None, refs=None, commit=HASH, cls=ResNetForImageClassification,
                **kwargs):
    with tempfile.TemporaryDirectory() as d:
        write_cache(d, "org/mini", files, refs=refs, commit=commit)
        return cls.from_pretrained("org/mini", cache_dir=d, revision=revision, **kwargs)


# The discovery matrix of hub.rs and pretrained.rs: present files -> mark loaded, for no format,
# safetensors and PyTorch (None: an error).
DISCOVERY = [
    ({1, 2, 3, 4}, 1, 1, 3),
    ({2, 3, 4}, 2, 2, 3),
    ({1, 2}, 1, 1, None),
    ({3, 4}, 3, None, 3),
    ({2}, 2, 2, None),
    ({4}, 4, None, 4),
    ({2, 4}, 2, 2, 4),
    (set(), None, None, None),
]


@block
def weight_discovery():
    config = {"config.json": config_bytes()}
    for present, auto, safe, torch_format in DISCOVERY:
        for use, expected in [(None, auto), (True, safe), (False, torch_format)]:
            files = {**config, **candidates(present)}
            for load in (load_local, load_cached):
                if expected is None:
                    assert fails(lambda: load(files, use_safetensors=use), OSError), (present, use)
                else:
                    assert mark_of(load(files, use_safetensors=use)) == expected, (present, use, load)


@block
def a_corrupt_selected_file_is_an_error():
    for load in (load_local, load_cached):
        files = {"config.json": config_bytes(), **candidates({1, 3})}
        files["model.safetensors"] = b"\x08\x00\x00\x00\x00\x00\x00\x00{}garbage"
        assert fails(lambda: load(files))


@block
def transformers_weights():
    files = {"config.json": config_bytes(transformers_weights="sub/w.safetensors"),
             "sub/w.safetensors": safetensors_bytes(marked(5)), **candidates({1, 3})}
    for load in (load_local, load_cached):
        assert mark_of(load(files)) == 5
        # It wins over the format choice.
        assert mark_of(load(files, use_safetensors=False)) == 5
    files = {"config.json": config_bytes(transformers_weights="w.safetensors.index.json"),
             "w.safetensors.index.json": index_bytes(marked(6), "w-1.safetensors"),
             "w-1.safetensors": safetensors_bytes(marked(6)), **candidates({1})}
    for load in (load_local, load_cached):
        assert mark_of(load(files)) == 6
    for name in ["w.bin", "../w.safetensors", "/abs/w.safetensors", "missing.safetensors"]:
        files = {"config.json": config_bytes(transformers_weights=name), **candidates({1})}
        for load in (load_local, load_cached):
            assert fails(lambda: load(files)), (name, load)
    # The local branch normalizes the path before checking that it stays inside; the offline
    # cache lookup doesn't check it at all, so `..` can leave the snapshot. homura rejects every
    # `..` segment.
    files = {"config.json": config_bytes(transformers_weights="sub/../w.safetensors"),
             "w.safetensors": safetensors_bytes(marked(7)), "sub/x": b""}
    assert mark_of(load_local(files)) == 7
    assert mark_of(load_cached(files)) == 7
    with tempfile.TemporaryDirectory() as d:
        repo = write_cache(d, "org/mini", {"config.json": config_bytes(transformers_weights="../e.safetensors")})
        (repo / "snapshots" / "e.safetensors").write_bytes(safetensors_bytes(marked(8)))
        assert mark_of(ResNetForImageClassification.from_pretrained("org/mini", cache_dir=d)) == 8


@block
def shards_of_a_nested_index_are_at_the_root():
    # pretrained.rs `shards_of_a_nested_index_are_at_the_root` (R117): (index in sub/, shard at
    # the root, shard in sub/) -> the mark loaded, or None for an error.
    cases = [(True, True, False, 2), (True, False, True, None), (True, True, True, 2),
             (False, True, False, 2)]
    for nested, at_root, in_sub, expected in cases:
        index = "sub/w.safetensors.index.json" if nested else "w.safetensors.index.json"
        files = {"config.json": config_bytes(transformers_weights=index),
                 index: index_bytes(marked(2), "weights.safetensors")}
        if at_root:
            files["weights.safetensors"] = safetensors_bytes(marked(2))
        if in_sub:
            files["sub/weights.safetensors"] = safetensors_bytes(marked(9))
        for load in (load_local, load_cached):
            if expected is None:
                assert fails(lambda: load(files)), (nested, at_root, in_sub, load)
            else:
                assert mark_of(load(files)) == expected, (nested, at_root, in_sub, load)


# --- revisions in the cache, offline.

@block
def revisions():
    files = {"config.json": config_bytes(), **candidates({1})}
    other = "f" * 40
    assert mark_of(load_cached(files)) == 1
    assert mark_of(load_cached(files, revision="main")) == 1
    # A branch with `/` is a file in a subdirectory of refs.
    assert mark_of(load_cached(files, revision="feature/x", refs={"feature/x": HASH})) == 1
    # A commit hash is used as it is, without refs.
    assert mark_of(load_cached(files, revision=HASH, refs={})) == 1
    # from_pretrained resolves the revision first (HfApi.resolve_revision): the ref's content,
    # stripped of whitespace.
    for content in [HASH + "\n", HASH + "\r\n", " " + HASH, "\n" + HASH + "\n\n"]:
        assert mark_of(load_cached(files, refs={"main": content})) == 1
    # Without a ref, the revision names the snapshot directory itself, whatever it looks like;
    # a ref's content too.
    assert mark_of(load_cached(files, refs={}, commit="main")) == 1
    assert mark_of(load_cached(files, revision=HASH.upper(), refs={}, commit=HASH.upper())) == 1
    assert mark_of(load_cached(files, refs={"main": "abc"}, commit="abc")) == 1
    # A snapshot is one directory: a commit with `/` isn't found.
    assert fails(lambda: load_cached(files, refs={"main": "x/y"}, commit="x/y"))
    assert fails(lambda: load_cached(files, revision="x/y", refs={}, commit="x/y"))
    # Missing ref, missing snapshot, missing repository.
    assert fails(lambda: load_cached(files, revision="dev"))
    assert fails(lambda: load_cached(files, refs={"main": other}))
    assert fails(lambda: load_cached(files, revision=other))
    with tempfile.TemporaryDirectory() as d:
        assert fails(lambda: ResNetForImageClassification.from_pretrained("org/none", cache_dir=d))
    # A hash in upper case is another snapshot name.
    assert fails(lambda: load_cached(files, revision=HASH.upper(), refs={}))
    # Without config.json, nothing loads.
    assert fails(lambda: load_cached(candidates({1})))


# --- the base model prefix, per key.

def base_state_dict():
    return {k.removeprefix("resnet."): v for k, v in basic_state_dict().items() if k.startswith("resnet.")}


@block
def base_model_prefix():
    expected = load_file(EXPECTED / "basic.safetensors")
    pixels = load_file(EXPECTED / "inputs.safetensors")["pixel_values"]
    # ResNetModel.save_pretrained writes the keys without `resnet.` and without the head.
    model = ResNetForImageClassification.from_pretrained(REPOS / "basic")
    with tempfile.TemporaryDirectory() as d:
        model.resnet.save_pretrained(d)
        saved = load_file(f"{d}/model.safetensors")
        assert sorted(saved) == sorted(base_state_dict())
        assert json.loads(Path(f"{d}/config.json").read_text())["architectures"] == ["ResNetModel"]
    # The base model from a classification repository: `resnet.` removed, the head unexpected.
    base, info = ResNetModel.from_pretrained(REPOS / "basic", output_loading_info=True)
    assert info["missing_keys"] == set() or not info["missing_keys"]
    assert set(info["unexpected_keys"]) == {"classifier.1.weight", "classifier.1.bias"}
    with torch.no_grad():
        out = base(pixels)
    assert torch.equal(out.pooler_output, expected["eval_f32.pooler_output"])
    # The classification model from a base repository: `resnet.` added, the head missing (the
    # reference initializes it; homura returns an error).
    files = {"config.json": config_bytes(architectures=["ResNetModel"]),
             "model.safetensors": safetensors_bytes(base_state_dict())}
    model, info = load_local(files, output_loading_info=True)
    assert set(info["missing_keys"]) == {"classifier.1.weight", "classifier.1.bias"}
    assert not info["unexpected_keys"]
    # Per key: a key is renamed only if the new name is the model's.
    sd = base_state_dict()
    sd["resnet.extra"] = torch.zeros(1)
    sd["other"] = torch.zeros(1)
    sd["classifier.1.weight"] = basic_state_dict()["classifier.1.weight"]
    sd["classifier.1.bias"] = basic_state_dict()["classifier.1.bias"]
    files = {"config.json": config_bytes(), "model.safetensors": safetensors_bytes(sd)}
    model, info = load_local(files, output_loading_info=True)
    assert not info["missing_keys"]
    assert set(info["unexpected_keys"]) == {"resnet.extra", "other"}
    # A key both with and without the prefix: one of the two is the model's (no error).
    sd = basic_state_dict()
    sd["embedder.embedder.convolution.weight"] = torch.zeros_like(sd["resnet.embedder.embedder.convolution.weight"])
    files = {"config.json": config_bytes(), "model.safetensors": safetensors_bytes(sd)}
    assert not fails(lambda: load_local(files))


# --- dtype ("auto").

def float_dtypes(model):
    return {t.dtype for t in model.state_dict().values() if t.is_floating_point()}


def int_dtypes(model):
    return {t.dtype for t in model.state_dict().values() if not t.is_floating_point()}


@block
def dtypes():
    f64 = {k: (v.double() if v.is_floating_point() else v) for k, v in basic_state_dict().items()}
    cases = [
        # (config dtype fields, weights, from_pretrained dtype, model dtype)
        ({"dtype": "float32"}, basic_state_dict(), None, torch.float32),
        ({"dtype": "float64"}, basic_state_dict(), None, torch.float64),
        ({"dtype": None, "torch_dtype": "float64"}, basic_state_dict(), None, torch.float64),
        ({"dtype": "float64", "torch_dtype": "float32"}, basic_state_dict(), None, torch.float64),
        ({"dtype": None}, f64, None, torch.float64),
        ({"dtype": None}, basic_state_dict(), None, torch.float32),
        ({"dtype": "float32"}, f64, None, torch.float32),
        ({"dtype": None}, f64, torch.float32, torch.float32),
        ({"dtype": "float32"}, basic_state_dict(), torch.float64, torch.float64),
        ({"dtype": None}, f64, "auto", torch.float64),
    ]
    for fields, sd, dtype, expected in cases:
        files = {"config.json": config_bytes(**fields), "model.safetensors": safetensors_bytes(sd)}
        kwargs = {} if dtype is None else {"dtype": dtype}
        model = load_local(files, **kwargs)
        assert float_dtypes(model) == {expected}, (fields, dtype)
        assert int_dtypes(model) == {torch.int64}
        assert model.config.dtype == expected
    # Sharded, without a configuration dtype: the index's metadata, else the first shard's weights.
    for metadata, expected in [({"dtype": "float64"}, torch.float64), ({"total_size": 0}, torch.float32)]:
        files = {"config.json": config_bytes(dtype=None),
                 "model.safetensors.index.json": index_bytes(basic_state_dict(), "m-1.safetensors", metadata),
                 "m-1.safetensors": safetensors_bytes(basic_state_dict())}
        assert float_dtypes(load_local(files)) == {expected}


# --- the loaded model.

@block
def loaded_models():
    model = ResNetForImageClassification.from_pretrained(REPOS / "basic")
    assert not model.training and not any(m.training for m in model.modules())
    assert all(p.requires_grad for p in model.parameters())
    # transformers 5 doesn't read `_metadata`: a .bin file whose BatchNorm is version 2 and lacks
    # the counts still loads (the counts are reported missing and set to 0).
    sd = torch.load(REPOS / "basic" / "pytorch_model.bin", weights_only=True)
    assert sd._metadata["resnet.embedder.embedder.normalization"]["version"] == 2
    del sd["resnet.embedder.embedder.normalization.num_batches_tracked"]
    files = {"config.json": config_bytes(), "pytorch_model.bin": pt_bytes(sd)}
    model, info = load_local(files, output_loading_info=True)
    assert info["missing_keys"] == {"resnet.embedder.embedder.normalization.num_batches_tracked"} or \
        set(info["missing_keys"]) == {"resnet.embedder.embedder.normalization.num_batches_tracked"}
    assert model.resnet.embedder.embedder.normalization.num_batches_tracked.item() == 0


print(f"check_m3_hub: {passed} / {passed} blocks passed")
