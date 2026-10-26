"""Generates homura-models/tests/golden/hf_resnet_logits.safetensors, what transformers computes
with microsoft/resnet-18 and microsoft/resnet-50 (M3 step 4b), for homura-models/tests/hf_resnet.rs.

The repositories are read offline from target/hf-cache/ (tools/golden/fetch_hf_repos.py), at
`main`, which that script points at the pinned commits. The input is 2 x 3 x 224 x 224 values
`((i·7919 + 13) mod 1021) / 1024 − 0.5`, exact in F32 and F64 (the modulus doesn't divide the
224 · 224 values of a channel, so the channels and the images differ). The file's `__metadata__`
records the FNV-1a 64 digest and the length of all its bytes (little-endian) in both dtypes, so
that the test checks the input it computes byte for byte, and the commits.

- `<name>.f32.logits`: the logits of the model as loaded (F32, the configuration's `torch_dtype`),
  in eval mode, as `from_pretrained` returns it.
- `<name>.f64.logits`: the same with `dtype=torch.float64` and the input in F64.

Asserted on the way: `main` is the pinned commit; the model loads from both `model.safetensors`
and `pytorch_model.bin` with no missing, unexpected or mismatched keys, in eval mode; both files
give the same logits bit for bit; the F32 and F64 top-5 classes agree. The largest F32 error
against F64 is printed, for the tolerance of the test.

A file is only rewritten when its content changes. Usage: python tools/golden/gen_m3_hf_resnet.py
(needs torch, transformers and the downloaded repositories).
"""

import json
import os
import struct
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "target" / "hf-cache"
OUT = ROOT / "homura-models" / "tests" / "golden" / "hf_resnet_logits.safetensors"

# Before transformers reads them.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = str(CACHE)

import reference  # noqa: E402

reference.check_hub()

import torch  # noqa: E402
from safetensors.torch import save as save_safetensors  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import ResNetForImageClassification  # noqa: E402

from fetch_hf_repos import REPOS  # noqa: E402

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data):
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def pixel_values():
    i = torch.arange(2 * 3 * 224 * 224, dtype=torch.int64)
    x = ((i * 7919 + 13) % 1021).to(torch.float64) / 1024 - 0.5
    assert torch.equal(x.to(torch.float32).to(torch.float64), x)
    x = x.view(2, 3, 224, 224)
    assert not torch.equal(x[0], x[1]) and not torch.equal(x[0, 0], x[0, 1])
    return x


def little_endian(t):
    assert t.is_contiguous()
    return t.numpy().astype(t.numpy().dtype.newbyteorder("<")).tobytes()


def safetensors_content(data):
    """The metadata and tensors of a safetensors file, whatever the order of its header."""
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    body = data[8 + n:]
    tensors = {k: (v["dtype"], v["shape"], body[v["data_offsets"][0]:v["data_offsets"][1]])
               for k, v in header.items() if k != "__metadata__"}
    return header.get("__metadata__"), tensors


def load(repo_id, use_safetensors, dtype):
    model, info = ResNetForImageClassification.from_pretrained(
        repo_id, revision="main", use_safetensors=use_safetensors, dtype=dtype,
        output_loading_info=True,
    )
    assert all(not v for v in info.values()), f"{repo_id}: {info}"
    assert not model.training and all(not m.training for m in model.modules())
    want = torch.float32 if dtype == "auto" else dtype
    assert all(p.dtype == want for p in model.parameters())
    return model


def main():
    x64 = pixel_values().contiguous()
    x32 = x64.to(torch.float32)
    metadata = dict(reference.metadata())
    for name, x in [("f32", x32), ("f64", x64)]:
        data = little_endian(x)
        metadata[f"input.{name}.bytes"] = str(len(data))
        metadata[f"input.{name}.fnv1a64"] = f"{fnv1a64(data):016x}"
    tensors = {}
    for repo_id, (commit, _) in REPOS.items():
        name = repo_id.split("/")[1]
        # `main` resolves offline to the pinned commit (transformers 5 doesn't record it).
        snapshot = snapshot_download(repo_id, revision="main", local_files_only=True,
                                     allow_patterns=["config.json"])
        assert Path(snapshot).name == commit, snapshot
        metadata[f"{name}.commit"] = commit
        with torch.no_grad():
            start = time.perf_counter()
            st = load(repo_id, True, "auto")(x32).logits
            seconds = time.perf_counter() - start
            bin_ = load(repo_id, False, "auto")(x32).logits
            assert torch.equal(st, bin_), f"{repo_id}: safetensors and .bin logits differ"
            f64 = load(repo_id, True, torch.float64)(x64).logits
        assert st.shape == (2, 1000) and f64.dtype == torch.float64
        top32 = st.topk(5).indices
        assert torch.equal(top32, f64.topk(5).indices), f"{repo_id}: top-5 differ"
        error = ((st.double() - f64).abs() / (f64.abs() + 1)).max().item()
        print(f"{repo_id}: F32 against F64 {error:.3e}; top-5 {top32.tolist()}; "
              f"load + forward {seconds:.2f} s")
        tensors[f"{name}.f32.logits"] = st.contiguous()
        tensors[f"{name}.f64.logits"] = f64.contiguous()
    data = save_safetensors(tensors, metadata=metadata)
    if not (OUT.exists() and safetensors_content(OUT.read_bytes()) == safetensors_content(data)):
        OUT.write_bytes(data)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size} bytes")
    print(json.dumps({k: v for k, v in metadata.items() if k.startswith("input.")}))


main()
