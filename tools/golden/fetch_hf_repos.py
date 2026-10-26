"""Downloads the two repositories of the M3 acceptance (step 4b) into target/hf-cache/.

microsoft/resnet-18 and microsoft/resnet-50, at the commits below, in the Hugging Face cache layout
(`models--microsoft--resnet-18/{blobs,refs,snapshots}`), with `refs/main` set to the commit so that
`Revision::main()` resolves to it offline. Only the files that homura reads are fetched (no TF /
Flax weights, about 300 MB in all), and each one's SHA-256 is checked against the values recorded
when the plan was made (M3.3). A run with the files in place downloads nothing.

Usage: python tools/golden/fetch_hf_repos.py (needs huggingface_hub and network access); then
HF_HUB_CACHE=$PWD/target/hf-cache cargo test --release -p homura-models --test hf_resnet -- --ignored
"""

import hashlib
from pathlib import Path

import reference

reference.check_hub()

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "target" / "hf-cache"

REPOS = {
    "microsoft/resnet-18": (
        "65a5785d9156231087c481e0c7dd33a5ff6f7e3e",
        {
            "config.json": "11a01bcc873444bb433140e5f30f3687356641ed923266d738eef1c3438f6d22",
            "preprocessor_config.json":
                "fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741",
            "model.safetensors":
                "1cf00ee468998c23d084361b02cffadabacb5074105564c596b8079f77ecb126",
            "pytorch_model.bin":
                "420fc189b95a1d31fa95e1cf494c25e3cead05f88480b5578c5efcebeb0db57a",
        },
    ),
    "microsoft/resnet-50": (
        "34c2154c194f829b11125337b98c8f5f9965ff19",
        {
            "config.json": "a1be0f56d516d0b55f9844ce0fbfe5fb359195cba6c6c102d41ae6c202c84d5e",
            "preprocessor_config.json":
                "fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741",
            "model.safetensors":
                "9c6061af1f450bb0847e529fd742aa5066017be379c71bbf5546b198e5b13a1e",
            "pytorch_model.bin":
                "ff8163a1323333126706d649ce73ecd76e45d241b42d623dea6c723690cafe07",
        },
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for repo_id, (commit, files) in REPOS.items():
        snapshot = Path(snapshot_download(
            repo_id, revision=commit, cache_dir=CACHE, allow_patterns=list(files),
        ))
        assert snapshot == CACHE / f"models--{repo_id.replace('/', '--')}" / "snapshots" / commit
        found = sorted(p.name for p in snapshot.iterdir())
        assert found == sorted(files), f"{repo_id}: {found}"
        for name, digest in files.items():
            got = sha256(snapshot / name)
            assert got == digest, f"{repo_id} {name}: sha256 {got}, expected {digest}"
        refs = snapshot.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(commit)
        print(f"{repo_id} {commit}: {', '.join(files)} ok")


if __name__ == "__main__":
    main()
