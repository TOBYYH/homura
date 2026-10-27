"""Checks the packages of the workspace before a release, as `cargo publish --workspace` will
upload them:

1. `cargo package --workspace --locked` packages homura-derive, homura and homura-models and
   builds each package; homura is built against the packaged homura-derive and homura-models
   against the packaged homura, which Cargo serves from a temporary registry since the versions
   aren't on crates.io yet.
2. The packaged homura depends on exactly the workspace version of homura-derive, and the
   packaged homura-models on exactly that of homura (`=0.y.z`, without a path), and no package
   contains the tests or the tools.
3. The tests of the unpacked packages pass (homura's unit tests and doctests, homura-derive's
   unit tests, homura-models' doctests), each built against the unpacked crates it depends on.
4. A crate outside the workspace that depends on the unpacked homura and homura-models under
   other names (`hm = { package = "homura", ... }`, `hmm = { package = "homura-models", ... }`)
   derives `Module` with `#[module(crate = "hm")]` and an `extra_repr`, a version and an
   `added_in` buffer, writes a ResNet repository and loads it back with `from_pretrained` from
   safetensors and PyTorch's format, builds and runs.

The working tree must be clean, as for `cargo publish` (the tool never passes `--allow-dirty`).
The packages are unpacked into a temporary directory outside the workspace; the build output goes
to target/check-package. Every failing command makes the tool fail.

Usage: python3 tools/check_package.py (needs Cargo 1.90 or later and network access to
crates.io; no Python packages besides the standard library).
"""

import json
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "target" / "check-package"
CRATES = ["homura-derive", "homura", "homura-models"]

EXTERNAL_MANIFEST = """\
[package]
name = "external"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
hm = {{ package = "homura", path = {homura} }}
hmm = {{ package = "homura-models", path = {models} }}

[patch.crates-io]
homura = {{ path = {homura} }}
homura-derive = {{ path = {derive} }}
"""

EXTERNAL_MAIN = """\
use hm::Tensor;
use hm::nn::{self, Module};

#[derive(nn::Module)]
#[module(crate = "hm", extra_repr = Self::describe, version = 2)]
struct Net {
    conv: nn::Conv2d,
    bn: nn::BatchNorm2d,
    #[module(rename = "fc")]
    head: nn::Linear,
    #[module(non_persistent)]
    scale: Tensor,
    #[module(added_in = 2)]
    steps: Option<Tensor>,
}

impl Net {
    fn describe(&self) -> String {
        "classes=2".into()
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.bn.forward(&self.conv.forward(x)).relu();
        self.head.forward(&x.flatten(1, -1)) * &self.scale
    }
}

/// A ResNet written as a local repository and loaded back from both weight formats.
fn models() {
    use hm::nn::Module;
    use hmm::hub::{Repo, WeightFormat};
    use hmm::pretrained::{FromPretrained, PretrainedOptions};
    use hmm::resnet::{ResNetConfig, ResNetForImageClassification};

    let json = r#"{"model_type": "resnet", "layer_type": "bottleneck", "embedding_size": 8,
        "hidden_sizes": [8, 16], "depths": [1, 1], "num_labels": 3}"#;
    let config = ResNetConfig::from_json(json).unwrap();
    let model = ResNetForImageClassification::new(&config);
    model.eval();
    std::fs::create_dir_all("repo").unwrap();
    std::fs::write("repo/config.json", json).unwrap();
    hm::save(&model.state_dict(), "repo/model.safetensors").unwrap();
    hm::save(&model.state_dict(), "repo/pytorch_model.bin").unwrap();
    let repo = Repo::local("repo").unwrap();
    let x = hm::randn([2, 3, 32, 32]);
    let expected = hm::with_no_grad(|| model.forward(&x)).to_vec::<f32>();
    for format in [WeightFormat::Safetensors, WeightFormat::PyTorch] {
        let options = PretrainedOptions::default().format(format);
        let loaded = ResNetForImageClassification::from_pretrained(&repo, options).unwrap();
        assert_eq!(hm::with_no_grad(|| loaded.forward(&x)).to_vec::<f32>(), expected);
    }
}

fn main() {
    models();
    let net = Net {
        conv: nn::Conv2d::new(3, 4, 3).padding(1),
        bn: nn::BatchNorm2d::new(4),
        head: nn::Linear::new(64, 2),
        scale: hm::tensor(2.0),
        steps: None,
    };
    let keys: Vec<&str> = vec![
        "conv.weight",
        "conv.bias",
        "bn.weight",
        "bn.bias",
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
        "fc.weight",
        "fc.bias",
    ];
    assert_eq!(net.state_dict().keys().collect::<Vec<_>>(), keys);
    assert_eq!(net.named_buffers().len(), 4);
    assert_eq!(net.num_parameters(), 108 + 4 + 8 + 130);
    assert_eq!((net.version(), net.added_in("steps")), (2, Some(2)));
    let mut sd = net.state_dict();
    assert_eq!(sd.module_version(""), Some(2));
    assert_eq!(sd.module_version("bn"), Some(2));
    sd.remove("bn.num_batches_tracked");
    sd.set_module_version("bn", 1);
    net.load_state_dict(&sd).unwrap();
    let text = net.display().to_string();
    assert!(text.starts_with("Net(\n  classes=2\n  (conv): Conv2d(3, 4, kernel_size=(3, 3)"), "{text}");
    assert!(text.contains("\n  (fc): Linear(in_features=64, out_features=2, bias=True)\n)"), "{text}");

    net.forward(&hm::randn([2, 3, 4, 4])).sum().backward();
    assert!(net.parameters().iter().all(|p| p.grad().is_some()));
    let shards = hm::io::ShardOptions::default().max_shard_size(256);
    hm::io::save_sharded(&net.state_dict(), "shards", shards).unwrap();
    let loaded = hm::load("shards/model.safetensors.index.json").unwrap();
    assert_eq!(loaded.keys().collect::<Vec<_>>(), keys);
    net.load_state_dict(&loaded).unwrap();
    // SAFETY: nothing modifies the shards while the program runs.
    let mapped = unsafe { hm::io::load_mmap("shards/model.safetensors.index.json") }.unwrap();
    assert_eq!(mapped.keys().collect::<Vec<_>>(), keys);
    net.load_state_dict_with(&mapped, hm::nn::LoadOptions::default().assign(true)).unwrap();
    drop(mapped);
    net.eval();
    assert_eq!(net.forward(&hm::randn([1, 3, 4, 4])).shape(), [1, 2]);
    println!("external crate: ok");
}
"""


def run(args, cwd):
    print("$", " ".join(str(a) for a in args), f"(in {cwd})", flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def toml_string(path):
    # A JSON string of a path is a valid TOML basic string.
    return json.dumps(str(path))


def check_cargo():
    version = subprocess.run(["cargo", "--version"], check=True, capture_output=True, text=True).stdout
    match = re.match(r"cargo (\d+)\.(\d+)", version)
    if not match or (int(match[1]), int(match[2])) < (1, 90):
        sys.exit(f"Cargo 1.90 or later is needed to package a workspace, found: {version.strip()}")


def check_manifest(unpacked, dependency, version):
    manifest = tomllib.loads((unpacked / "Cargo.toml").read_text())
    found = manifest["dependencies"][dependency]
    name = manifest["package"]["name"]
    if found != {"version": f"={version}"}:
        sys.exit(f"the packaged {name} depends on {dependency} as {found}, expected version ={version}")
    print(f"the packaged {name} depends on {dependency} ={version}")


def check_contents(crate):
    with tarfile.open(crate) as tar:
        names = [n.split("/", 1)[1] for n in tar.getnames() if "/" in n]
    unexpected = [n for n in names if n.startswith(("tests/", "tools/"))]
    if unexpected:
        sys.exit(f"{crate.name} contains {unexpected}")


def main():
    check_cargo()
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    target = ["--target-dir", str(TARGET)]
    run(["cargo", "package", "--workspace", "--locked", *target], ROOT)

    crates = {name: TARGET / "package" / f"{name}-{version}.crate" for name in CRATES}
    for crate in crates.values():
        check_contents(crate)
    with tempfile.TemporaryDirectory(prefix="homura-check-package-") as tmp:
        tmp = Path(tmp)
        for crate in crates.values():
            with tarfile.open(crate) as tar:
                tar.extractall(tmp, filter="data")
        derive = tmp / f"homura-derive-{version}"
        homura = tmp / f"homura-{version}"
        models = tmp / f"homura-models-{version}"
        check_manifest(homura, "homura-derive", version)
        check_manifest(models, "homura", version)

        run(["cargo", "test", "--locked", *target], derive)
        # The lock file of the package names the derive crate of the temporary registry, which the
        # patch replaces, so it can't be used as is.
        patch = f"patch.crates-io.homura-derive.path={toml_string(derive)}"
        run(["cargo", "test", "--config", patch, *target], homura)
        patch_homura = f"patch.crates-io.homura.path={toml_string(homura)}"
        run(["cargo", "test", "--config", patch, "--config", patch_homura, *target], models)

        external = tmp / "external"
        (external / "src").mkdir(parents=True)
        manifest = EXTERNAL_MANIFEST.format(
            homura=toml_string(homura), derive=toml_string(derive), models=toml_string(models)
        )
        (external / "Cargo.toml").write_text(manifest)
        (external / "src" / "main.rs").write_text(EXTERNAL_MAIN)
        run(["cargo", "run", *target], external)
    print(f"the packages of homura {version} are ready to publish")


if __name__ == "__main__":
    main()
