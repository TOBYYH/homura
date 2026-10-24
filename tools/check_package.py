"""Checks the packages of the workspace before a release, as `cargo publish --workspace` will
upload them:

1. `cargo package --workspace --locked` packages homura-derive and homura and builds each package;
   homura is built against the packaged homura-derive, which Cargo serves from a temporary
   registry since it isn't on crates.io yet.
2. The packaged homura depends on exactly the workspace version of homura-derive (`=0.y.z`,
   without a path), and neither package contains the tests or the tools.
3. The tests of both unpacked packages pass (homura's unit tests and doctests, homura-derive's
   unit tests), with homura built against the unpacked homura-derive.
4. A crate outside the workspace that depends on the unpacked homura under another name
   (`hm = { package = "homura", ... }`) derives `Module` with `#[module(crate = "hm")]` and an
   `extra_repr`, builds and runs.

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
CRATES = ["homura-derive", "homura"]

EXTERNAL_MANIFEST = """\
[package]
name = "external"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
hm = {{ package = "homura", path = {homura} }}

[patch.crates-io]
homura-derive = {{ path = {derive} }}
"""

EXTERNAL_MAIN = """\
use hm::Tensor;
use hm::nn::{self, Module};

#[derive(nn::Module)]
#[module(crate = "hm", extra_repr = Self::describe)]
struct Net {
    conv: nn::Conv2d,
    bn: nn::BatchNorm2d,
    #[module(rename = "fc")]
    head: nn::Linear,
    #[module(non_persistent)]
    scale: Tensor,
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

fn main() {
    let net = Net {
        conv: nn::Conv2d::new(3, 4, 3).padding(1),
        bn: nn::BatchNorm2d::new(4),
        head: nn::Linear::new(64, 2),
        scale: hm::tensor(2.0),
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
    let text = net.display().to_string();
    assert!(text.starts_with("Net(\n  classes=2\n  (conv): Conv2d(3, 4, kernel_size=(3, 3)"), "{text}");
    assert!(text.contains("\n  (fc): Linear(in_features=64, out_features=2, bias=True)\n)"), "{text}");

    net.forward(&hm::randn([2, 3, 4, 4])).sum().backward();
    assert!(net.parameters().iter().all(|p| p.grad().is_some()));
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


def check_manifest(unpacked, version):
    manifest = tomllib.loads((unpacked / "Cargo.toml").read_text())
    derive = manifest["dependencies"]["homura-derive"]
    if derive != {"version": f"={version}"}:
        sys.exit(f"the packaged homura depends on homura-derive as {derive}, expected version ={version}")
    print(f"the packaged homura depends on homura-derive ={version}")


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
        check_manifest(homura, version)

        run(["cargo", "test", "--locked", *target], derive)
        # The lock file of the package names the derive crate of the temporary registry, which the
        # patch replaces, so it can't be used as is.
        patch = f"patch.crates-io.homura-derive.path={toml_string(derive)}"
        run(["cargo", "test", "--config", patch, *target], homura)

        external = tmp / "external"
        (external / "src").mkdir(parents=True)
        manifest = EXTERNAL_MANIFEST.format(homura=toml_string(homura), derive=toml_string(derive))
        (external / "Cargo.toml").write_text(manifest)
        (external / "src" / "main.rs").write_text(EXTERNAL_MAIN)
        run(["cargo", "run", *target], external)
    print(f"the packages of homura {version} are ready to publish")


if __name__ == "__main__":
    main()
