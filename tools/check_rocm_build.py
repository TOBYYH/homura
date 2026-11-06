#!/usr/bin/env python3
"""Checks the ROCm build configuration (build.rs) end to end. Local only: needs ROCm, and a
gfx1201 GPU for the last check.

1. Unsupported `HOMURA_OFFLOAD_ARCH` values fail the build with an explanation (checked with
   `HOMURA_ROCM_CHECK_ONLY=1`, which validates the list before skipping the native build).
2. gfx1200, which is compile-only, builds: alone and together with gfx1201. A list with spaces
   and duplicates builds for the clean list, which the library reports (compared with a list
   written in this script, not with build.rs's own).
3. A library built for gfx1200 alone refuses the gfx1201 GPU at context creation, with a message
   that says how to rebuild, instead of failing at the first kernel launch.

usage: tools/check_rocm_build.py
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "target" / "rocm-build-check"

REJECTED = {
    "gfx1100": "gfx1100 is not supported; supported architectures: gfx1201, gfx1200",
    "gfx1151": "gfx1151 is not supported; supported architectures: gfx1201, gfx1200",
    "gfx1202": "gfx1202 is not supported; supported architectures: gfx1201, gfx1200",
    "gfx942": "gfx942 is not supported (wave64 GPUs are not supported: homura's kernels are "
    "written for wave32)",
    "gfx1201;gfx90a": "gfx90a is not supported (wave64",
    "": "HOMURA_OFFLOAD_ARCH is empty",
    " ; ": "HOMURA_OFFLOAD_ARCH is empty",
}


def cargo(args, env, target):
    full_env = {**os.environ, **env, "CARGO_TARGET_DIR": str(TARGET / target)}
    return subprocess.run(
        ["cargo", *args], cwd=ROOT, env=full_env, capture_output=True, text=True
    )


def main() -> int:
    failures = []
    for arch, message in REJECTED.items():
        r = cargo(
            ["check", "--lib", "--features", "rocm", "--quiet"],
            {"HOMURA_OFFLOAD_ARCH": arch, "HOMURA_ROCM_CHECK_ONLY": "1"},
            "check-only",
        )
        ok = r.returncode != 0 and message in r.stderr
        print(f"{'ok  ' if ok else 'FAIL'} rejects HOMURA_OFFLOAD_ARCH={arch!r}")
        if not ok:
            failures.append(f"{arch!r}: exit {r.returncode}\n{r.stderr[-2000:]}")
    for arch in ["gfx1200", "gfx1200;gfx1201"]:
        r = cargo(
            ["build", "--lib", "--features", "rocm", "--quiet"],
            {"HOMURA_OFFLOAD_ARCH": arch},
            arch.replace(";", "-"),
        )
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} builds HOMURA_OFFLOAD_ARCH={arch!r}")
        if not ok:
            failures.append(f"{arch!r}: exit {r.returncode}\n{r.stderr[-2000:]}")
    # Spaces and duplicates are dropped, keeping the first occurrences in order: the library is
    # built for (and reports) the clean list (review round 118, R138). The expected list is
    # written here, not taken from build.rs (review round 119, R141).
    for arch, expected, target in [
        (" gfx1201 ; gfx1201 ", "gfx1201", "spaced"),
        ("gfx1201;gfx1200;gfx1201", "gfx1201;gfx1200", "repeated"),
    ]:
        r = cargo(
            ["test", "--lib", "--features", "rocm", "--quiet", "--",
             "backend::rocm::tests::the_library_was_built_for_the_configured_architectures"],
            {"HOMURA_OFFLOAD_ARCH": arch, "HOMURA_TEST_EXPECTED_ARCHS": expected},
            target,
        )
        ok = r.returncode == 0 and "1 passed" in r.stdout
        print(f"{'ok  ' if ok else 'FAIL'} HOMURA_OFFLOAD_ARCH={arch!r} builds and reports {expected!r}")
        if not ok:
            failures.append(f"{arch!r}: exit {r.returncode}\n{(r.stdout + r.stderr)[-3000:]}")
    # The device API test asks for a context; with only gfx1200 code it must be refused.
    r = cargo(
        ["test", "--features", "rocm", "--test", "rocm_device", "--quiet", "--",
         "devices_are_counted_and_described"],
        {"HOMURA_OFFLOAD_ARCH": "gfx1200"},
        "gfx1200",
    )
    expected = ("rocm:0 is a gfx1201 GPU, but homura was built for gfx1200; rebuild it with "
                "HOMURA_OFFLOAD_ARCH set to include gfx1201")
    ok = r.returncode != 0 and expected in r.stdout + r.stderr
    print(f"{'ok  ' if ok else 'FAIL'} a gfx1200 build refuses the gfx1201 GPU")
    if not ok:
        failures.append(f"gfx1200 on gfx1201: exit {r.returncode}\n{(r.stdout + r.stderr)[-3000:]}")
    for f in failures:
        print(f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
