#!/usr/bin/env python3
"""Runs csrc/tests/host_failures.cpp: the ROCm layer's host failure paths (cleanup that fails,
device memory that runs out, the HIP errors failures leave behind, host memory that runs out for
good) against a stand-in HIP, so no ROCm or GPU is needed.

Compiles csrc/runtime.hip and csrc/memory.hip with the host C++ compiler ($CXX, else c++) and
runs every mode in its own process: an abort fails that mode only. Exit status 0 if all pass.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"
OUT = ROOT / "target" / "rocm-host-failures"


def main() -> int:
    cxx = os.environ.get("CXX") or shutil.which("c++")
    if not cxx:
        print("no C++ compiler: set CXX")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    exe = OUT / "host_failures"
    cmd = [
        cxx, "-std=c++17", "-O1", "-g", "-Wall", "-Werror",
        "-I", str(CSRC / "tests" / "fake_hip"), "-I", str(CSRC),
        '-DHMR_OFFLOAD_ARCHS="gfx1201"',
        str(CSRC / "tests" / "host_failures.cpp"), "-o", str(exe),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL compile: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
        return 1
    modes = subprocess.run([str(exe), "--list"], capture_output=True, text=True,
                           check=True).stdout.split()
    failures = 0
    for mode in modes:
        r = subprocess.run([str(exe), mode], capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {mode}" + ("" if ok else f" (exit {r.returncode})"))
        if not ok:
            failures += 1
            print((r.stdout + r.stderr).rstrip())
    print(f"{len(modes) - failures} of {len(modes)} modes passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
