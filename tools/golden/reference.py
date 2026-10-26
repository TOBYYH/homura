"""The PyTorch (and transformers) versions that golden data is generated with.

Gradient boundary semantics (ties, NaN, clamp bounds, degrees of freedom) differ between PyTorch
releases, so every generator calls `check()` first and refuses to run with any other build.
Upgrading the reference means regenerating all fixtures and reviewing the documented semantics.
"""

import torch

TORCH_VERSION = "2.14.1+cpu"
TORCH_GIT = "5c4886908584029761b579af026dcfb627c84070"


def check():
    """Exits unless the running PyTorch is exactly the pinned reference build."""
    found = (torch.__version__, torch.version.git_version)
    if found != (TORCH_VERSION, TORCH_GIT):
        raise SystemExit(
            f"golden data must be generated with PyTorch {TORCH_VERSION} ({TORCH_GIT}), "
            f"found {found[0]} ({found[1]})"
        )


TRANSFORMERS_VERSION = "5.20.2"


def check_transformers():
    """Exits unless the running transformers is the pinned reference release (after `check()`)."""
    check()
    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(
            f"golden data must be generated with transformers {TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )


def metadata():
    """The safetensors `__metadata__` entries recording the reference build."""
    return {"torch_version": TORCH_VERSION, "torch_git": TORCH_GIT}
