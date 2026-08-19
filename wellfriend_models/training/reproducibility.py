"""Reproducibility evidence written beside every MP5 baseline experiment."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def set_deterministic_seed(seed: int) -> None:
    """Set the standard-library random seed and deterministic process hint."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def environment_summary(root: Path) -> dict[str, Any]:
    """Collect portable environment/version evidence without a hardware-specific claim."""
    dependencies = {}
    for package in ("setuptools", "pytest", "ruff", "torch", "onnx"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "git_commit": _git_sha(root),
        "dependencies": dependencies,
    }
