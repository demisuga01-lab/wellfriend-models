"""Dataset manifest loading and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wellfriend_models.schemas import validate_dataset_manifest


def load_dataset_manifest(path: Path, *, production_use: bool = False) -> dict[str, Any]:
    """Read and strictly validate a DatasetManifest from JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("dataset manifest root must be an object")
    validate_dataset_manifest(value, production_use=production_use)
    return value


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """Return a stable hash for experiment provenance and reproducible comparisons."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
