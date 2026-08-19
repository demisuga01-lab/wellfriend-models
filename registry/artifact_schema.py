"""Legacy import compatibility; use :mod:`wellfriend_models.registry` for MP5 validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wellfriend_models.registry.artifact import validate_artifact_directory
from wellfriend_models.schemas import (
    ContractError,
    validate_dataset_manifest,
    validate_model_components,
)


def validate_manifest(manifest: dict[str, Any], *, allow_placeholder: bool = False) -> None:
    """Preserve the old parsed-manifest API using the stronger MP5 component contract."""
    components = {
        "manifest": manifest,
        "preprocess": {"steps": []},
        "postprocess": {"steps": []},
        "labels": {"labels": []},
        "checksums": {"files": {}},
        "metrics": {"metrics": {}},
    }
    validate_model_components(components, production=not allow_placeholder)


__all__ = [
    "ContractError",
    "Path",
    "validate_artifact_directory",
    "validate_dataset_manifest",
    "validate_manifest",
]
