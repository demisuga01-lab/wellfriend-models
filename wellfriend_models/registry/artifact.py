"""Validate a model artifact before a production runtime is permitted to consume it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wellfriend_models.schemas import ContractError, validate_model_components

COMPONENT_FILES = {
    "manifest": "manifest.json",
    "preprocess": "preprocess.json",
    "postprocess": "postprocess.json",
    "labels": "labels.json",
    "checksums": "checksums.json",
    "metrics": "metrics.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path.name}: {error}") from error
    if not isinstance(document, dict):
        raise ContractError(f"{path.name} must contain a JSON object")
    return document


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact_directory(
    directory: Path, *, allow_placeholder: bool = False
) -> dict[str, Any]:
    """Validate structure, contract documents, and listed checksums for one artifact directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ContractError(f"artifact directory does not exist: {directory}")
    components = {
        name: _read_json(directory / filename) for name, filename in COMPONENT_FILES.items()
    }
    manifest = components["manifest"]
    placeholder = manifest.get("status") == "placeholder"
    validate_model_components(components, production=not allow_placeholder)
    if placeholder and not allow_placeholder:
        raise ContractError("placeholder artifact requires explicit --allow-placeholder validation")
    if not (directory / "LICENSES" / "README.md").is_file():
        raise ContractError("artifact requires LICENSES/README.md")
    if not (directory / "README.md").is_file():
        raise ContractError("artifact requires README.md")
    model_path = directory / "model.onnx"
    if not placeholder and not model_path.is_file():
        raise ContractError("released artifact requires model.onnx")
    for relative_path, expected in components["checksums"]["files"].items():
        path = directory / relative_path
        if not path.is_file():
            raise ContractError(f"checksum references a missing file: {relative_path}")
        if _checksum(path) != expected:
            raise ContractError(f"checksum mismatch: {relative_path}")
    return {
        "model_id": manifest["model_id"],
        "status": manifest["status"],
        "production_ready": manifest["production_ready"],
        "directory": str(directory),
    }


def _placeholder_documents(*, family: str, task: str) -> dict[str, dict[str, Any]]:
    safe_id = family.replace("_", "-")
    return {
        "manifest": {
            "schema_version": 1,
            "model_id": f"wellfriend.{safe_id}.placeholder",
            "model_name": family,
            "version": "0.1.0",
            "status": "placeholder",
            "production_ready": False,
            "domain": "document",
            "task": task,
            "architecture": "contract-only-placeholder",
            "input_spec": {"schema": "image[1,C,H,W]", "pixel_format": "Gray8"},
            "output_spec": {"schema": "task-specific-placeholder-output"},
            "preprocess_ref": "preprocess.json",
            "postprocess_ref": "postprocess.json",
            "training_data_refs": [],
            "evaluation_data_refs": ["synthetic-only"],
            "license": {"code": "Apache-2.0", "notice": "No implementation or weights included."},
            "weights_license": "not-included",
            "intended_runtime": ["future-wellfriend-perception-adapter"],
            "device_class": "unknown",
            "precision": "not-applicable",
            "dynamic_shape_support": False,
            "metrics_ref": "metrics.json",
            "hashes_ref": "checksums.json",
            "limitations": ["placeholder only", "no model weights", "not production ready"],
            "safety_notes": ["production validation must reject this placeholder"],
            "provenance": {"source": "MP5 registry scaffold", "weights": "not included"},
        },
        "preprocess": {"schema_version": 1, "steps": [], "status": "placeholder"},
        "postprocess": {"schema_version": 1, "steps": [], "status": "placeholder"},
        "labels": {"schema_version": 1, "labels": [], "status": "placeholder"},
        "checksums": {"schema_version": 1, "files": {}, "status": "placeholder"},
        "metrics": {
            "schema_version": 1,
            "metrics": {},
            "status": "placeholder",
            "data": "synthetic-only",
        },
    }


def write_placeholder_artifact(directory: Path, *, family: str, task: str) -> Path:
    """Create a fully explicit no-weights placeholder artifact for a registry family."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    documents = _placeholder_documents(family=family, task=task)
    for component, filename in COMPONENT_FILES.items():
        (directory / filename).write_text(
            json.dumps(documents[component], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    licenses = directory / "LICENSES"
    licenses.mkdir(exist_ok=True)
    (licenses / "README.md").write_text(
        "No model weights or third-party code are included in this placeholder.\n", encoding="utf-8"
    )
    description = (
        "# Placeholder artifact\n\nThis directory exists to validate the artifact contract. "
        "It contains no weights and is not production-ready.\n"
    )
    (directory / "README.md").write_text(
        description,
        encoding="utf-8",
    )
    return directory
