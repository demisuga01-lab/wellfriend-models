"""Strict, dependency-free validators for exported production model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FILES = {
    "model.onnx",
    "manifest.json",
    "preprocess.json",
    "postprocess.json",
    "labels.json",
    "checksums.json",
    "metrics.json",
    "LICENSES",
    "README.md",
}
REQUIRED_MANIFEST_FIELDS = {
    "model_name",
    "version",
    "task",
    "domain",
    "input_shape",
    "dynamic_shape_support",
    "pixel_format",
    "preprocessing",
    "postprocessing",
    "expected_output_schema",
    "training_dataset_references",
    "license_notes",
    "intended_runtime",
    "device_class",
    "metrics",
    "hashes",
}
REQUIRED_DATASET_FIELDS = {"schema_version", "name", "version", "license", "samples", "splits", "provenance"}


class ContractError(ValueError):
    """Raised when a registry document or release artifact violates its contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required at {path}")
    return value


def validate_manifest(manifest: dict[str, Any], *, allow_placeholder: bool = False) -> None:
    if manifest.get("status") == "placeholder":
        if allow_placeholder and {"model_name", "task", "domain"}.issubset(manifest):
            return
        raise ContractError("placeholder manifests are not production artifacts")
    missing = REQUIRED_MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise ContractError(f"manifest missing fields: {', '.join(sorted(missing))}")
    if manifest["domain"] == "medical" and not manifest.get("research_only"):
        raise ContractError("medical artifacts must explicitly set research_only")
    if not isinstance(manifest["input_shape"], list) or not manifest["input_shape"]:
        raise ContractError("input_shape must be a non-empty list")


def validate_dataset_manifest(manifest: dict[str, Any]) -> None:
    missing = REQUIRED_DATASET_FIELDS.difference(manifest)
    if missing:
        raise ContractError(f"dataset manifest missing fields: {', '.join(sorted(missing))}")
    if manifest["schema_version"] != 1:
        raise ContractError("unsupported dataset manifest schema_version")
    if not isinstance(manifest["samples"], list):
        raise ContractError("samples must be a list")
    if not isinstance(manifest["license"], dict) or not manifest["license"].get("name"):
        raise ContractError("dataset license requires a name")


def validate_artifact_directory(directory: Path, *, allow_placeholder: bool = False) -> None:
    manifest = _read_json(directory / "manifest.json")
    validate_manifest(manifest, allow_placeholder=allow_placeholder)
    if manifest.get("status") == "placeholder":
        return
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (directory / name).exists()]
    if missing:
        raise ContractError(f"artifact directory missing: {', '.join(sorted(missing))}")
    checksums = _read_json(directory / "checksums.json")
    files = checksums.get("files", {})
    if not isinstance(files, dict):
        raise ContractError("checksums.json files must be an object")
    for relative_path, expected in files.items():
        payload = (directory / relative_path).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ContractError(f"checksum mismatch: {relative_path}")

