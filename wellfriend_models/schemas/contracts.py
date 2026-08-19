"""Strict standard-library contract validation for datasets, experiments, and artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MODEL_TASKS = frozenset(
    {
        "document_segmentation",
        "document_corner_regression",
        "document_quality",
        "document_restoration",
        "document_cleanup_mask",
        "ocr_detection",
        "ocr_recognition",
        "layout_detection",
    }
)
ANNOTATION_TYPES = frozenset(
    {
        "quad",
        "polygon",
        "mask",
        "bounding_box",
        "keypoints",
        "quality_labels",
        "restoration_pair",
        "ocr_text",
        "layout_regions",
        "cleanup_mask",
    }
)
DATASET_LICENSE_STATUSES = frozenset(
    {
        "redistributable",
        "download_script_only",
        "local_private",
        "synthetic",
        "research_only",
        "commercially_unsafe",
        "unknown",
    }
)
REQUIRED_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "relative_path",
        "sha256",
        "width",
        "height",
        "channels",
        "pixel_format",
        "domain",
        "split",
        "annotations",
        "source",
        "license_ref",
    }
)
REQUIRED_DATASET_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "name",
        "version",
        "description",
        "domain",
        "tasks",
        "license",
        "source_url",
        "citation",
        "allowed_uses",
        "redistribution",
        "provenance",
        "splits",
        "samples",
        "annotations",
        "transforms",
        "hashes",
        "warnings",
    }
)
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_name",
        "version",
        "status",
        "production_ready",
        "domain",
        "task",
        "architecture",
        "input_spec",
        "output_spec",
        "preprocess_ref",
        "postprocess_ref",
        "training_data_refs",
        "evaluation_data_refs",
        "license",
        "weights_license",
        "intended_runtime",
        "device_class",
        "precision",
        "dynamic_shape_support",
        "metrics_ref",
        "hashes_ref",
        "limitations",
        "safety_notes",
        "provenance",
    }
)
REQUIRED_EXPERIMENT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "seed",
        "task",
        "model",
        "dataset",
        "splits",
        "transforms",
        "losses",
        "optimizer",
        "scheduler",
        "batch_size",
        "epochs",
        "device",
        "precision",
        "logging",
        "checkpoints",
        "export",
        "reproducibility",
    }
)


class ContractError(ValueError):
    """Raised when a durable Wellfriend data contract is invalid."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _require_fields(value: Mapping[str, Any], required: frozenset[str], name: str) -> None:
    missing = sorted(required.difference(value))
    if missing:
        raise ContractError(f"{name} missing required fields: {', '.join(missing)}")


def _reject_unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unexpected = sorted(set(value).difference(allowed))
    if unexpected:
        raise ContractError(f"{name} has unsupported fields: {', '.join(unexpected)}")


def _require_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value.lower())
    ):
        raise ContractError(f"{name} must be a lowercase-or-uppercase SHA-256 hex string")


def validate_dataset_manifest(manifest: Mapping[str, Any], *, production_use: bool = False) -> None:
    """Validate the complete DatasetManifest contract and license-use boundary."""
    _require_fields(manifest, REQUIRED_DATASET_FIELDS, "dataset manifest")
    _reject_unknown_fields(manifest, REQUIRED_DATASET_FIELDS, "dataset manifest")
    if manifest["schema_version"] != 1:
        raise ContractError("unsupported dataset manifest schema_version")
    if (
        not isinstance(manifest["tasks"], Sequence)
        or isinstance(manifest["tasks"], str)
        or not manifest["tasks"]
    ):
        raise ContractError("dataset tasks must be a non-empty list")
    if any(task not in MODEL_TASKS for task in manifest["tasks"]):
        raise ContractError("dataset tasks include an unsupported model task")
    license_info = _require_mapping(manifest["license"], "dataset license")
    for field in ("name", "status", "reference"):
        if not license_info.get(field):
            raise ContractError(f"dataset license requires {field}")
    status = license_info["status"]
    if status not in DATASET_LICENSE_STATUSES:
        raise ContractError("dataset license status is not recognized")
    if status == "unknown":
        raise ContractError("unknown dataset license is blocked")
    if production_use and status in {"research_only", "commercially_unsafe", "local_private"}:
        raise ContractError(f"dataset license status {status} is blocked from production artifacts")
    if not isinstance(manifest["samples"], list):
        raise ContractError("dataset samples must be a list")
    splits = _require_mapping(manifest["splits"], "dataset splits")
    split_ids = set(splits)
    if not split_ids:
        raise ContractError("dataset splits must not be empty")
    declared_annotations = set(manifest["annotations"])
    if not declared_annotations.issubset(ANNOTATION_TYPES):
        raise ContractError("dataset annotations include an unsupported annotation type")
    for sample in manifest["samples"]:
        sample_mapping = _require_mapping(sample, "dataset sample")
        _require_fields(sample_mapping, REQUIRED_SAMPLE_FIELDS, "dataset sample")
        _reject_unknown_fields(sample_mapping, REQUIRED_SAMPLE_FIELDS, "dataset sample")
        _require_sha256(sample_mapping["sha256"], "sample sha256")
        if not all(
            isinstance(sample_mapping[field], int) and sample_mapping[field] > 0
            for field in ("width", "height", "channels")
        ):
            raise ContractError("sample width, height, and channels must be positive integers")
        if sample_mapping["split"] not in split_ids:
            raise ContractError("sample split is missing from dataset splits")
        annotations = _require_mapping(sample_mapping["annotations"], "sample annotations")
        if not set(annotations).issubset(ANNOTATION_TYPES):
            raise ContractError("sample uses an unsupported annotation type")


def validate_model_components(
    components: Mapping[str, Mapping[str, Any]], *, production: bool
) -> None:
    """Validate parsed artifact JSON components before file/checksum validation."""
    required = {"manifest", "preprocess", "postprocess", "labels", "checksums", "metrics"}
    missing = sorted(required.difference(components))
    if missing:
        raise ContractError(f"artifact component documents missing: {', '.join(missing)}")
    manifest = _require_mapping(components["manifest"], "artifact manifest")
    _require_fields(manifest, REQUIRED_MANIFEST_FIELDS, "artifact manifest")
    _reject_unknown_fields(manifest, REQUIRED_MANIFEST_FIELDS, "artifact manifest")
    if manifest["schema_version"] != 1:
        raise ContractError("unsupported model manifest schema_version")
    if manifest["task"] not in MODEL_TASKS:
        raise ContractError("artifact manifest task is unsupported")
    if not isinstance(manifest["input_spec"], Mapping) or not isinstance(
        manifest["output_spec"], Mapping
    ):
        raise ContractError("artifact input_spec and output_spec must be objects")
    for spec_name in ("input_spec", "output_spec"):
        spec = _require_mapping(manifest[spec_name], spec_name)
        if not spec.get("schema"):
            raise ContractError(f"{spec_name} requires schema")
    status = manifest["status"]
    ready = manifest["production_ready"]
    if status == "placeholder":
        if ready:
            raise ContractError("placeholder artifact cannot be production_ready")
        if production:
            raise ContractError("placeholder artifact is not a production artifact")
    elif status != "released":
        raise ContractError("artifact status must be placeholder or released")
    elif not isinstance(ready, bool) or not ready:
        raise ContractError("released artifact must set production_ready true")
    if not isinstance(components["preprocess"].get("steps"), list):
        raise ContractError("preprocess steps must be a list")
    if not isinstance(components["postprocess"].get("steps"), list):
        raise ContractError("postprocess steps must be a list")
    if not isinstance(components["labels"].get("labels"), list):
        raise ContractError("labels labels must be a list")
    if not isinstance(components["metrics"].get("metrics"), Mapping):
        raise ContractError("metrics metrics must be an object")
    if not isinstance(components["checksums"].get("files"), Mapping):
        raise ContractError("checksums files must be an object")


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate a reproducible experiment configuration without requiring an ML framework."""
    _require_fields(config, REQUIRED_EXPERIMENT_FIELDS, "experiment config")
    _reject_unknown_fields(config, REQUIRED_EXPERIMENT_FIELDS, "experiment config")
    if config["schema_version"] != 1:
        raise ContractError("unsupported experiment config schema_version")
    if config["task"] not in MODEL_TASKS:
        raise ContractError("experiment task is unsupported")
    if not isinstance(config["seed"], int):
        raise ContractError("experiment seed must be an integer")
    if not isinstance(config["batch_size"], int) or config["batch_size"] < 1:
        raise ContractError("experiment batch_size must be positive")
    if not isinstance(config["epochs"], int) or config["epochs"] < 1:
        raise ContractError("experiment epochs must be positive")
    for field in (
        "model",
        "dataset",
        "optimizer",
        "scheduler",
        "logging",
        "checkpoints",
        "export",
        "reproducibility",
    ):
        _require_mapping(config[field], f"experiment {field}")
