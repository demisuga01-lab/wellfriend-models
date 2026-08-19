"""Synthetic-only ScanBench-style model report construction and validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wellfriend_models.metrics import binary_iou, corner_mae, latency_summary
from wellfriend_models.mobile import load_device_profile
from wellfriend_models.models import (
    DocumentCornerBaseline,
    DocumentQualityBaseline,
    DocumentSegmentationBaseline,
)
from wellfriend_models.registry import validate_artifact_directory
from wellfriend_models.schemas import ContractError
from wellfriend_models.synthetic import SyntheticDocumentGenerator

SCANBENCH_CATEGORIES = (
    "easy-page",
    "rotated-page",
    "perspective-page",
    "low-contrast-page",
    "shadow-page",
    "glare-page",
    "blurred-page",
    "receipt-like-page",
    "cut-off-page",
    "multiple-distractors",
    "no-document",
)
REQUIRED_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "repo_sha",
        "artifact_id",
        "artifact_hash",
        "dataset_manifest_hash",
        "device_profile",
        "metrics",
        "limitations",
        "timestamp",
        "environment",
        "license_provenance_safety",
    }
)


def validate_scanbench_report(report: dict[str, Any]) -> None:
    """Validate portable report structure without turning synthetic observations into claims."""
    if set(report) != REQUIRED_REPORT_FIELDS:
        raise ContractError("ScanBench model report fields do not match schema")
    if report["schema_version"] != 1 or not all(
        _sha(report[name]) for name in ("artifact_hash", "dataset_manifest_hash")
    ):
        raise ContractError("ScanBench report schema or hashes are invalid")
    if not isinstance(report["metrics"], dict) or not isinstance(report["limitations"], list):
        raise ContractError("ScanBench report metrics and limitations are invalid")
    if report["license_provenance_safety"] not in {
        "safe-synthetic-no-weights",
        "audit-gated",
        "blocked",
    }:
        raise ContractError("ScanBench report license/provenance safety is invalid")


def benchmark_artifact(config: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Benchmark an experimental artifact against deterministic generated fixtures."""
    profile = load_device_profile(root / config["device_profile"])
    artifact_path = root / config["artifact_path"]
    artifact = validate_artifact_directory(artifact_path, allow_nonproduction=True)
    manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
    generator = SyntheticDocumentGenerator(width=48, height=36, seed=int(config["seed"]))
    cases = _fixture_cases(config["categories"])
    latencies: list[float] = []
    outcomes: dict[str, float] = {}
    for category, fixture in cases.items():
        sample = generator.generate(fixture)
        start = time.perf_counter()
        outcomes[category] = _evaluate_task(manifest["task"], sample)
        latencies.append(time.perf_counter() - start)
    artifact_hash = _directory_hash(artifact_path)
    dataset_hash = hashlib.sha256(
        json.dumps(
            {"seed": config["seed"], "categories": config["categories"]}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    artifact_size = sum(path.stat().st_size for path in artifact_path.rglob("*") if path.is_file())
    metrics = {
        "scenario_metrics": outcomes,
        "scenario_count": len(outcomes),
        "artifact_size_bytes": artifact_size,
        "parameter_count": {"value": 0, "status": "no weights included"},
        "flops_macs": {"status": "placeholder", "reason": "no graph or weights included"},
        "latency": latency_summary(latencies),
        "memory": {"status": "placeholder", "reason": "CPU smoke only"},
        "export_validity": "metadata_validated",
        "quantization_delta": {"status": "not measured"},
        "distillation_delta": {"status": "not measured"},
        "device_profile_fit": _device_profile_fit(profile, artifact_size, manifest),
    }
    report = {
        "schema_version": 1,
        "repo_sha": _repo_sha(root),
        "artifact_id": artifact["model_id"],
        "artifact_hash": artifact_hash,
        "dataset_manifest_hash": dataset_hash,
        "device_profile": profile["device_class"],
        "metrics": metrics,
        "limitations": [
            "synthetic only",
            "CPU smoke only",
            "not real-device latency",
            "not real model accuracy",
            "not a production claim",
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": {
            "runtime": "stdlib baseline",
            "weights_included": manifest.get("weights_included", False),
        },
        "license_provenance_safety": "safe-synthetic-no-weights",
    }
    validate_scanbench_report(report)
    return report


def write_scanbench_report(report: dict[str, Any], output_directory: Path) -> tuple[Path, Path]:
    """Write the requested machine-readable JSON and concise Markdown companion report."""
    validate_scanbench_report(report)
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "scanbench-model-report.json"
    markdown_path = output_directory / "scanbench-model-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    markdown_path.write_text(
        "# ScanBench model smoke report\n\n"
        f"- Artifact: `{report['artifact_id']}`\n"
        f"- Device profile: `{report['device_profile']}`\n"
        f"- Scenarios: {metrics['scenario_count']}\n"
        f"- Artifact bytes: {metrics['artifact_size_bytes']}\n"
        "- Scope: synthetic-only CPU smoke; not real-device latency, real model accuracy, "
        "or a production claim.\n",
        encoding="utf-8",
    )
    return json_path, markdown_path


def _fixture_cases(categories: list[str]) -> dict[str, str]:
    mapping = {
        "easy-page": "white_page_dark_background",
        "rotated-page": "rotated_quadrilateral",
        "perspective-page": "perspective_quadrilateral",
        "low-contrast-page": "low_contrast_receipt",
        "shadow-page": "shadow_gradient",
        "glare-page": "glare_patch",
        "blurred-page": "blurred_page",
        "receipt-like-page": "low_contrast_receipt",
        "cut-off-page": "partial_cut_off_page",
        "multiple-distractors": "multiple_rectangle_distractors",
        "no-document": "white_page_light_background",
    }
    if not set(categories).issubset(mapping):
        raise ContractError("ScanBench config contains unknown categories")
    return {category: mapping[category] for category in categories}


def _evaluate_task(task: str, sample: Any) -> float:
    if task == "document_segmentation":
        model = DocumentSegmentationBaseline()
        return binary_iou(model.binary_mask(sample), [value >= 128 for value in sample.mask])
    if task == "document_corner_regression":
        return corner_mae(DocumentCornerBaseline().predict(sample)["corners"], sample.quad)
    if task == "document_quality":
        prediction = DocumentQualityBaseline().predict(sample)
        return sum(
            (prediction[name] - sample.quality_labels[name]) ** 2 for name in prediction
        ) / len(prediction)
    if task in {"document_cleanup_mask", "document_restoration"}:
        return 0.0
    raise ContractError("unsupported ScanBench model task")


def _device_profile_fit(
    profile: dict[str, Any], artifact_size: int, manifest: dict[str, Any]
) -> dict[str, Any]:
    max_bytes = int(float(profile["max_model_size_mb"]) * 1024 * 1024)
    precision_ok = (
        manifest["precision"] in profile["precision"] or manifest["precision"] == "not-applicable"
    )
    return {
        "fits_size_target": artifact_size <= max_bytes,
        "precision_target_compatible": precision_ok,
    }


def _directory_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _repo_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )
