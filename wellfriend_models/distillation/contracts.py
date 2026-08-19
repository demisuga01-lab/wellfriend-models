"""Dependency-free contracts for teacher/student comparisons and reproducibility evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from wellfriend_models.schemas import ContractError

TEACHER_TYPES = frozenset(
    {
        "torch_model",
        "classical_processor",
        "artifact",
        "cached_predictions",
        "external_audit_gated",
    }
)
STUDENT_TASKS = frozenset(
    {
        "document_segmentation",
        "document_corner_regression",
        "document_quality",
        "document_cleanup_mask",
        "document_restoration",
    }
)
REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "task",
        "teacher",
        "student",
        "dataset_manifest_hash",
        "losses",
        "reproducibility",
    }
)
REQUIRED_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "task",
        "teacher_manifest",
        "student_manifest",
        "dataset_manifest_hash",
        "loss_curves",
        "metrics",
        "comparison",
        "reproducibility",
        "timestamp",
    }
)


def validate_distillation_config(config: dict[str, Any]) -> None:
    """Validate teacher/student configuration before a training runtime is selected."""
    _exact_fields(config, REQUIRED_CONFIG_FIELDS, "distillation config")
    if config["schema_version"] != 1 or config["task"] not in STUDENT_TASKS:
        raise ContractError("distillation config schema_version or task is invalid")
    teacher = _mapping(config["teacher"], "teacher")
    student = _mapping(config["student"], "student")
    if teacher.get("type") not in TEACHER_TYPES or not teacher.get("id"):
        raise ContractError("distillation teacher type or id is invalid")
    if teacher["type"] == "external_audit_gated" and teacher.get("audit_status") != "audit-gated":
        raise ContractError("external teacher must retain audit-gated status")
    if not student.get("model_id") or student.get("task") != config["task"]:
        raise ContractError("distillation student must identify the matching task")
    if not _sha(config["dataset_manifest_hash"]):
        raise ContractError("distillation dataset_manifest_hash must be SHA-256")
    if not isinstance(config["losses"], list) or not config["losses"]:
        raise ContractError("distillation losses must be non-empty")
    if not isinstance(config["reproducibility"], dict):
        raise ContractError("distillation reproducibility must be an object")


def validate_distillation_report(report: dict[str, Any]) -> None:
    """Validate reusable comparison evidence emitted by a distillation experiment."""
    _exact_fields(report, REQUIRED_REPORT_FIELDS, "distillation report")
    if report["schema_version"] != 1 or report["task"] not in STUDENT_TASKS:
        raise ContractError("distillation report schema_version or task is invalid")
    if not _sha(report["dataset_manifest_hash"]):
        raise ContractError("distillation report requires dataset manifest hash")
    if not isinstance(report["loss_curves"], dict) or not isinstance(report["metrics"], dict):
        raise ContractError("distillation report requires loss curves and metrics")
    comparison = _mapping(report["comparison"], "comparison")
    if not {"teacher_metrics", "student_metrics", "delta"}.issubset(comparison):
        raise ContractError("distillation report comparison is incomplete")
    teacher = _mapping(report["teacher_manifest"], "teacher_manifest")
    student = _mapping(report["student_manifest"], "student_manifest")
    if teacher.get("type") == "external_audit_gated" and student.get("production_ready"):
        raise ContractError("audit-gated external teacher cannot promote its student to production")


def build_distillation_report(config: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    """Build a tiny deterministic report for smoke experiments, never a production promotion."""
    validate_distillation_config(config)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "task": config["task"],
        "teacher_manifest": config["teacher"],
        "student_manifest": {**config["student"], "production_ready": False},
        "dataset_manifest_hash": config["dataset_manifest_hash"],
        "loss_curves": {"distillation": [float(metrics.get("distillation_loss", 0.0))]},
        "metrics": metrics,
        "comparison": {
            "teacher_metrics": metrics.get("teacher", {}),
            "student_metrics": metrics.get("student", {}),
            "delta": metrics.get("delta", {}),
        },
        "reproducibility": config["reproducibility"],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    validate_distillation_report(report)
    return report


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _exact_fields(value: dict[str, Any], fields: frozenset[str], name: str) -> None:
    missing, unknown = fields.difference(value), set(value).difference(fields)
    if missing or unknown:
        raise ContractError(f"{name} fields do not match schema")


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )
