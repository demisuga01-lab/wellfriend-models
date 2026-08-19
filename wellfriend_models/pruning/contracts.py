"""Validate pruning evidence while keeping real tensor pruning optional."""

from __future__ import annotations

from typing import Any

from wellfriend_models.schemas import ContractError

PRUNING_MODES = frozenset({"unstructured", "structured", "channel_placeholder"})
REQUIRED_CONFIG_FIELDS = frozenset(
    {"schema_version", "artifact", "mode", "target_sparsity", "metric_guard"}
)
REQUIRED_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "mode",
        "target_sparsity",
        "achieved_sparsity",
        "metric_delta",
        "status",
    }
)


def validate_pruning_config(config: dict[str, Any]) -> None:
    """Validate a pruning plan before an optional tensor backend performs it."""
    _exact(config, REQUIRED_CONFIG_FIELDS, "pruning config")
    if config["schema_version"] != 1 or config["mode"] not in PRUNING_MODES:
        raise ContractError("pruning config schema_version or mode is invalid")
    if (
        not isinstance(config["target_sparsity"], (float, int))
        or not 0 <= config["target_sparsity"] < 1
    ):
        raise ContractError("pruning target_sparsity must be in [0, 1)")
    if not isinstance(config["artifact"], dict) or not isinstance(config["metric_guard"], dict):
        raise ContractError("pruning artifact and metric guard must be objects")


def validate_pruning_report(report: dict[str, Any]) -> None:
    """Require observed sparsity and metric delta, even for metadata-only experiments."""
    _exact(report, REQUIRED_REPORT_FIELDS, "pruning report")
    if report["schema_version"] != 1 or report["mode"] not in PRUNING_MODES:
        raise ContractError("pruning report schema_version or mode is invalid")
    for field in ("target_sparsity", "achieved_sparsity"):
        if not isinstance(report[field], (float, int)) or not 0 <= report[field] < 1:
            raise ContractError(f"pruning report {field} must be in [0, 1)")
    if not isinstance(report["metric_delta"], dict):
        raise ContractError("pruning report requires metric_delta")
    if report["status"] not in {"metadata_only", "tensor_smoke", "blocked"}:
        raise ContractError("pruning report status is invalid")


def build_pruning_report(config: dict[str, Any]) -> dict[str, Any]:
    """Build a transparent metadata-only report when no optional tensor backend is installed."""
    validate_pruning_config(config)
    report = {
        "schema_version": 1,
        "artifact": config["artifact"],
        "mode": config["mode"],
        "target_sparsity": config["target_sparsity"],
        "achieved_sparsity": 0.0,
        "metric_delta": {},
        "status": "metadata_only",
    }
    validate_pruning_report(report)
    return report


def _exact(value: dict[str, Any], fields: frozenset[str], name: str) -> None:
    if fields.difference(value) or set(value).difference(fields):
        raise ContractError(f"{name} fields do not match schema")
