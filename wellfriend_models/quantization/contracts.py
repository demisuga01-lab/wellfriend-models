"""Metadata-first quantization contracts; optional ONNX execution is deliberately external."""

from __future__ import annotations

from typing import Any

from wellfriend_models.schemas import ContractError

PRECISION_MODES = frozenset({"fp32", "fp16", "dynamic_int8", "static_int8", "qat_placeholder"})
REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "target_precision",
        "calibration",
        "unsupported_ops",
        "validation_plan",
    }
)
REQUIRED_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "original_artifact",
        "quantized_artifact",
        "precision",
        "operator_coverage",
        "size_before_bytes",
        "size_after_bytes",
        "metric_delta",
        "known_unsupported_ops",
        "calibration_source",
        "validation_status",
        "production_ready",
    }
)


def validate_quantization_config(config: dict[str, Any]) -> None:
    """Validate planning inputs without requiring an ONNX package in base CI."""
    _exact(config, REQUIRED_CONFIG_FIELDS, "quantization config")
    if config["schema_version"] != 1 or config["target_precision"] not in PRECISION_MODES:
        raise ContractError("quantization config schema_version or precision is invalid")
    if not isinstance(config["artifact"], dict) or not config["artifact"].get("model_id"):
        raise ContractError("quantization config requires an artifact model_id")
    if not isinstance(config["calibration"], dict) or not isinstance(
        config["validation_plan"], dict
    ):
        raise ContractError("quantization calibration and validation plan must be objects")
    if not isinstance(config["unsupported_ops"], list):
        raise ContractError("quantization unsupported_ops must be a list")


def validate_quantization_report(report: dict[str, Any]) -> None:
    """Validate size/accuracy/runtime evidence and prevent metadata from faking a release."""
    _exact(report, REQUIRED_REPORT_FIELDS, "quantization report")
    if report["schema_version"] != 1 or report["precision"] not in PRECISION_MODES:
        raise ContractError("quantization report schema_version or precision is invalid")
    if report["production_ready"]:
        raise ContractError(
            "quantization report cannot independently mark an artifact production-ready"
        )
    for field in ("size_before_bytes", "size_after_bytes"):
        if not isinstance(report[field], int) or report[field] < 0:
            raise ContractError(f"quantization report {field} must be non-negative")
    if not isinstance(report["metric_delta"], dict) or not isinstance(
        report["operator_coverage"], dict
    ):
        raise ContractError("quantization report requires metric delta and operator coverage")
    if not isinstance(report["known_unsupported_ops"], list):
        raise ContractError("quantization report unsupported ops must be a list")
    if report["validation_status"] not in {
        "metadata_validated",
        "onnx_validated",
        "runtime_validated",
        "blocked",
    }:
        raise ContractError("quantization validation_status is invalid")


def build_quantization_report(
    config: dict[str, Any], *, size_before: int, size_after: int
) -> dict[str, Any]:
    """Build an honest metadata-only report for a no-weights experimental artifact."""
    validate_quantization_config(config)
    report = {
        "schema_version": 1,
        "original_artifact": config["artifact"],
        "quantized_artifact": {**config["artifact"], "variant": config["target_precision"]},
        "precision": config["target_precision"],
        "operator_coverage": {"status": "metadata-only", "covered": [], "total": 0},
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
        "metric_delta": {},
        "known_unsupported_ops": config["unsupported_ops"],
        "calibration_source": config["calibration"],
        "validation_status": "metadata_validated",
        "production_ready": False,
    }
    validate_quantization_report(report)
    return report


def optional_dynamic_int8_quantize(input_model: str, output_model: str) -> dict[str, str]:
    """Use ONNX Runtime only when explicitly installed; base CI receives a clear blocked result."""
    try:
        from onnxruntime.quantization import (  # type: ignore[import-not-found]
            QuantType,
            quantize_dynamic,
        )
    except ImportError:
        return {"status": "blocked", "reason": "onnxruntime optional dependency is not installed"}
    quantize_dynamic(input_model, output_model, weight_type=QuantType.QInt8)
    return {"status": "quantized", "output_model": output_model}


def _exact(value: dict[str, Any], fields: frozenset[str], name: str) -> None:
    if fields.difference(value) or set(value).difference(fields):
        raise ContractError(f"{name} fields do not match schema")
