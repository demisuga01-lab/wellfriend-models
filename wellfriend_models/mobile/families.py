"""Explicit mobile model-family definitions without bundled neural weights."""

from __future__ import annotations

from typing import Any

from wellfriend_models.schemas import ContractError

MOBILE_FAMILIES = {
    "document-detector-mobile": "document_segmentation",
    "document-corners-mobile": "document_corner_regression",
    "document-quality-mobile": "document_quality",
    "document-cleanup-mobile": "document_cleanup_mask",
    "document-restoration-mobile": "document_restoration",
}
VARIANTS = frozenset({"low", "mid", "high"})
REQUIRED_MOBILE_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "family",
        "variant",
        "task",
        "architecture",
        "input_size",
        "output_schema",
        "precision_target",
        "device_profile",
        "export_target",
        "expected_runtime",
        "tiling_policy",
        "training_smoke",
        "evaluation",
        "export",
        "quantization",
        "benchmark",
        "limitations",
        "status",
    }
)


def mobile_family_variant(family: str, variant: str) -> dict[str, Any]:
    """Return an experimental architecture contract for one profile, not a trained model."""
    if family not in MOBILE_FAMILIES or variant not in VARIANTS:
        raise ContractError("unsupported mobile model family or variant")
    dimension = {"low": 256, "mid": 512, "high": 768}[variant]
    precision = {"low": "int8", "mid": "fp16", "high": "fp16"}[variant]
    tiling = "overlap_tile" if variant == "low" else "pyramid_lowres_predict_highres_apply"
    return {
        "family": family,
        "variant": variant,
        "task": MOBILE_FAMILIES[family],
        "architecture": f"experimental-{family}-{variant}-contract",
        "input_size": [dimension, dimension],
        "output_schema": "task-specific-placeholder-output",
        "precision_target": precision,
        "device_class": variant,
        "export_target": "onnx-optional",
        "expected_runtime": ["cpu", "xnnpack"],
        "tiling_mode": tiling,
        "limitations": ["synthetic only", "no weights included", "not production ready"],
        "status": "experimental",
    }


def validate_mobile_candidate_config(config: dict[str, Any]) -> None:
    """Validate a combined training/evaluation/export mobile candidate plan."""
    if set(config) != REQUIRED_MOBILE_CONFIG_FIELDS:
        raise ContractError("mobile candidate config fields do not match schema")
    if config["schema_version"] != 1 or config["family"] not in MOBILE_FAMILIES:
        raise ContractError("mobile candidate config schema_version or family is invalid")
    if config["variant"] not in VARIANTS or config["task"] != MOBILE_FAMILIES[config["family"]]:
        raise ContractError("mobile candidate config variant or task is invalid")
    if config["status"] != "experimental":
        raise ContractError("MP6 mobile candidate configs must remain experimental")
    if not isinstance(config["input_size"], list) or len(config["input_size"]) != 2:
        raise ContractError("mobile candidate input_size must be a pair")
    if not all(
        isinstance(config[field], dict)
        for field in ("training_smoke", "evaluation", "export", "quantization", "benchmark")
    ):
        raise ContractError("mobile candidate flow sections must be objects")
