"""Validated engineering targets for device-class-specific model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wellfriend_models.schemas import ContractError

DEVICE_CLASSES = frozenset({"low", "mid", "high", "server", "web", "unknown"})
PRECISIONS = frozenset({"fp32", "fp16", "int8", "dynamic_int8"})
RUNTIME_TARGETS = frozenset({"cpu", "xnnpack", "nnapi", "coreml", "webgpu", "wasm", "server"})
RESOLUTION_POLICIES = frozenset({"native", "downsample", "downsample_or_tile", "tile", "pyramid"})
FALLBACK_STRATEGIES = frozenset(
    {"classical_or_light_model", "light_model", "server_if_allowed", "no_model"}
)
REQUIRED_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "device_class",
        "precision",
        "max_model_size_mb",
        "max_latency_ms_placeholder",
        "max_memory_mb_placeholder",
        "input_resolution_policy",
        "tiling_policy",
        "runtime_targets",
        "fallback_strategy",
    }
)


def validate_device_profile(profile: dict[str, Any]) -> None:
    """Validate an engineering target without treating it as measured device performance."""
    missing = REQUIRED_PROFILE_FIELDS.difference(profile)
    unknown = set(profile).difference(REQUIRED_PROFILE_FIELDS)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported: {', '.join(sorted(unknown))}")
        raise ContractError(f"invalid device profile ({'; '.join(detail)})")
    if profile["schema_version"] != 1 or profile["device_class"] not in DEVICE_CLASSES:
        raise ContractError("device profile schema_version or device_class is invalid")
    precision = profile["precision"]
    if not isinstance(precision, list) or not precision or not set(precision).issubset(PRECISIONS):
        raise ContractError("device profile precision must contain supported values")
    for field in ("max_model_size_mb", "max_latency_ms_placeholder", "max_memory_mb_placeholder"):
        if not isinstance(profile[field], (int, float)) or profile[field] <= 0:
            raise ContractError(f"device profile {field} must be positive")
    if profile["input_resolution_policy"] not in RESOLUTION_POLICIES:
        raise ContractError("device profile input_resolution_policy is invalid")
    if profile["tiling_policy"] not in {
        "no_tiling",
        "fixed_tile",
        "overlap_tile",
        "pyramid_lowres_predict_highres_apply",
        "full_page_downsample",
    }:
        raise ContractError("device profile tiling_policy is invalid")
    targets = profile["runtime_targets"]
    if not isinstance(targets, list) or not targets or not set(targets).issubset(RUNTIME_TARGETS):
        raise ContractError("device profile runtime_targets are invalid")
    if profile["fallback_strategy"] not in FALLBACK_STRATEGIES:
        raise ContractError("device profile fallback_strategy is invalid")


def load_device_profile(path: Path) -> dict[str, Any]:
    """Load a checked-in JSON profile and validate it before planning an artifact."""
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load device profile: {error}") from error
    if not isinstance(profile, dict):
        raise ContractError("device profile must be a JSON object")
    validate_device_profile(profile)
    return profile
