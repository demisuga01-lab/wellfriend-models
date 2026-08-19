"""Portable model-side tiling and coordinate-remapping policy validation."""

from __future__ import annotations

from typing import Any

from wellfriend_models.schemas import ContractError

TILING_MODES = frozenset(
    {
        "no_tiling",
        "fixed_tile",
        "overlap_tile",
        "pyramid_lowres_predict_highres_apply",
        "full_page_downsample",
    }
)
BLEND_POLICIES = frozenset({"none", "average", "feather", "max_confidence"})
REMAP_POLICIES = frozenset({"identity", "tile_offset", "scale_and_offset", "pyramid_scale"})
REQUIRED_TILING_FIELDS = frozenset(
    {
        "mode",
        "input_resolution",
        "max_resolution",
        "tile_size",
        "overlap",
        "blend_policy",
        "coordinate_remap_policy",
    }
)


def validate_tiling_policy(policy: dict[str, Any]) -> None:
    """Validate a declared tiling plan; execution remains a future runtime responsibility."""
    missing = REQUIRED_TILING_FIELDS.difference(policy)
    unknown = set(policy).difference(REQUIRED_TILING_FIELDS)
    if missing or unknown:
        raise ContractError("tiling policy fields do not match the schema")
    if policy["mode"] not in TILING_MODES:
        raise ContractError("tiling policy mode is invalid")
    for field in ("input_resolution", "max_resolution", "tile_size"):
        value = policy[field]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(v, int) and v > 0 for v in value)
        ):
            raise ContractError(f"tiling policy {field} must be two positive integers")
    if not isinstance(policy["overlap"], int) or policy["overlap"] < 0:
        raise ContractError("tiling policy overlap must be non-negative")
    if policy["overlap"] >= min(policy["tile_size"]):
        raise ContractError("tiling policy overlap must be smaller than tile_size")
    if policy["blend_policy"] not in BLEND_POLICIES:
        raise ContractError("tiling policy blend_policy is invalid")
    if policy["coordinate_remap_policy"] not in REMAP_POLICIES:
        raise ContractError("tiling policy coordinate_remap_policy is invalid")
