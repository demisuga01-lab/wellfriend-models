"""Materialize checked-in experimental mobile metadata without emitting any model weights."""

from __future__ import annotations

import json
from pathlib import Path

from wellfriend_models.mobile.families import mobile_family_variant
from wellfriend_models.mobile.profiles import load_device_profile
from wellfriend_models.registry import write_mobile_experimental_artifact

FAMILIES = {
    "document-detector": ("document-detector-mobile", "document_segmentation"),
    "document-corners": ("document-corners-mobile", "document_corner_regression"),
    "document-quality": ("document-quality-mobile", "document_quality"),
    "document-cleanup": ("document-cleanup-mobile", "document_cleanup_mask"),
    "document-restoration": ("document-restoration-mobile", "document_restoration"),
}
VARIANTS = ("low", "mid", "high")


def _tiling_policy(variant: str) -> dict[str, object]:
    size = {"low": 256, "mid": 512, "high": 768}[variant]
    mode = "overlap_tile" if variant == "low" else "pyramid_lowres_predict_highres_apply"
    return {
        "mode": mode,
        "input_resolution": [size, size],
        "max_resolution": [2048, 2048],
        "tile_size": [size, size],
        "overlap": 32 if variant == "low" else 0,
        "blend_policy": "feather" if variant == "low" else "none",
        "coordinate_remap_policy": "tile_offset" if variant == "low" else "pyramid_scale",
    }


def main() -> int:
    """Write metadata-only registry entries and index records for mobile experiment planning."""
    root = Path(__file__).resolve().parents[2]
    registry = root / "registry"
    entries = [
        {"family": family, "path": f"{family}/placeholder", "production_ready": False}
        for family in (*FAMILIES, "ocr")
    ]
    for directory_family, (mobile_family, task) in FAMILIES.items():
        for variant in VARIANTS:
            profile = load_device_profile(root / "configs" / "device-profiles" / f"{variant}.json")
            name = f"{variant}-mobile-experimental"
            write_mobile_experimental_artifact(
                registry / directory_family / name,
                family=mobile_family,
                task=task,
                variant=variant,
                profile=profile,
                tiling_policy=_tiling_policy(variant),
            )
            entries.append(
                {
                    "family": directory_family,
                    "path": f"{directory_family}/{name}",
                    "production_ready": False,
                }
            )
            definition = mobile_family_variant(mobile_family, variant)
            config = {
                "schema_version": 1,
                "family": mobile_family,
                "variant": variant,
                "task": task,
                "architecture": definition["architecture"],
                "input_size": definition["input_size"],
                "output_schema": definition["output_schema"],
                "precision_target": definition["precision_target"],
                "device_profile": f"configs/device-profiles/{variant}.json",
                "export_target": definition["export_target"],
                "expected_runtime": definition["expected_runtime"],
                "tiling_policy": _tiling_policy(variant),
                "training_smoke": {"dataset": "synthetic", "epochs": 1},
                "evaluation": {"dataset": "synthetic", "metrics": "task-default"},
                "export": {"artifact_path": f"registry/{directory_family}/{name}"},
                "quantization": {"target": definition["precision_target"], "metadata_only": True},
                "benchmark": {"config": "configs/benchmarks/document_mobile_smoke.json"},
                "limitations": definition["limitations"],
                "status": "experimental",
            }
            config_path = root / "configs" / "mobile" / directory_family / f"{variant}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (registry / "index.json").write_text(
        json.dumps({"schema_version": 1, "entries": entries}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
