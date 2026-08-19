"""Materialize the checked-in no-weights registry artifacts from the canonical writer."""

from __future__ import annotations

from pathlib import Path

from wellfriend_models.registry.artifact import write_placeholder_artifact

FAMILIES = {
    "document-detector": "document_segmentation",
    "document-corners": "document_corner_regression",
    "document-quality": "document_quality",
    "document-restoration": "document_restoration",
    "document-cleanup": "document_cleanup_mask",
    "ocr": "ocr_recognition",
}


def main() -> int:
    """Write only explicit placeholder artifacts; it never creates a model or weights file."""
    root = Path(__file__).resolve().parents[2] / "registry"
    for family, task in FAMILIES.items():
        write_placeholder_artifact(root / family / "placeholder", family=family, task=task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
