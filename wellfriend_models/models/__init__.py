"""Baseline model contracts and CPU-only reference implementations."""

from .document import (
    ClassicalRestorationBaseline,
    DocumentCleanupBaseline,
    DocumentCornerBaseline,
    DocumentQualityBaseline,
    DocumentSegmentationBaseline,
)

__all__ = [
    "ClassicalRestorationBaseline",
    "DocumentCleanupBaseline",
    "DocumentCornerBaseline",
    "DocumentQualityBaseline",
    "DocumentSegmentationBaseline",
]
