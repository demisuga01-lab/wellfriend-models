"""Clearly non-production document baselines for reproducible MP5 smoke experiments."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Protocol

from wellfriend_models.synthetic import SyntheticDocumentSample


class BaselineModel(Protocol):
    """Minimal model interface shared by no-dependency and optional tensor implementations."""

    task: str


def _binary(mask: tuple[int, ...]) -> list[int]:
    return [int(value >= 128) for value in mask]


def _box_from_mask(
    mask: tuple[int, ...], width: int, height: int
) -> tuple[tuple[float, float], ...]:
    points = [(index % width, index // width) for index, value in enumerate(mask) if value >= 128]
    if not points:
        raise ValueError("cannot derive corners from an empty mask")
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    top, bottom = min(y for _, y in points), max(y for _, y in points)
    return (
        (float(left), float(top)),
        (float(right), float(top)),
        (float(right), float(bottom)),
        (float(left), float(bottom)),
    )


@dataclass(frozen=True)
class DocumentSegmentationBaseline:
    """A luminance-threshold baseline, not a learned segmenter or production model."""

    threshold: int = 128
    task: str = "document_segmentation"

    def predict(self, sample: SyntheticDocumentSample) -> list[float]:
        """Return per-pixel document probabilities from normalized luminance."""
        background_median = statistics.median(sample.pixels)
        direction = 1 if statistics.fmean(sample.pixels) >= background_median else -1
        probabilities = []
        for value in sample.pixels:
            signal = (value - self.threshold) / max(1, 255 - self.threshold)
            if direction < 0:
                signal = (self.threshold - value) / max(1, self.threshold)
            probabilities.append(min(1.0, max(0.0, signal)))
        return probabilities

    def binary_mask(self, sample: SyntheticDocumentSample) -> list[int]:
        """Threshold baseline probabilities for base metrics."""
        return [int(value >= 0.5) for value in self.predict(sample)]


@dataclass(frozen=True)
class DocumentCornerBaseline:
    """Corner contract reference derived from the synthetic segmentation target."""

    task: str = "document_corner_regression"

    def predict(self, sample: SyntheticDocumentSample) -> dict[str, object]:
        """Return a bounding-box quad and heuristic confidence, preserving a realistic interface."""
        quad = _box_from_mask(sample.mask, sample.width, sample.height)
        coverage = sum(value >= 128 for value in sample.mask) / len(sample.mask)
        confidence = min(0.95, max(0.05, coverage * 1.8))
        return {"corners": quad, "confidence": confidence, "status": "baseline_heuristic"}


@dataclass(frozen=True)
class DocumentQualityBaseline:
    """Lightweight image-statistics quality baseline, explicitly not calibrated."""

    task: str = "document_quality"

    def predict(self, sample: SyntheticDocumentSample) -> dict[str, float]:
        """Estimate documented quality labels without accessing the fixture labels."""
        values = [float(value) for value in sample.pixels]
        mean = statistics.fmean(values)
        variance = statistics.pvariance(values)
        saturation = sum(value >= 250 for value in values) / len(values)
        dark = sum(value <= 45 for value in values) / len(values)
        transitions = sum(
            abs(values[index] - values[index - 1]) for index in range(1, len(values))
        ) / (len(values) - 1)
        return {
            "blur": max(0.0, min(1.0, 1.0 - transitions / 64.0)),
            "shadow": max(0.0, min(1.0, dark * 2.0)),
            "glare": max(0.0, min(1.0, saturation * 4.0)),
            "curvature": 0.0,
            "noise": max(0.0, min(1.0, variance / (255.0**2))),
            "faded": max(
                0.0, min(1.0, 1.0 - math.sqrt(variance) / 64.0 + abs(mean - 128.0) / 512.0)
            ),
        }


@dataclass(frozen=True)
class DocumentCleanupBaseline:
    """Return a high-luminance cleanup mask as a test-only contract baseline."""

    task: str = "document_cleanup_mask"
    threshold: int = 250

    def predict(self, sample: SyntheticDocumentSample) -> list[float]:
        """Mark likely glare/debris pixels; this is not hand/finger segmentation."""
        return [float(value >= self.threshold) for value in sample.pixels]


@dataclass(frozen=True)
class ClassicalRestorationBaseline:
    """CPU scalar restoration seam for evaluation, never a DocRes replacement."""

    task: str = "document_restoration"

    def process(self, sample: SyntheticDocumentSample, operation: str) -> dict[str, object]:
        """Run supported scalar operations or return an explicit placeholder result."""
        if operation == "binarization_baseline":
            values = tuple(255 if pixel >= 128 else 0 for pixel in sample.pixels)
            return {"status": "baseline", "operation": operation, "pixels": values}
        if operation == "appearance_enhancement_baseline":
            mean = statistics.fmean(sample.pixels)
            values = tuple(
                min(255, max(0, round((pixel - mean) * 1.2 + 128))) for pixel in sample.pixels
            )
            return {"status": "baseline", "operation": operation, "pixels": values}
        if operation == "deshadowing_baseline":
            row_means = [
                statistics.fmean(sample.pixels[row * sample.width : (row + 1) * sample.width])
                for row in range(sample.height)
            ]
            values = []
            for row, row_mean in enumerate(row_means):
                values.extend(
                    min(255, max(0, round(pixel + (statistics.fmean(row_means) - row_mean))))
                    for pixel in sample.pixels[row * sample.width : (row + 1) * sample.width]
                )
            return {"status": "baseline", "operation": operation, "pixels": tuple(values)}
        if operation in {"dewarping_placeholder", "deblurring_placeholder", "end2end_placeholder"}:
            return {
                "status": "placeholder",
                "operation": operation,
                "reason": "requires an audited learned restoration model in a later phase",
            }
        raise ValueError(f"unsupported restoration operation: {operation}")
