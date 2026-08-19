"""Small numeric references used to make experiment configuration semantics testable."""

from __future__ import annotations

import math
from collections.abc import Sequence

from wellfriend_models.metrics import binary_iou, corner_mae, dice


def _same(left: Sequence[float], right: Sequence[float]) -> None:
    if not left or len(left) != len(right):
        raise ValueError("loss inputs must be non-empty and equal length")


def binary_cross_entropy(
    prediction: Sequence[float], reference: Sequence[float], *, epsilon: float = 1e-7
) -> float:
    """Compute mean binary cross entropy for probabilities in the closed unit interval."""
    _same(prediction, reference)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    total = 0.0
    for probability, target in zip(prediction, reference, strict=True):
        if not 0.0 <= probability <= 1.0 or not 0.0 <= target <= 1.0:
            raise ValueError("BCE inputs must be in [0, 1]")
        bounded = min(1.0 - epsilon, max(epsilon, probability))
        total -= target * math.log(bounded) + (1.0 - target) * math.log(1.0 - bounded)
    return total / len(prediction)


def dice_loss(prediction: Sequence[int | bool], reference: Sequence[int | bool]) -> float:
    """Return one minus the binary Dice score."""
    return 1.0 - dice(prediction, reference)


def iou_loss(prediction: Sequence[int | bool], reference: Sequence[int | bool]) -> float:
    """Return one minus binary IoU."""
    return 1.0 - binary_iou(prediction, reference)


def l1_loss(prediction: Sequence[float], reference: Sequence[float]) -> float:
    """Return mean absolute error."""
    _same(prediction, reference)
    return sum(abs(a - b) for a, b in zip(prediction, reference, strict=True)) / len(prediction)


def l2_loss(prediction: Sequence[float], reference: Sequence[float]) -> float:
    """Return mean squared error."""
    _same(prediction, reference)
    return sum((a - b) ** 2 for a, b in zip(prediction, reference, strict=True)) / len(prediction)


def smooth_l1_loss(
    prediction: Sequence[float], reference: Sequence[float], *, beta: float = 1.0
) -> float:
    """Return Huber/smooth-L1 loss."""
    _same(prediction, reference)
    if beta <= 0:
        raise ValueError("beta must be positive")

    def item(delta: float) -> float:
        return 0.5 * delta**2 / beta if delta < beta else delta - 0.5 * beta

    return sum(item(abs(a - b)) for a, b in zip(prediction, reference, strict=True)) / len(
        prediction
    )


def corner_regression_loss(
    prediction: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]
) -> float:
    """Use mean Euclidean corner error as a clear scalar baseline loss."""
    return corner_mae(prediction, reference)


def quality_regression_loss(prediction: dict[str, float], reference: dict[str, float]) -> float:
    """Return mean squared error across exactly matching quality-label keys."""
    if not prediction or prediction.keys() != reference.keys():
        raise ValueError("quality labels must be non-empty with matching keys")
    return l2_loss(
        [prediction[key] for key in sorted(prediction)],
        [reference[key] for key in sorted(reference)],
    )


def edge_aware_loss_placeholder(*_: object, **__: object) -> None:
    """Reserve the contract without implying an MP5 tensor implementation."""
    raise NotImplementedError("edge-aware loss requires an optional tensor backend")


def ocr_aware_loss_placeholder(*_: object, **__: object) -> None:
    """Reserve OCR-aware optimization for later audited OCR integrations."""
    raise NotImplementedError("OCR-aware loss is intentionally not implemented in MP5")
