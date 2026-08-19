"""Task-aware scalar teacher/student losses for synthetic smoke comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from wellfriend_models.losses import binary_cross_entropy, corner_regression_loss, l1_loss
from wellfriend_models.schemas import ContractError


def distillation_loss(task: str, student: Any, teacher: Any) -> float:
    """Select a transparent scalar loss for an MP6 task-level teacher/student smoke run."""
    if task in {"document_segmentation", "document_cleanup_mask"}:
        return binary_cross_entropy(_probabilities(student), _probabilities(teacher))
    if task == "document_corner_regression":
        return corner_regression_loss(student, teacher)
    if task in {"document_quality", "document_restoration"}:
        return l1_loss(_numbers(student), _numbers(teacher))
    raise ContractError("unsupported distillation task")


def ocr_aware_restoration_loss_hook(*_: object, **__: object) -> None:
    """Reserve OCR-aware loss wiring without importing an OCR runtime in MP6."""
    raise NotImplementedError(
        "OCR-aware restoration loss is an evaluation hook, not an OCR integration"
    )


def structure_loss_placeholder(*_: object, **__: object) -> None:
    """Reserve structural distillation for optional tensor-backed implementations."""
    raise NotImplementedError("structure loss requires an optional tensor backend")


def _probabilities(value: Sequence[float]) -> list[float]:
    numbers = [float(item) for item in value]
    if any(item < 0 or item > 1 for item in numbers):
        raise ContractError("segmentation distillation inputs must be probabilities")
    return numbers


def _numbers(value: Sequence[float]) -> list[float]:
    return [float(item) for item in value]
