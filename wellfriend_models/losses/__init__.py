"""Reference scalar losses; tensor-backed implementations remain optional."""

from .reference import (
    binary_cross_entropy,
    corner_regression_loss,
    dice_loss,
    iou_loss,
    l1_loss,
    l2_loss,
    quality_regression_loss,
    smooth_l1_loss,
)

__all__ = [
    "binary_cross_entropy",
    "corner_regression_loss",
    "dice_loss",
    "iou_loss",
    "l1_loss",
    "l2_loss",
    "quality_regression_loss",
    "smooth_l1_loss",
]
