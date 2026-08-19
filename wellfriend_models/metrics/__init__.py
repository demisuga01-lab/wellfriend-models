"""Dependency-light, testable evaluation metrics for model research."""

from .document import (
    binary_iou,
    cer,
    corner_mae,
    corner_rmse,
    dice,
    latency_summary,
    mask_precision_recall_f1,
    normalized_corner_error,
    polygon_iou_approx,
    psnr,
    ssim,
    wer,
)

__all__ = [
    "binary_iou",
    "cer",
    "corner_mae",
    "corner_rmse",
    "dice",
    "latency_summary",
    "mask_precision_recall_f1",
    "normalized_corner_error",
    "polygon_iou_approx",
    "psnr",
    "ssim",
    "wer",
]
