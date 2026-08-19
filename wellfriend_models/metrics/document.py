"""Reference metrics with no image, OCR, or machine-learning runtime dependency."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Sequence
from typing import Any


def _require_same_length(left: Sequence[object], right: Sequence[object]) -> None:
    if not left or not right:
        raise ValueError("metric inputs must be non-empty")
    if len(left) != len(right):
        raise ValueError("metric inputs must have equal length")


def binary_iou(prediction: Sequence[int | bool], reference: Sequence[int | bool]) -> float:
    """Return binary intersection-over-union, defining two empty masks as 1.0."""
    _require_same_length(prediction, reference)
    intersection = sum(bool(p) and bool(r) for p, r in zip(prediction, reference, strict=True))
    union = sum(bool(p) or bool(r) for p, r in zip(prediction, reference, strict=True))
    return 1.0 if union == 0 else intersection / union


def dice(prediction: Sequence[int | bool], reference: Sequence[int | bool]) -> float:
    """Return Sørensen-Dice overlap for two binary masks."""
    _require_same_length(prediction, reference)
    intersection = sum(bool(p) and bool(r) for p, r in zip(prediction, reference, strict=True))
    total = sum(bool(value) for value in prediction) + sum(bool(value) for value in reference)
    return 1.0 if total == 0 else (2.0 * intersection) / total


def mask_precision_recall_f1(
    prediction: Sequence[int | bool], reference: Sequence[int | bool]
) -> dict[str, float]:
    """Return thresholded-mask precision, recall, and F1 with explicit empty handling."""
    _require_same_length(prediction, reference)
    true_positive = sum(bool(p) and bool(r) for p, r in zip(prediction, reference, strict=True))
    false_positive = sum(
        bool(p) and not bool(r) for p, r in zip(prediction, reference, strict=True)
    )
    false_negative = sum(
        not bool(p) and bool(r) for p, r in zip(prediction, reference, strict=True)
    )
    precision = (
        1.0
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        1.0
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _point_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 2 or len(right) != 2:
        raise ValueError("corner points must be pairs")
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def corner_mae(
    prediction: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]
) -> float:
    """Return mean Euclidean corner error in pixels."""
    _require_same_length(prediction, reference)
    return sum(_point_distance(p, r) for p, r in zip(prediction, reference, strict=True)) / len(
        prediction
    )


def corner_rmse(
    prediction: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]
) -> float:
    """Return root-mean-square Euclidean corner error in pixels."""
    _require_same_length(prediction, reference)
    return math.sqrt(
        sum(_point_distance(p, r) ** 2 for p, r in zip(prediction, reference, strict=True))
        / len(prediction)
    )


def normalized_corner_error(
    prediction: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    image_diagonal: float,
) -> float:
    """Normalize mean corner error by a positive image diagonal."""
    if image_diagonal <= 0:
        raise ValueError("image_diagonal must be positive")
    return corner_mae(prediction, reference) / image_diagonal


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    px, py = point
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > py) != (y2 > py):
            crossing_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < crossing_x:
                inside = not inside
    return inside


def polygon_iou_approx(
    prediction: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    *,
    samples: int = 64,
) -> float:
    """Approximate polygon IoU on a deterministic sample grid; adequate for baseline diagnostics."""
    if len(prediction) < 3 or len(reference) < 3 or samples < 2:
        raise ValueError("polygons need at least three points and samples must be at least two")
    points = [*prediction, *reference]
    min_x, max_x = min(float(p[0]) for p in points), max(float(p[0]) for p in points)
    min_y, max_y = min(float(p[1]) for p in points), max(float(p[1]) for p in points)
    if min_x == max_x or min_y == max_y:
        return 0.0
    intersection = union = 0
    for y_index in range(samples):
        y = min_y + (max_y - min_y) * (y_index + 0.5) / samples
        for x_index in range(samples):
            x = min_x + (max_x - min_x) * (x_index + 0.5) / samples
            in_prediction = _point_in_polygon((x, y), prediction)
            in_reference = _point_in_polygon((x, y), reference)
            intersection += int(in_prediction and in_reference)
            union += int(in_prediction or in_reference)
    return 0.0 if union == 0 else intersection / union


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def cer(prediction: str, reference: str) -> float:
    """Return character error rate, including a defined empty-reference policy."""
    if not reference:
        return 0.0 if not prediction else 1.0
    return _levenshtein(list(prediction), list(reference)) / len(reference)


def wer(prediction: str, reference: str) -> float:
    """Return whitespace-tokenized word error rate."""
    expected = reference.split()
    observed = prediction.split()
    if not expected:
        return 0.0 if not observed else 1.0
    return _levenshtein(observed, expected) / len(expected)


def psnr(
    prediction: Sequence[float], reference: Sequence[float], *, max_value: float = 255.0
) -> float:
    """Return peak signal-to-noise ratio, or infinity for identical signals."""
    _require_same_length(prediction, reference)
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    mse = sum((float(p) - float(r)) ** 2 for p, r in zip(prediction, reference, strict=True)) / len(
        prediction
    )
    return math.inf if mse == 0 else 10.0 * math.log10((max_value**2) / mse)


def ssim(
    prediction: Sequence[float], reference: Sequence[float], *, max_value: float = 255.0
) -> float:
    """Return a global SSIM baseline; windowed/MS-SSIM remains an explicit future extension."""
    _require_same_length(prediction, reference)
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    left = [float(value) for value in prediction]
    right = [float(value) for value in reference]
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    variance_left = statistics.pvariance(left)
    variance_right = statistics.pvariance(right)
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    ) / len(left)
    c1, c2 = (0.01 * max_value) ** 2, (0.03 * max_value) ** 2
    return ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2)
    )


def msssim_placeholder(*_: object, **__: object) -> None:
    """Declare MS-SSIM intentionally unavailable in the dependency-light MP5 baseline."""
    raise NotImplementedError("MS-SSIM is a documented MP6+ extension")


def latency_summary(samples_seconds: Iterable[float]) -> dict[str, float]:
    """Summarize non-empty latency samples in seconds."""
    values = [float(value) for value in samples_seconds]
    if not values:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean_seconds": statistics.fmean(values),
        "p50_seconds": ordered[(len(ordered) - 1) // 2],
        "p95_seconds": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
    }


def memory_summary_placeholder() -> dict[str, str]:
    """Keep memory measurement in the result contract without making platform-specific claims."""
    return {
        "status": "placeholder",
        "reason": "platform-specific memory probe is not part of MP5 base CI",
    }


def model_size_bytes(path: Any) -> int:
    """Return an artifact file size through its standard pathlib protocol."""
    from pathlib import Path

    return Path(path).stat().st_size


def timed(callable_: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Execute one lightweight benchmark operation and return its elapsed seconds."""
    start = time.perf_counter()
    result = callable_(*args, **kwargs)
    return result, time.perf_counter() - start
