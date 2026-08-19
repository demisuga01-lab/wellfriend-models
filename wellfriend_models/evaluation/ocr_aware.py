"""OCR-aware restoration evaluation hooks without an OCR engine dependency."""

from __future__ import annotations

from collections.abc import Sequence

from wellfriend_models.metrics import cer, wer


def ocr_aware_restoration_metrics(
    *,
    baseline_text: str,
    restored_text: str,
    reference_text: str,
    baseline_edges: Sequence[float],
    restored_edges: Sequence[float],
    expected_strokes: Sequence[int | bool],
    restored_strokes: Sequence[int | bool],
) -> dict[str, float]:
    """Report deltas only from supplied observations; it does not infer OCR improvement."""
    if len(baseline_edges) != len(restored_edges) or len(expected_strokes) != len(restored_strokes):
        raise ValueError("OCR-aware metrics require paired edge and stroke vectors")
    baseline_cer, restored_cer = (
        cer(baseline_text, reference_text),
        cer(restored_text, reference_text),
    )
    baseline_wer, restored_wer = (
        wer(baseline_text, reference_text),
        wer(restored_text, reference_text),
    )
    edge_preservation = _normalized_similarity(baseline_edges, restored_edges)
    stroke_preservation = sum(
        bool(expected) == bool(restored)
        for expected, restored in zip(expected_strokes, restored_strokes, strict=True)
    ) / len(expected_strokes)
    foreground = sum(bool(item) for item in restored_strokes) / len(restored_strokes)
    return {
        "cer_delta_after_restoration": restored_cer - baseline_cer,
        "wer_delta_after_restoration": restored_wer - baseline_wer,
        "text_edge_preservation": edge_preservation,
        "foreground_stroke_preservation": stroke_preservation,
        "binarization_text_contrast": foreground,
    }


def _normalized_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    span = max(1.0, max([abs(float(item)) for item in [*left, *right]], default=1.0))
    mean_error = sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)) / len(left)
    return max(0.0, 1.0 - mean_error / span)
