"""Quantization planning/report contracts with optional ONNX runtime integration seams."""

from .contracts import (
    PRECISION_MODES,
    build_quantization_report,
    optional_dynamic_int8_quantize,
    validate_quantization_config,
    validate_quantization_report,
)

__all__ = [
    "PRECISION_MODES",
    "build_quantization_report",
    "optional_dynamic_int8_quantize",
    "validate_quantization_config",
    "validate_quantization_report",
]
