"""Artifact export seams; ONNX graph/runtime validation remains optional and audited."""

from .onnx_validation import validate_onnx_backed_artifact

__all__ = ["validate_onnx_backed_artifact"]
