"""Strict model-artifact and registry validation APIs."""

from .artifact import validate_artifact_directory, write_placeholder_artifact

__all__ = ["validate_artifact_directory", "write_placeholder_artifact"]
