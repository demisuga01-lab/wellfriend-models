"""Deterministic experiment utilities and CPU-only baseline training entry point."""

from .reproducibility import environment_summary, set_deterministic_seed

__all__ = ["environment_summary", "set_deterministic_seed"]
