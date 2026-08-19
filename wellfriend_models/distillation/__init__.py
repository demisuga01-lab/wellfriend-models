"""Teacher/student contracts for auditable model compression experiments."""

from .contracts import (
    STUDENT_TASKS,
    TEACHER_TYPES,
    build_distillation_report,
    validate_distillation_config,
    validate_distillation_report,
)
from .losses import distillation_loss

__all__ = [
    "STUDENT_TASKS",
    "TEACHER_TYPES",
    "build_distillation_report",
    "distillation_loss",
    "validate_distillation_config",
    "validate_distillation_report",
]
