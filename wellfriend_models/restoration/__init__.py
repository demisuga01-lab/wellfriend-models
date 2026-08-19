"""Mobile restoration teacher/student boundary contracts without DocRes integration."""

from .mobile_boundary import RESTORATION_TASKS, validate_docres_mobile_boundary

__all__ = ["RESTORATION_TASKS", "validate_docres_mobile_boundary"]
