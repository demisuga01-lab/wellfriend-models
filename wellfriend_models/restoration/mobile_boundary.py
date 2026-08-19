"""Validate the DocRes-Mobile research boundary while preserving the audit gate."""

from __future__ import annotations

from typing import Any

from wellfriend_models.schemas import ContractError

RESTORATION_TASKS = frozenset(
    {"dewarping", "deshadowing", "appearance", "deblurring", "binarization", "end2end"}
)
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "teacher_registry_slot",
        "student_registry_slot",
        "tasks",
        "mobile_target_profiles",
        "tiling_policy",
        "ocr_aware_metric_hooks",
        "quality_metric_hooks",
        "status",
    }
)


def validate_docres_mobile_boundary(boundary: dict[str, Any]) -> None:
    """Require explicit audit gating and all planned restoration task/metric seams."""
    if set(boundary) != REQUIRED_FIELDS:
        raise ContractError("DocRes-Mobile boundary fields do not match schema")
    if boundary["schema_version"] != 1 or boundary["status"] != "audit-gated":
        raise ContractError("DocRes-Mobile boundary must remain audit-gated")
    teacher = boundary["teacher_registry_slot"]
    student = boundary["student_registry_slot"]
    if not isinstance(teacher, dict) or teacher.get("weights_included") is not False:
        raise ContractError("DocRes teacher slot must not include weights")
    if not isinstance(student, dict) or student.get("production_ready") is not False:
        raise ContractError("DocRes student slot must not be production-ready")
    if set(boundary["tasks"]) != RESTORATION_TASKS:
        raise ContractError("DocRes-Mobile boundary must enumerate restoration tasks")
    if not isinstance(boundary["ocr_aware_metric_hooks"], list) or not isinstance(
        boundary["quality_metric_hooks"], list
    ):
        raise ContractError("DocRes-Mobile metric hooks must be lists")
