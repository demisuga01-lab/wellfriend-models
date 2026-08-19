"""Device-aware model-family and resolution contracts for future mobile runtimes."""

from .families import MOBILE_FAMILIES, mobile_family_variant, validate_mobile_candidate_config
from .profiles import DEVICE_CLASSES, load_device_profile, validate_device_profile
from .tiling import TILING_MODES, validate_tiling_policy

__all__ = [
    "DEVICE_CLASSES",
    "MOBILE_FAMILIES",
    "TILING_MODES",
    "load_device_profile",
    "mobile_family_variant",
    "validate_mobile_candidate_config",
    "validate_device_profile",
    "validate_tiling_policy",
]
