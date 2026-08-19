"""CLI for strict model-artifact validation."""

from __future__ import annotations

import argparse

from wellfriend_models.registry.artifact import validate_artifact_directory
from wellfriend_models.schemas import ContractError


def main() -> int:
    """Run ``python -m wellfriend_models.registry.validate artifact-dir``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=str)
    parser.add_argument("--allow-placeholder", action="store_true")
    arguments = parser.parse_args()
    try:
        result = validate_artifact_directory(
            arguments.artifact, allow_placeholder=arguments.allow_placeholder
        )
    except ContractError as error:
        parser.error(str(error))
    print(f"artifact validation passed: {result['model_id']} ({result['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
