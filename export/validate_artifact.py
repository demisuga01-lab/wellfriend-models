"""Compatibility wrapper for the package-level artifact validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wellfriend_models.registry.artifact import validate_artifact_directory
from wellfriend_models.schemas import ContractError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-placeholder", action="store_true")
    args = parser.parse_args()
    try:
        validate_artifact_directory(args.directory, allow_placeholder=args.allow_placeholder)
    except ContractError as exc:
        print(f"artifact validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"artifact validation passed: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
