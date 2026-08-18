"""Validate a production artifact directory or an explicitly permitted registry placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry.artifact_schema import ContractError, validate_artifact_directory


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
