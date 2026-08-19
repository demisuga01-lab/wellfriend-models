"""CLI for strict DatasetManifest validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .manifest import load_dataset_manifest, manifest_sha256


def main() -> int:
    """Validate one manifest and emit only its stable identifier/hash on success."""
    parser = argparse.ArgumentParser(description="Validate a Wellfriend DatasetManifest")
    parser.add_argument("path", type=Path)
    parser.add_argument("--production-use", action="store_true")
    args = parser.parse_args()
    try:
        manifest = load_dataset_manifest(args.path, production_use=args.production_use)
    except ValueError as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"dataset validation passed: {manifest['dataset_id']} {manifest_sha256(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
