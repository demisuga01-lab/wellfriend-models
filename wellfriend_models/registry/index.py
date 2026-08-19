"""Validate the checked-in registry index and every declared placeholder artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wellfriend_models.registry.artifact import validate_artifact_directory
from wellfriend_models.schemas import ContractError


def validate_registry_index(
    path: Path, *, allow_placeholders: bool = True
) -> list[dict[str, object]]:
    """Validate index shape and resolve each relative artifact path safely."""
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read registry index: {error}") from error
    if (
        not isinstance(index, dict)
        or index.get("schema_version") != 1
        or not isinstance(index.get("entries"), list)
    ):
        raise ContractError("registry index must be schema version 1 with entries list")
    results = []
    for entry in index["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ContractError("registry index entries require a relative path")
        target = (path.parent / entry["path"]).resolve()
        if path.parent.resolve() not in target.parents:
            raise ContractError("registry entry path escapes registry root")
        result = validate_artifact_directory(target, allow_placeholder=allow_placeholders)
        if result["status"] == "placeholder" and entry.get("production_ready"):
            raise ContractError("registry index cannot mark a placeholder entry production-ready")
        results.append(result)
    return results


def main() -> int:
    """Run registry index validation in CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--allow-placeholders", action="store_true")
    arguments = parser.parse_args()
    try:
        results = validate_registry_index(
            arguments.index, allow_placeholders=arguments.allow_placeholders
        )
    except ContractError as error:
        parser.error(str(error))
    print(f"registry validation passed: {len(results)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
