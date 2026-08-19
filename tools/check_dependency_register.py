"""Reject incomplete or disallowed direct-dependency records in the TOML register."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REQUIRED = {
    "name",
    "version",
    "license",
    "source_url",
    "purpose",
    "risk_level",
    "status",
    "used_by",
    "scope",
}
BLOCKED_LICENSE_TOKENS = ("gpl", "agpl", "lgpl", "non-commercial", "research-only", "unknown")


def main() -> int:
    """Validate the direct dependency records used by this repository."""
    path = Path(__file__).resolve().parents[1] / "third_party" / "dependency-register.toml"
    records = tomllib.loads(path.read_text(encoding="utf-8")).get("dependency", [])
    errors: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index} is not a TOML table")
            continue
        missing = sorted(REQUIRED.difference(record))
        if missing:
            errors.append(f"record {record.get('name', index)} missing: {', '.join(missing)}")
        license_name = str(record.get("license", "")).lower()
        if any(token in license_name for token in BLOCKED_LICENSE_TOKENS):
            detail = f"record {record.get('name', index)} has blocked or unclear license"
            errors.append(f"{detail}: {record.get('license')}")
    if errors:
        print("dependency register validation failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    print(f"dependency register validation passed: {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
