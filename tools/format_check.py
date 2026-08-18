"""Small dependency-free formatting gate for MP1 Python sources."""
from pathlib import Path
import sys

paths = [path for path in Path(".").rglob("*.py") if "__pycache__" not in path.parts]
issues = [f"{path}: trailing whitespace" for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip() != line]
if issues:
    print("\n".join(issues), file=sys.stderr)
    raise SystemExit(1)
print(f"format check passed for {len(paths)} Python files")

