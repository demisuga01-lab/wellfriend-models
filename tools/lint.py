"""Compile every repository Python source as a portable MP1 lint baseline."""

import py_compile
import sys
from pathlib import Path

paths = [path for path in Path(".").rglob("*.py") if "__pycache__" not in path.parts]
errors = []
for path in paths:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        errors.append(str(exc))
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"lint baseline passed for {len(paths)} Python files")
