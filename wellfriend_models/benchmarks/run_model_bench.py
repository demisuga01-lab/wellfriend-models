"""Run a synthetic-only ScanBench model smoke configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wellfriend_models.benchmarks.scanbench import benchmark_artifact, write_scanbench_report
from wellfriend_models.schemas import ContractError


def main() -> int:
    """Provide `python -m wellfriend_models.benchmarks.run_model_bench --config ...`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    try:
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot load benchmark config: {error}")
    required = {"schema_version", "artifact_path", "device_profile", "categories", "seed"}
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
        parser.error("invalid ScanBench model config")
    root = Path(__file__).resolve().parents[2]
    try:
        report = benchmark_artifact(config, root=root)
    except ContractError as error:
        parser.error(str(error))
    output = arguments.output_dir or root / "benchmarks" / "out"
    json_path, markdown_path = write_scanbench_report(report, output)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
