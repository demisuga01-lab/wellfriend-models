"""Run one tiny synthetic mobile-candidate campaign without weights, GPUs, or external data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wellfriend_models.benchmarks.scanbench import (
    SCANBENCH_CATEGORIES,
    benchmark_artifact,
    write_scanbench_report,
)
from wellfriend_models.mobile.families import validate_mobile_candidate_config
from wellfriend_models.training.reproducibility import environment_summary


def main() -> int:
    """Write metrics, reproducibility evidence, and ScanBench reports for one smoke candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    validate_mobile_candidate_config(config)
    root = Path(__file__).resolve().parents[2]
    artifact_path = config["export"]["artifact_path"]
    benchmark_config = {
        "schema_version": 1,
        "artifact_path": artifact_path,
        "device_profile": config["device_profile"],
        "categories": list(SCANBENCH_CATEGORIES),
        "seed": 601,
    }
    report = benchmark_artifact(benchmark_config, root=root)
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_scanbench_report(report, output)
    (output / "metrics.json").write_text(
        json.dumps(report["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "artifact-manifest.json").write_text(
        (root / artifact_path / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "reproducibility.json").write_text(
        json.dumps(environment_summary(root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "mobile_smoke_complete", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
