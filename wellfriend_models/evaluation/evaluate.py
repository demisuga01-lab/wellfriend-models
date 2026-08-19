"""Evaluate a configuration and emit durable metric/diagnostic manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wellfriend_models.training.train import load_config, run_baseline_experiment


def main() -> int:
    """Provide the documented synthetic evaluation command without model-framework imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write-predictions-manifest", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    root = Path(__file__).resolve().parents[2]
    output = arguments.output_dir or (
        root
        / config["logging"].get("output_dir", "experiments")
        / f"{config['experiment_id']}-evaluation"
    )
    result = run_baseline_experiment(config, output, root=root)
    diagnostics = {
        "status": "baseline_diagnostics",
        "limitations": [
            "synthetic fixtures only",
            "no production model inference",
            "no OCR runtime",
        ],
    }
    failures = {
        "status": "no_failure_examples_materialized",
        "reason": "small synthetic baseline run",
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    (output / "failure-examples-index.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    if arguments.write_predictions_manifest:
        predictions = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "synthetic_baseline",
            "metrics": result["metrics"],
        }
        (output / "predictions-manifest.json").write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"status": "evaluation_complete", "output": str(output), "metrics": result["metrics"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
