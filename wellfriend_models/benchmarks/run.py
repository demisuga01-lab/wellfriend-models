"""Measure dependency-free MP5 baseline operations and emit a JSON benchmark record."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from wellfriend_models.metrics import binary_iou, latency_summary
from wellfriend_models.schemas import validate_experiment_config
from wellfriend_models.synthetic import SyntheticDocumentGenerator


def _measure(name: str, count: int, operation: object) -> dict[str, object]:
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        operation()  # type: ignore[operator]
        samples.append(time.perf_counter() - start)
    return {"name": name, **latency_summary(samples)}


def run(*, smoke: bool) -> dict[str, object]:
    """Run small deterministic benchmark cases rather than an unstable performance gate."""
    generator = SyntheticDocumentGenerator(
        width=48 if smoke else 96, height=36 if smoke else 72, seed=211
    )
    sample = generator.generate("perspective_quadrilateral")
    repetitions = 2 if smoke else 20
    return {
        "schema_version": 1,
        "benchmark": "wellfriend-models-mp5-baseline",
        "mode": "smoke" if smoke else "manual",
        "results": [
            _measure(
                "synthetic_sample_generation",
                repetitions,
                lambda: generator.generate("glare_patch"),
            ),
            _measure(
                "manifest_contract_validation",
                repetitions,
                lambda: validate_experiment_config(_config_fixture()),
            ),
            _measure(
                "metric_calculation", repetitions, lambda: binary_iou(sample.mask, sample.mask)
            ),
            _measure(
                "tiny_evaluation_loop", repetitions, lambda: sum(sample.mask) / len(sample.mask)
            ),
        ],
        "limitations": [
            "numbers are baseline observations, not CI thresholds",
            "no GPU model forward pass in base dependencies",
        ],
    }


def _config_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "benchmark-fixture",
        "seed": 1,
        "task": "document_segmentation",
        "model": {},
        "dataset": {},
        "splits": {},
        "transforms": [],
        "losses": [],
        "optimizer": {},
        "scheduler": {},
        "batch_size": 1,
        "epochs": 1,
        "device": "cpu",
        "precision": "fp32",
        "logging": {},
        "checkpoints": {},
        "export": {},
        "reproducibility": {},
    }


def main() -> int:
    """Run `python -m wellfriend_models.benchmarks.run --smoke`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(smoke=arguments.smoke)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
