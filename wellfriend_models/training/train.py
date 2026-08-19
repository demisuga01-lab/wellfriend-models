"""Run deterministic synthetic baseline experiments; this is not production ML training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from wellfriend_models.metrics import binary_iou, corner_mae
from wellfriend_models.models import (
    DocumentCornerBaseline,
    DocumentQualityBaseline,
    DocumentSegmentationBaseline,
)
from wellfriend_models.schemas import ContractError, validate_experiment_config
from wellfriend_models.synthetic import SyntheticDocumentGenerator
from wellfriend_models.training.reproducibility import environment_summary, set_deterministic_seed


def load_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate a JSON experiment configuration."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read experiment config {path}: {error}") from error
    validate_experiment_config(config)
    return config


def _dataset_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config["dataset"], sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def run_baseline_experiment(
    config: dict[str, Any], output_directory: Path, *, root: Path
) -> dict[str, Any]:
    """Evaluate a synthetic baseline and write its reproducibility evidence."""
    set_deterministic_seed(config["seed"])
    output_directory.mkdir(parents=True, exist_ok=True)
    generator_config = config["dataset"]
    generator = SyntheticDocumentGenerator(
        width=int(generator_config.get("width", 96)),
        height=int(generator_config.get("height", 72)),
        seed=config["seed"],
    )
    count = int(generator_config.get("count_per_case", 1))
    samples = generator.generate_suite(count)
    task = config["task"]
    if task == "document_segmentation":
        model = DocumentSegmentationBaseline()
        scores = [
            binary_iou(model.binary_mask(sample), [value >= 128 for value in sample.mask])
            for sample in samples
        ]
        metrics = {"iou": sum(scores) / len(scores)}
    elif task == "document_corner_regression":
        model = DocumentCornerBaseline()
        scores = [corner_mae(model.predict(sample)["corners"], sample.quad) for sample in samples]
        metrics = {"corner_mae_pixels": sum(scores) / len(scores)}
    elif task == "document_quality":
        model = DocumentQualityBaseline()
        scores = []
        for sample in samples:
            prediction = model.predict(sample)
            scores.append(
                sum((prediction[key] - sample.quality_labels[key]) ** 2 for key in prediction)
                / len(prediction)
            )
        metrics = {"quality_mse": sum(scores) / len(scores)}
    else:
        raise ContractError(f"MP5 baseline runner does not support task {task}")
    run = {
        "experiment_id": config["experiment_id"],
        "task": task,
        "status": "baseline_complete",
        "model": config["model"],
        "sample_count": len(samples),
        "dataset_manifest_hash": _dataset_hash(config),
        "metrics": metrics,
    }
    (output_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "environment.json").write_text(
        json.dumps(environment_summary(root), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "logs.jsonl").write_text(
        json.dumps({"event": "baseline_complete", **run}) + "\n", encoding="utf-8"
    )
    return run


def main() -> int:
    """Provide the documented ``python -m wellfriend_models.training.train`` entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    root = Path(__file__).resolve().parents[2]
    configured = config["logging"].get("output_dir", "experiments")
    output = arguments.output_dir or (root / configured / config["experiment_id"])
    result = run_baseline_experiment(config, output, root=root)
    print(
        json.dumps(
            {"status": result["status"], "output": str(output), "metrics": result["metrics"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
