# Wellfriend Models

`wellfriend-models` is the reproducible research, evaluation, export, and artifact-registry platform for the Wellfriend Open-Source Perception Ecosystem.

It is not a production inference runtime, a dataset redistribution service, a source of untracked model weights, or a claim that any placeholder/baseline model is production quality. The Rust production engine is [wellfriend-perception](https://github.com/demisuga01-lab/wellfriend-perception); it consumes exported artifacts without importing arbitrary Python. [wellfriend-scan](https://github.com/demisuga01-lab/wellfriend-scan) is the reference scanner product.

## Current status

MP5 provides strict dataset, experiment, and artifact contracts; deterministic synthetic document fixtures; scalar baseline interfaces; training/evaluation smoke commands; placeholder-only registry entries; and a lightweight benchmark harness. No restricted data, third-party code, or model weights are included.

## Install and validate

```powershell
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
python -m pytest -q
python -m wellfriend_models.datasets.validate path/to/dataset_manifest.json
python -m wellfriend_models.registry.validate registry/document-detector/placeholder --allow-placeholder
python -m wellfriend_models.registry.index registry/index.json --allow-placeholders
```

## Baseline smoke flows

```powershell
python -m wellfriend_models.training.train --config configs/smoke/document_segmentation_synthetic.json
python -m wellfriend_models.training.train --config configs/smoke/document_corners_synthetic.json
python -m wellfriend_models.training.train --config configs/smoke/document_quality_synthetic.json
python -m wellfriend_models.evaluation.evaluate --config configs/smoke/document_segmentation_synthetic.json --write-predictions-manifest
python -m wellfriend_models.export.export --config configs/export/document_detector_baseline.json
python -m wellfriend_models.benchmarks.run --smoke
```

The repository is Apache-2.0. Direct dependencies and planned external references are recorded in [`third_party/dependency-register.toml`](third_party/dependency-register.toml); unknown, non-commercial, research-only, GPL-family, or unclear licenses are blocked from production artifacts.
