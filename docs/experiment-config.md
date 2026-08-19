# Experiment configuration

`ExperimentConfig` is JSON schema version 1 with `experiment_id`, `seed`, task/model/dataset/split/transform/loss/optimizer/scheduler fields, batch/epoch/device/precision controls, logging/checkpoint/export settings, and reproducibility metadata. The dependency-light MP5 runner supports synthetic CPU smoke baselines for segmentation, corners, and quality.

```powershell
python -m wellfriend_models.training.train --config configs/document/segmentation_baseline.json
```

Each run writes `config.json`, `environment.json`, `metrics.json`, and `logs.jsonl`. They are evidence of a baseline pipeline run—not measured model quality or a training checkpoint.
