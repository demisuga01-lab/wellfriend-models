# Wellfriend Models

`wellfriend-models` is the Python research, training, evaluation, distillation, quantization, export, and artifact-registry repository for the Wellfriend ecosystem. It is not a production inference runtime, a dataset redistribution service, or a source of untracked model weights.

The production engine is [`wellfriend-perception`](https://github.com/wellfriend/wellfriend-perception); this repository produces auditable exported artifacts that it can consume. [`wellfriend-scan`](https://github.com/wellfriend/wellfriend-scan) is the reference scanner product.

## Validate

```powershell
python tools/format_check.py
python tools/lint.py
python -m unittest discover -s tests -v
python export/validate_artifact.py --allow-placeholder registry/document-detector
```

MP1 supplies schemas, validators, training/evaluation scaffolds, and placeholder registries only. It does not train, redistribute, or claim performance for models.

The repository is Apache-2.0. Dependencies and model/dataset licenses require explicit provenance records before adoption.
