# Model artifact contract

A released artifact is a self-contained, immutable directory that a future `wellfriend-perception` model adapter can consume without importing Python:

```text
model-name/
├── model.onnx
├── manifest.json
├── preprocess.json
├── postprocess.json
├── labels.json
├── checksums.json
├── metrics.json
├── LICENSES/
└── README.md
```

`manifest.json` is schema version 1 and requires identity/version, task/domain, architecture, input/output schemas, preprocess/postprocess references, data references, code and weights license information, intended runtime/device/precision, dynamic-shape behavior, metrics/checksum references, limitations, safety notes, provenance, `status`, and `production_ready`.

Supported tasks are document segmentation, corner regression, quality prediction, restoration, cleanup mask prediction, OCR detection/recognition, and layout detection. A `placeholder` must set `production_ready: false`, contain no model file, and only validate with explicit placeholder permission. A `released` artifact must include `model.onnx` and pass checksum validation without that permission.

```powershell
python -m wellfriend_models.registry.validate path/to/model-artifact
python -m wellfriend_models.registry.validate registry/document-detector/placeholder --allow-placeholder
```

The contract intentionally separates research Python from the production runtime. It does not itself endorse ONNX Runtime or any particular execution provider.
