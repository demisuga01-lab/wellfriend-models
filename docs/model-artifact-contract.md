# Model artifact contract

A released artifact directory is immutable and contains:

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

`manifest.json` must include model name, version, task, domain, input shape, dynamic-shape support, pixel format, preprocessing, postprocessing, expected output schema, training-dataset references, license notes, intended runtime, device class, metrics, and hashes. `registry/artifact_schema.py` rejects an incomplete artifact, bad checksums, or a placeholder passed as a production release. Production consumers use these exported files without importing Python.

