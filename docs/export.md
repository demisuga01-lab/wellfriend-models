# Export

MP5 exports a valid **placeholder artifact** from a validated experiment config, computes/writes its contract documents, and validates it. It does not claim ONNX conversion until optional, audited `torch` and `onnx` dependencies are configured and a real model is available.

```powershell
python -m wellfriend_models.export.export --config configs/export/document_detector_baseline.json
```

The generated placeholder has no `model.onnx` and cannot be accepted by production validation. A future ONNX path must preserve input/output semantics, preprocessing/postprocessing, dynamic-shape behavior, data/weights provenance, metrics, and checksums.
