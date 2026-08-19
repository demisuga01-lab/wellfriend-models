# Wellfriend model registry

Registry entries describe model-artifact directories, not importable Python models. Every MP5 entry is a contract-valid `placeholder`: it includes no `model.onnx`, no model weights, and sets `production_ready` to `false`.

Validate the index with:

```powershell
python -m wellfriend_models.registry.index registry/index.json --allow-placeholders
```

A released artifact must pass validation without `--allow-placeholder`, include a model file, verified checksums, documented provenance, and an audited weights license.
