# Mobile optimization

Mobile families are `document-detector-mobile`, `document-corners-mobile`, `document-quality-mobile`, `document-cleanup-mobile`, and `document-restoration-mobile`, each with low/mid/high **experimental** metadata variants. Their config contains the synthetic training/evaluation/export/quantization/benchmark plan, input size, precision target, runtime target, and tiling policy.

The checked-in entries include no weights or `model.onnx`. They establish a portable contract from research model to a mobile candidate; they do not establish mobile performance or production quality.

Run a tiny synthetic campaign to write `metrics.json`, an artifact-manifest copy, reproducibility evidence, and both ScanBench reports:

```powershell
python -m wellfriend_models.mobile.campaign --config configs/mobile/document-detector/low.json --output-dir $env:TEMP\wellfriend-mobile-smoke
```
