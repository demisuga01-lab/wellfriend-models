# ScanBench model reports

Run the synthetic-only model benchmark:

```powershell
python -m wellfriend_models.benchmarks.run_model_bench --config configs/benchmarks/document_mobile_smoke.json
```

It writes `scanbench-model-report.json` and `.md` with repository/artifact/dataset hashes, device profile, scenario metrics, artifact size, parameter/FLOP/memory placeholders, latency smoke observations, export state, fit result, provenance safety, limitations, timestamp, and environment.

Categories cover easy, rotated, perspective, contrast, shadow, glare, blur, receipt-like, cut-off, distractor, and no-document fixtures. Results are synthetic-only CPU smoke evidence—not real-device latency, real model accuracy, or a production claim.
