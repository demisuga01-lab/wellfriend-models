# Architecture

`wellfriend-models` has a narrow, auditable boundary: datasets and synthetic fixtures feed experiments; reference baselines provide smoke evidence; metrics/evaluation record results; export produces an artifact contract; the registry lists artifacts. Production code remains in Rust in `wellfriend-perception` and later consumes artifact files through an adapter, never arbitrary training Python.

```text
dataset manifest / synthetic generator
        ↓
experiment config → baseline or optional ML implementation → evaluation metrics
        ↓                                                ↓
provenance and environment evidence                    export contract
                                                        ↓
                                               registry → perception adapter (later)
```

Optional heavy ML dependencies are intentionally absent from base CI. Any future tensor, ONNX, quantization, or distillation implementation must retain the same contracts and license/provenance gates.

## MP6 mobile optimization boundary

MP6 adds device profiles, no-weight experimental mobile entries, distillation/quantization/pruning reports, tiling policies, promotion gates, and synthetic ScanBench model reports. These layers transform research evidence into a candidate artifact plan without turning Python into a production runtime or creating a production model claim.
