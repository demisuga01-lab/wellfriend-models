# Architecture overview

This repository keeps dataset manifests, experiments, evaluation, export, registry, promotion gates, and benchmarks separate from production runtime. Artifacts are consumed by adapters later; Python is never imported by the perception engine.
