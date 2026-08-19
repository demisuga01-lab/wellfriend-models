# ADR-0005: Optional heavy ML dependencies

Base CI stays standard-library plus development tooling. Tensor/ONNX dependencies are opt-in and must be audited before becoming project dependencies or release requirements.
