# Quantization

MP6 validates fp32, fp16, dynamic/static int8, and QAT-placeholder plans and reports. A report records source/target artifact, precision, operator coverage, sizes, metric delta, unsupported operations, calibration source, validation status, and `production_ready: false`.

Base CI validates metadata only. Optional `onnx` and `onnxruntime` extras unlock graph checking, dynamic int8 conversion, and CPU-session validation when a real audited ONNX artifact exists. Optional execution does not waive artifact, checksum, license, or provenance requirements.
