"""Optional ONNX graph/runtime validation layered on mandatory artifact metadata checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wellfriend_models.registry import validate_artifact_directory
from wellfriend_models.schemas import ContractError


def validate_onnx_backed_artifact(
    directory: Path, *, runtime_smoke: bool = False, allow_nonproduction: bool = False
) -> dict[str, Any]:
    """Check an ONNX-bearing artifact; optional packages improve, never replace, metadata checks."""
    artifact = validate_artifact_directory(directory, allow_nonproduction=allow_nonproduction)
    model_path = Path(directory) / "model.onnx"
    if not model_path.is_file():
        raise ContractError("ONNX-backed artifact requires model.onnx")
    result: dict[str, Any] = {"artifact": artifact, "metadata": "validated"}
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError:
        result["onnx"] = "not-installed; metadata-only validation completed"
        return result
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    result["onnx"] = "checker-passed"
    if runtime_smoke:
        try:
            import onnxruntime  # type: ignore[import-not-found]
        except ImportError:
            result["runtime"] = "onnxruntime-not-installed"
        else:
            session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            result["runtime"] = {"session_created": True, "input_count": len(session.get_inputs())}
    return result
