"""Emit a contract-valid placeholder artifact from an export configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from wellfriend_models.registry import validate_artifact_directory, write_placeholder_artifact
from wellfriend_models.training.train import load_config


def main() -> int:
    """Provide ``python -m wellfriend_models.export.export --config ...``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    root = Path(__file__).resolve().parents[2]
    export_config = config["export"]
    family = str(export_config.get("family", config["model"].get("family", "document-model")))
    output = arguments.output_dir or root / export_config.get(
        "output_dir", f"artifacts/{config['experiment_id']}"
    )
    write_placeholder_artifact(output, family=family, task=config["task"])
    result = validate_artifact_directory(output, allow_placeholder=True)
    print(f"exported placeholder artifact: {result['directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
