from __future__ import annotations

import json
from pathlib import Path

payload = json.loads(
    (Path(__file__).parents[1] / "docs" / "ecosystem-compatibility.json").read_text(
        encoding="utf-8"
    )
)
assert payload["schema_version"] == 1 and payload["ecosystem_version"] == "0.1.0-alpha.1"
for field in (
    "device_classes",
    "document_tasks",
    "filter_presets",
    "processor_ids",
    "guidance_codes",
    "export_formats",
):
    assert payload["shared_contracts"][field], field
assert payload["known_blockers"] and payload["mock_boundaries"] and payload["audit_gates"]
print("Ecosystem compatibility manifest passed")
