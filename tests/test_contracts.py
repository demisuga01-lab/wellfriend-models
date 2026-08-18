from pathlib import Path
import tempfile
import unittest

from registry.artifact_schema import ContractError, validate_artifact_directory, validate_dataset_manifest, validate_manifest


class ArtifactContractTests(unittest.TestCase):
    def test_production_manifest_requires_all_declared_fields(self) -> None:
        with self.assertRaises(ContractError):
            validate_manifest({"model_name": "missing-fields"})

    def test_placeholder_is_never_a_production_artifact(self) -> None:
        placeholder = {"status": "placeholder", "model_name": "detector", "task": "segmentation", "domain": "document"}
        with self.assertRaises(ContractError):
            validate_manifest(placeholder)
        validate_manifest(placeholder, allow_placeholder=True)

    def test_registry_placeholder_has_no_fake_model(self) -> None:
        directory = Path("registry/document-detector")
        validate_artifact_directory(directory, allow_placeholder=True)
        self.assertFalse((directory / "model.onnx").exists())


class DatasetContractTests(unittest.TestCase):
    def test_dataset_manifest_requires_license_and_provenance(self) -> None:
        valid = {"schema_version": 1, "name": "synthetic-pages", "version": "0.1", "license": {"name": "CC0-1.0"}, "samples": [], "splits": {}, "provenance": {"generator": "documented"}}
        validate_dataset_manifest(valid)
        invalid = dict(valid)
        del invalid["provenance"]
        with self.assertRaises(ContractError):
            validate_dataset_manifest(invalid)


if __name__ == "__main__":
    unittest.main()

