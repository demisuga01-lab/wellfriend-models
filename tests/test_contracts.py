"""Schema, artifact, registry, synthetic, metric, and smoke-flow coverage for MP5."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wellfriend_models.evaluation.evaluate import main as evaluation_main
from wellfriend_models.export.export import main as export_main
from wellfriend_models.metrics import (
    binary_iou,
    cer,
    corner_mae,
    dice,
    mask_precision_recall_f1,
    psnr,
    ssim,
    wer,
)
from wellfriend_models.models import ClassicalRestorationBaseline
from wellfriend_models.registry import validate_artifact_directory, write_placeholder_artifact
from wellfriend_models.registry.index import validate_registry_index
from wellfriend_models.schemas import (
    ContractError,
    validate_dataset_manifest,
    validate_model_components,
)
from wellfriend_models.synthetic import SyntheticDocumentGenerator
from wellfriend_models.training.train import load_config, run_baseline_experiment

ROOT = Path(__file__).resolve().parents[1]


def dataset_manifest() -> dict[str, object]:
    """Provide a fully specified minimal synthetic DatasetManifest fixture."""
    sample = SyntheticDocumentGenerator(width=32, height=24, seed=3).generate(
        "perspective_quadrilateral"
    )
    return {
        "schema_version": 1,
        "dataset_id": "synthetic-pages-v1",
        "name": "synthetic-pages",
        "version": "1.0.0",
        "description": "deterministic test data",
        "domain": "document",
        "tasks": ["document_segmentation"],
        "license": {
            "name": "Apache-2.0",
            "status": "synthetic",
            "reference": "generator://wellfriend",
        },
        "source_url": "https://github.com/demisuga01-lab/wellfriend-models",
        "citation": "Wellfriend synthetic generator",
        "allowed_uses": ["testing"],
        "redistribution": "generated on demand",
        "provenance": {"generator": "SyntheticDocumentGenerator", "seed": 3},
        "splits": {"train": {"seed": 3}},
        "samples": [
            {
                "sample_id": sample.sample_id,
                "relative_path": f"images/{sample.sample_id}.pgm",
                "sha256": sample.sha256(),
                "width": sample.width,
                "height": sample.height,
                "channels": 1,
                "pixel_format": "Gray8",
                "domain": "document",
                "split": "train",
                "annotations": sample.annotations(),
                "source": "synthetic",
                "license_ref": "Apache-2.0",
            }
        ],
        "annotations": ["quad", "mask", "keypoints", "quality_labels", "cleanup_mask"],
        "transforms": [],
        "hashes": {"manifest": "generated during test"},
        "warnings": ["synthetic only"],
    }


class DatasetContractTests(unittest.TestCase):
    def test_valid_dataset_manifest_passes(self) -> None:
        validate_dataset_manifest(dataset_manifest())

    def test_missing_required_dataset_field_fails(self) -> None:
        value = dataset_manifest()
        del value["provenance"]
        with self.assertRaises(ContractError):
            validate_dataset_manifest(value)

    def test_unknown_dataset_license_is_blocked(self) -> None:
        value = dataset_manifest()
        value["license"] = {"name": "unknown", "status": "unknown", "reference": "none"}
        with self.assertRaises(ContractError):
            validate_dataset_manifest(value)


class SyntheticTests(unittest.TestCase):
    def test_generation_is_deterministic_and_annotations_are_bounded(self) -> None:
        first = SyntheticDocumentGenerator(seed=31).generate("perspective_quadrilateral")
        second = SyntheticDocumentGenerator(seed=31).generate("perspective_quadrilateral")
        self.assertEqual(first.pixels, second.pixels)
        self.assertEqual(len(first.mask), first.width * first.height)
        for x, y in first.quad:
            self.assertGreaterEqual(x, -10)
            self.assertLessEqual(x, first.width)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, first.height)

    def test_synthetic_quality_variants_differ(self) -> None:
        generator = SyntheticDocumentGenerator(seed=37)
        self.assertNotEqual(
            generator.generate("blurred_page").pixels, generator.generate("glare_patch").pixels
        )
        self.assertEqual(generator.generate("blurred_page").quality_labels["blur"], 1.0)
        self.assertEqual(generator.generate("glare_patch").quality_labels["glare"], 1.0)


class MetricTests(unittest.TestCase):
    def test_known_mask_and_corner_metrics(self) -> None:
        self.assertEqual(binary_iou([1, 0, 1], [1, 1, 0]), 1 / 3)
        self.assertEqual(dice([1, 0, 1], [1, 1, 0]), 0.5)
        self.assertEqual(mask_precision_recall_f1([1, 0], [1, 0])["f1"], 1.0)
        self.assertEqual(corner_mae([(0, 0)], [(3, 4)]), 5.0)

    def test_text_and_signal_metrics(self) -> None:
        self.assertEqual(cer("axc", "abc"), 1 / 3)
        self.assertEqual(wer("hello brave world", "hello world"), 0.5)
        self.assertEqual(psnr([1, 2], [1, 2]), float("inf"))
        self.assertAlmostEqual(ssim([1, 2, 3], [1, 2, 3]), 1.0)


class ArtifactRegistryTests(unittest.TestCase):
    def test_placeholder_artifact_only_validates_when_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = write_placeholder_artifact(
                Path(raw_directory) / "placeholder",
                family="document-detector",
                task="document_segmentation",
            )
            self.assertEqual(
                validate_artifact_directory(path, allow_placeholder=True)["status"], "placeholder"
            )
            with self.assertRaises(ContractError):
                validate_artifact_directory(path)

    def test_bad_checksum_and_invalid_task_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = write_placeholder_artifact(
                Path(raw_directory) / "placeholder",
                family="document-detector",
                task="document_segmentation",
            )
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            manifest["task"] = "invalid-task"
            components = {
                name: json.loads((path / filename).read_text(encoding="utf-8"))
                for name, filename in {
                    "manifest": "manifest.json",
                    "preprocess": "preprocess.json",
                    "postprocess": "postprocess.json",
                    "labels": "labels.json",
                    "checksums": "checksums.json",
                    "metrics": "metrics.json",
                }.items()
            }
            components["manifest"] = manifest
            with self.assertRaises(ContractError):
                validate_model_components(components, production=False)
            manifest["task"] = "document_segmentation"
            manifest["status"] = "released"
            manifest["production_ready"] = True
            (path / "model.onnx").write_bytes(b"not-an-onnx-file-but-a-checksum-test")
            checksums = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
            checksums["files"] = {"model.onnx": "0" * 64}
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (path / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
            with self.assertRaises(ContractError):
                validate_artifact_directory(path)

    def test_invalid_input_output_specs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = write_placeholder_artifact(
                Path(raw_directory) / "placeholder",
                family="document-detector",
                task="document_segmentation",
            )
            components = {
                name: json.loads((path / filename).read_text(encoding="utf-8"))
                for name, filename in {
                    "manifest": "manifest.json",
                    "preprocess": "preprocess.json",
                    "postprocess": "postprocess.json",
                    "labels": "labels.json",
                    "checksums": "checksums.json",
                    "metrics": "metrics.json",
                }.items()
            }
            components["manifest"]["input_spec"] = {}
            with self.assertRaises(ContractError):
                validate_model_components(components, production=False)

    def test_registry_index_and_placeholders_validate(self) -> None:
        results = validate_registry_index(ROOT / "registry" / "index.json")
        self.assertEqual(len(results), 6)
        self.assertTrue(all(result["production_ready"] is False for result in results))


class BaselineFlowTests(unittest.TestCase):
    def test_training_smoke_for_three_document_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            for name in (
                "document_segmentation_synthetic.json",
                "document_corners_synthetic.json",
                "document_quality_synthetic.json",
            ):
                config = load_config(ROOT / "configs" / "smoke" / name)
                output = Path(raw_directory) / config["experiment_id"]
                result = run_baseline_experiment(config, output, root=ROOT)
                self.assertEqual(result["status"], "baseline_complete")
                self.assertTrue((output / "metrics.json").is_file())

    def test_evaluation_and_restoration_baseline(self) -> None:
        config_path = ROOT / "configs" / "smoke" / "document_segmentation_synthetic.json"
        with tempfile.TemporaryDirectory() as raw_directory:
            import sys
            from unittest.mock import patch

            with patch.object(
                sys,
                "argv",
                [
                    "evaluate",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    raw_directory,
                    "--write-predictions-manifest",
                ],
            ):
                self.assertEqual(evaluation_main(), 0)
            self.assertTrue((Path(raw_directory) / "metrics.json").is_file())
            sample = SyntheticDocumentGenerator().generate("shadow_gradient")
            restored = ClassicalRestorationBaseline().process(sample, "binarization_baseline")
            self.assertEqual(len(restored["pixels"]), len(sample.pixels))

    def test_export_emits_a_valid_placeholder_artifact(self) -> None:
        config_path = ROOT / "configs" / "export" / "document_detector_baseline.json"
        with tempfile.TemporaryDirectory() as raw_directory:
            import sys
            from unittest.mock import patch

            with patch.object(
                sys,
                "argv",
                ["export", "--config", str(config_path), "--output-dir", raw_directory],
            ):
                self.assertEqual(export_main(), 0)
            self.assertEqual(
                validate_artifact_directory(Path(raw_directory), allow_placeholder=True)["status"],
                "placeholder",
            )


if __name__ == "__main__":
    unittest.main()
