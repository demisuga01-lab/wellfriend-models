"""MP6 contracts: mobile profiles, optimization evidence, registry promotion, and ScanBench."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from wellfriend_models.benchmarks.scanbench import (
    benchmark_artifact,
    validate_scanbench_report,
    write_scanbench_report,
)
from wellfriend_models.distillation import (
    build_distillation_report,
    distillation_loss,
    validate_distillation_config,
    validate_distillation_report,
)
from wellfriend_models.evaluation.ocr_aware import ocr_aware_restoration_metrics
from wellfriend_models.mobile import (
    load_device_profile,
    validate_device_profile,
    validate_mobile_candidate_config,
    validate_tiling_policy,
)
from wellfriend_models.mobile.campaign import main as campaign_main
from wellfriend_models.pruning import (
    build_pruning_report,
    validate_pruning_config,
    validate_pruning_report,
)
from wellfriend_models.quantization import (
    build_quantization_report,
    validate_quantization_config,
    validate_quantization_report,
)
from wellfriend_models.registry import validate_artifact_directory
from wellfriend_models.registry.index import validate_registry_index
from wellfriend_models.restoration import validate_docres_mobile_boundary
from wellfriend_models.schemas import ContractError

ROOT = Path(__file__).resolve().parents[1]


class DeviceProfileTests(unittest.TestCase):
    def test_all_profiles_and_tiling_policy_validate(self) -> None:
        for name in ("low", "mid", "high", "server", "web", "unknown"):
            profile = load_device_profile(ROOT / "configs" / "device-profiles" / f"{name}.json")
            self.assertEqual(profile["device_class"], name)
        policy = {
            "mode": "overlap_tile",
            "input_resolution": [256, 256],
            "max_resolution": [2048, 2048],
            "tile_size": [256, 256],
            "overlap": 32,
            "blend_policy": "feather",
            "coordinate_remap_policy": "tile_offset",
        }
        validate_tiling_policy(policy)
        invalid_policy = copy.deepcopy(policy)
        invalid_policy["mode"] = "made_up"
        with self.assertRaises(ContractError):
            validate_tiling_policy(invalid_policy)

    def test_invalid_profile_values_are_rejected(self) -> None:
        profile = load_device_profile(ROOT / "configs" / "device-profiles" / "low.json")
        bad_precision = copy.deepcopy(profile)
        bad_precision["precision"] = ["bf16"]
        with self.assertRaises(ContractError):
            validate_device_profile(bad_precision)
        bad_size = copy.deepcopy(profile)
        bad_size["max_model_size_mb"] = 0
        with self.assertRaises(ContractError):
            validate_device_profile(bad_size)
        bad_runtime = copy.deepcopy(profile)
        bad_runtime["runtime_targets"] = ["unknown-runtime"]
        with self.assertRaises(ContractError):
            validate_device_profile(bad_runtime)

    def test_all_mobile_family_configs_remain_experimental(self) -> None:
        configs = sorted((ROOT / "configs" / "mobile").glob("*/*.json"))
        self.assertEqual(len(configs), 15)
        for path in configs:
            config = json.loads(path.read_text())
            validate_mobile_candidate_config(config)
            self.assertEqual(config["status"], "experimental")


class OptimizationContractTests(unittest.TestCase):
    def test_distillation_report_and_audit_gated_teacher(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "distillation" / "document_detector_mobile.json").read_text()
        )
        validate_distillation_config(config)
        report = build_distillation_report(
            config,
            {
                "distillation_loss": 0.0,
                "teacher": {"iou": 1.0},
                "student": {"iou": 1.0},
                "delta": {"iou": 0.0},
            },
        )
        validate_distillation_report(report)
        cached = copy.deepcopy(config)
        cached["teacher"] = {"type": "cached_predictions", "id": "synthetic-cache-v1"}
        validate_distillation_config(cached)
        self.assertAlmostEqual(
            distillation_loss("document_segmentation", [0.5], [0.5]), 0.69314718056
        )
        external = copy.deepcopy(config)
        external["teacher"] = {
            "type": "external_audit_gated",
            "id": "DocRes",
            "audit_status": "audit-gated",
        }
        report = build_distillation_report(
            external,
            {"distillation_loss": 0.0, "teacher": {}, "student": {}, "delta": {}},
        )
        self.assertFalse(report["student_manifest"]["production_ready"])

    def test_quantization_and_pruning_reports_validate(self) -> None:
        quant_config = json.loads(
            (ROOT / "configs" / "quantization" / "document_detector_low.json").read_text()
        )
        validate_quantization_config(quant_config)
        quant_report = build_quantization_report(quant_config, size_before=100, size_after=60)
        validate_quantization_report(quant_report)
        self.assertEqual(quant_report["size_before_bytes"] - quant_report["size_after_bytes"], 40)
        self.assertEqual(quant_report["known_unsupported_ops"], [])
        pruning_config = json.loads(
            (ROOT / "configs" / "pruning" / "document_detector_low.json").read_text()
        )
        validate_pruning_config(pruning_config)
        pruning_report = build_pruning_report(pruning_config)
        validate_pruning_report(pruning_report)
        self.assertEqual(pruning_report["metric_delta"], {})


class MobileRegistryTests(unittest.TestCase):
    def test_experimental_registry_entries_validate_and_are_not_production(self) -> None:
        results = validate_registry_index(ROOT / "registry" / "index.json")
        self.assertEqual(len(results), 21)
        experimental = ROOT / "registry" / "document-detector" / "low-mobile-experimental"
        result = validate_artifact_directory(experimental, allow_nonproduction=True)
        self.assertEqual(result["status"], "experimental")
        self.assertFalse(result["production_ready"])
        with self.assertRaises(ContractError):
            validate_artifact_directory(experimental)

    def test_no_weight_or_incomplete_production_promotion_is_rejected(self) -> None:
        path = ROOT / "registry" / "document-detector" / "low-mobile-experimental"
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["status"] = "production_ready"
        manifest["production_ready"] = True
        manifest["weights_included"] = False
        components = {
            "manifest": manifest,
            "preprocess": json.loads((path / "preprocess.json").read_text()),
            "postprocess": json.loads((path / "postprocess.json").read_text()),
            "labels": json.loads((path / "labels.json").read_text()),
            "checksums": json.loads((path / "checksums.json").read_text()),
            "metrics": json.loads((path / "metrics.json").read_text()),
        }
        from wellfriend_models.schemas import validate_model_components

        with self.assertRaises(ContractError):
            validate_model_components(components, production=True)


class ScanBenchAndRestorationTests(unittest.TestCase):
    def test_scanbench_json_and_markdown_reports(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "benchmarks" / "document_mobile_smoke.json").read_text()
        )
        report = benchmark_artifact(config, root=ROOT)
        validate_scanbench_report(report)
        self.assertTrue(report["metrics"]["device_profile_fit"]["fits_size_target"])
        with tempfile.TemporaryDirectory() as raw_directory:
            json_path, markdown_path = write_scanbench_report(report, Path(raw_directory))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())

    def test_mobile_campaign_emits_required_smoke_evidence(self) -> None:
        import sys
        from unittest.mock import patch

        config = ROOT / "configs" / "mobile" / "document-detector" / "low.json"
        with tempfile.TemporaryDirectory() as raw_directory:
            with patch.object(
                sys,
                "argv",
                ["campaign", "--config", str(config), "--output-dir", raw_directory],
            ):
                self.assertEqual(campaign_main(), 0)
            for name in (
                "metrics.json",
                "artifact-manifest.json",
                "reproducibility.json",
                "scanbench-model-report.json",
                "scanbench-model-report.md",
            ):
                self.assertTrue((Path(raw_directory) / name).is_file())

    def test_ocr_aware_hooks_are_observation_based(self) -> None:
        metrics = ocr_aware_restoration_metrics(
            baseline_text="scam",
            restored_text="scan",
            reference_text="scan",
            baseline_edges=[1.0, 2.0],
            restored_edges=[1.0, 2.0],
            expected_strokes=[1, 0],
            restored_strokes=[1, 0],
        )
        self.assertLess(metrics["cer_delta_after_restoration"], 0)
        self.assertEqual(metrics["foreground_stroke_preservation"], 1.0)

    def test_docres_mobile_boundary_remains_audit_gated(self) -> None:
        boundary = json.loads(
            (ROOT / "document" / "restoration" / "mp6-mobile-boundary.json").read_text()
        )
        validate_docres_mobile_boundary(boundary)
        self.assertFalse(boundary["student_registry_slot"]["production_ready"])


if __name__ == "__main__":
    unittest.main()
