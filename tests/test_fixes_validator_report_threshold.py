"""
Tests for validator data-shape handling, deterministic report generation (no thinking sentences),
and threshold selection alignment.
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.schema import load_registry, ModelRegistry
from agents.preflight_gate import run_preflight, _extract_registry_models, _sync_threshold_analysis_ml_summary
from agents.artifact_validator import extract_claimed_artifacts, find_artifact_semantic, validate_artifacts
from agents.reporter_agent import clean_thinking_sentences


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class TestValidatorDataShapeHandling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_load_registry_accepts_list_and_dict(self):
        reg_list = [
            {
                "model_id": "lr_p0",
                "model_family": "LogisticRegression",
                "artifact_filename": "lr_p0.pkl",
                "version": "p0",
                "stage": "initial_comparison",
                "task_id": 4,
                "dataset_source": "data.csv",
                "target_column": "target",
                "feature_count": 10,
                "preprocessing_description": "scaled",
                "model_parameters": {},
                "evaluation_metrics": {"recall": 0.85},
                "primary_selection_metric": "recall",
                "secondary_metrics": ["f1"],
                "status": "champion",
            }
        ]
        list_path = os.path.join(self.tmp, "registry_list.json")
        _write_json(list_path, reg_list)
        reg_obj = load_registry(list_path)
        self.assertIsInstance(reg_obj, ModelRegistry)
        self.assertEqual(len(reg_obj.models), 1)
        self.assertEqual(reg_obj.models[0].model_id, "lr_p0")

        dict_path = os.path.join(self.tmp, "registry_dict.json")
        _write_json(dict_path, {"models": reg_list})
        reg_obj2 = load_registry(dict_path)
        self.assertIsInstance(reg_obj2, ModelRegistry)
        self.assertEqual(len(reg_obj2.models), 1)

    def test_extract_registry_models_handles_various_shapes(self):
        self.assertEqual(_extract_registry_models(None), [])
        self.assertEqual(_extract_registry_models("malformed"), [])
        self.assertEqual(_extract_registry_models([]), [])
        self.assertEqual(_extract_registry_models([{"a": 1}, "not_dict"]), [{"a": 1}])
        self.assertEqual(_extract_registry_models({"models": [{"b": 2}]}), [{"b": 2}])
        self.assertEqual(_extract_registry_models({"models": "not_list"}), [])

    def test_preflight_gate_with_list_registry_does_not_crash(self):
        reg_list = [
            {
                "model_id": "logisticregression_p0",
                "model_family": "LogisticRegression",
                "artifact_filename": "logisticregression_p0.pkl",
                "status": "champion",
                "evaluation_metrics": {"recall": 0.8772, "precision": 0.0323},
            }
        ]
        _write_json(os.path.join(self.tmp, "model_registry.json"), reg_list)
        _write_json(os.path.join(self.tmp, "champion_model.json"), {
            "model": "logisticregression_p0.pkl",
            "model_name": "LogisticRegression",
            "artifact_filename": "logisticregression_p0.pkl",
            "primary_metric": "recall",
            "score": 0.8772,
            "selection_basis": "highest recall",
            "selected_by": "code (champion_selector)",
        })
        # Dummy pkl file
        open(os.path.join(self.tmp, "logisticregression_p0.pkl"), "w").close()

        # Run preflight with repair=True - must NOT crash with 'list' object has no attribute 'get'
        res = run_preflight(self.tmp, repair=True)
        self.assertIsInstance(res, dict)
        detail_messages = [f.get("message", "") for f in res.get("findings", [])]
        for msg in detail_messages:
            self.assertNotIn("has no attribute 'get'", msg)

    def test_artifact_validator_extract_claimed_artifacts_list_input(self):
        # Passes a raw list of filenames or mixed objects
        claimed = extract_claimed_artifacts(["cleaned_dataset.csv", "feature_importance.csv"])
        self.assertIn("cleaned_dataset.csv", claimed)
        self.assertIn("feature_importance.csv", claimed)

        # Non-dict and null safety
        self.assertEqual(extract_claimed_artifacts(None), [])
        self.assertEqual(extract_claimed_artifacts("string"), [])

    def test_artifact_validator_extract_claimed_artifacts_null_safety(self):
        # Handles null artifact in model_artifacts and null file in generated_artifacts
        response = {
            "summary_data": {
                "model_artifacts": [
                    {"model": "Majority Class Baseline", "artifact": None},
                    {"model": "logistic_regression", "artifact": "logistic_regression_p0.pkl"}
                ],
                "generated_artifacts": [
                    None,
                    "",
                    {"file": None},
                    {"file": "model_comparison.csv"}
                ]
            }
        }
        claimed = extract_claimed_artifacts(response)
        self.assertIn("logistic_regression_p0.pkl", claimed)
        self.assertIn("model_comparison.csv", claimed)
        self.assertNotIn(None, claimed)
        self.assertNotIn("", claimed)

        # validate_artifacts must not throw TypeError when entries contain None
        result = validate_artifacts(
            agent_response=response,
            output_dir=self.tmp,
            agent_name="MLAgent",
            task_type="model_comparison"
        )
        self.assertIsInstance(result.missing_artifacts, list)

    def test_find_artifact_semantic_matching(self):
        # Create a per-task variant in directory
        variant_name = "analyst_summary_task_1.json"
        open(os.path.join(self.tmp, variant_name), "w").close()
        found = find_artifact_semantic(self.tmp, "analyst_summary.json")
        self.assertEqual(found, variant_name)


class TestDeterministicReportGeneration(unittest.TestCase):
    def test_clean_thinking_sentences_removes_monologue(self):
        leaked_text = (
            "**Factors increasing fraud probability (positive coefficients):**\n"
            "1. V4 (coefficient +1.318)\n"
            "2. V22 (coefficient +0.496)\n"
            "3. V1 (coefficient -0.383? Wait — V1 coefficient is negative in feature_importance but listed as increasing in FACT_SHEET. Let me check... Actually the FACT_SHEET lists V1 as increasing. The feature_importance.csv shows V1 coefficient -0.383? No, looking at VERIFIED_METRICS.top_features, V1 is not in top 10. The FACT_SHEET drivers_increasing_outcome lists V1, V11, Time_hour. I'll use the FACT_SHEET as authoritative for the driver lists.)"
        )
        cleaned = clean_thinking_sentences(leaked_text)
        self.assertNotIn("Wait —", cleaned)
        self.assertNotIn("Let me check", cleaned)
        self.assertNotIn("Actually the FACT_SHEET", cleaned)
        self.assertIn("1. V4 (coefficient +1.318)", cleaned)
        self.assertIn("2. V22 (coefficient +0.496)", cleaned)

    def test_clean_thinking_sentences_preserves_clean_report(self):
        clean_text = (
            "## Executive Summary\n\n"
            "This report summarizes the findings. The champion model achieves 0.88 recall.\n"
            "Recommendations:\n"
            "1. Deploy champion with default threshold 0.5.\n"
            "2. Collect additional labeled examples."
        )
        cleaned = clean_thinking_sentences(clean_text)
        self.assertEqual(cleaned, clean_text)


class TestThresholdSelectionGateAndSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_threshold_analysis_and_ml_summary_sync(self):
        champ = {
            "model": "logisticregression_p0.pkl",
            "artifact_filename": "logisticregression_p0.pkl",
            "primary_metric": "recall",
        }
        _write_json(os.path.join(self.tmp, "threshold_analysis.json"), {
            "model_artifact": "logisticregression_p0.pkl",
            "default_threshold": 0.5,
            "default_metrics": {"recall": 0.8772, "precision": 0.0323},
            "recommended_threshold": 0.9595,
            "recommended_metrics": {"recall": 0.807, "precision": 0.293},
            "selection_rationale": "Optimized to keep recall >= 0.80 while improving precision.",
        })
        _write_json(os.path.join(self.tmp, "ml_summary.json"), {
            "threshold_analysis": {
                "default_threshold": 0.5,
                "recommended_threshold": 1.0,
                "recommended_metrics": {"recall": 0.6667, "precision": 0.95},
                "selection_rationale": "Maximized F1.",
            }
        })

        repairs = _sync_threshold_analysis_ml_summary(self.tmp, champ)
        self.assertTrue(len(repairs) > 0)

        # Verify ml_summary.json updated to match threshold_analysis.json
        with open(os.path.join(self.tmp, "ml_summary.json"), "r") as f:
            updated_summary = json.load(f)
        self.assertEqual(updated_summary["threshold_analysis"]["recommended_threshold"], 0.9595)
        self.assertEqual(updated_summary["threshold_analysis"]["recommended_metrics"]["recall"], 0.807)


if __name__ == "__main__":
    unittest.main()
