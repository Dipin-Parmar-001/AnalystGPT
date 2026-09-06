"""Tests for agents.consistency_checker — cross-artifact reconciliation."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.consistency_checker import run_consistency_checks


def _write_csv(path, header, n_rows=5):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(n_rows):
            f.write(",".join(str(i % 3) for _ in header) + "\n")


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


import json


class TestConsistencyChecker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_clean_run_has_no_critical_findings(self):
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"),
                   ["a", "b", "c"], n_rows=10)
        _write_json(os.path.join(self.tmp, "cleaning_summary.json"), {
            "dataset_overview": {
                "initial_shape": {"rows": 10, "columns": 3},
                "processed_shape": {"rows": 10, "columns": 3},
            },
            "data_cleaning": {"missing_values_handled": {}, "imputation_methods": {}},
        })
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertEqual(len(report.critical_findings), 0, [f.message for f in report.critical_findings])

    def test_shape_claim_matching_no_artifact_is_critical(self):
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"),
                   ["a", "b", "c"], n_rows=10)
        _write_json(os.path.join(self.tmp, "analyst_summary.json"), {
            "dataset_overview": {
                "initial_shape": {"rows": 10, "columns": 3},
                # 54 columns exists nowhere on disk -> hallucination
                "processed_shape": {"rows": 10, "columns": 54},
            },
        })
        report = run_consistency_checks(self.tmp, persist=False)
        codes = [f.code for f in report.critical_findings]
        self.assertIn("shape_mismatch", codes)

    def test_shape_claim_matching_cleaned_stage_is_not_flagged(self):
        # A cleaning summary describing the cleaned stage (44 cols) is legitimate
        # even when a feature-engineered dataset with more columns exists.
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"),
                   [f"c{i}" for i in range(44)], n_rows=10)
        _write_csv(os.path.join(self.tmp, "feature_engineered_dataset.csv"),
                   [f"c{i}" for i in range(50)], n_rows=10)
        _write_json(os.path.join(self.tmp, "cleaning_summary.json"), {
            "processed_shape": {"rows": 10, "columns": 44},
        })
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertEqual(report.critical_findings, [])

    def test_phantom_columns_detected(self):
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"), ["a", "b"], n_rows=5)
        _write_json(os.path.join(self.tmp, "analyst_summary.json"), {
            "dataset_overview": {
                "categorical_columns": ["a", "b", "Contract_Type_Phantom"],
            },
        })
        report = run_consistency_checks(self.tmp, persist=False)
        phantom = [f for f in report.findings if f.code == "phantom_columns"]
        self.assertTrue(phantom)
        self.assertIn("Contract_Type_Phantom", phantom[0].message)

    def test_stale_champion_pointer_is_critical(self):
        _write_json(os.path.join(self.tmp, "champion_model.json"),
                    {"model": "logistic_regression_p0.pkl"})
        report = run_consistency_checks(self.tmp, persist=False)
        codes = [f.code for f in report.critical_findings]
        self.assertIn("stale_champion_pointer", codes)

    def test_champion_registry_disagreement_is_critical(self):
        with open(os.path.join(self.tmp, "logistic_regression_p0.pkl"), "wb") as f:
            f.write(b"fake")
        with open(os.path.join(self.tmp, "random_forest_p0.pkl"), "wb") as f:
            f.write(b"fake")
        _write_json(os.path.join(self.tmp, "champion_model.json"),
                    {"model": "logistic_regression_p0.pkl"})
        _write_json(os.path.join(self.tmp, "model_registry.json"),
                    {"models": [
                        {"model_id": "m1", "artifact_filename": "logistic_regression_p0.pkl", "status": "superseded"},
                        {"model_id": "m2", "artifact_filename": "random_forest_p0.pkl", "status": "champion"},
                    ]})
        report = run_consistency_checks(self.tmp, persist=False)
        codes = [f.code for f in report.critical_findings]
        self.assertIn("champion_registry_disagreement", codes)

    def test_annotated_model_label_flagged(self):
        with open(os.path.join(self.tmp, "tuned_model_comparison.csv"), "w", encoding="utf-8") as f:
            f.write("model,ROC_AUC\nXGBoost (parent),0.84\n")
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertTrue(any(f.code == "annotated_model_label" for f in report.findings))

    def test_duplicate_task_summaries_flagged(self):
        content = {"task_id": 1, "dataset_overview": {"processed_shape": {"rows": 5, "columns": 2}}}
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"), ["a", "b"], n_rows=5)
        _write_json(os.path.join(self.tmp, "analyst_summary_task_1.json"), content)
        _write_json(os.path.join(self.tmp, "analyst_summary_task_3.json"), content)
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertTrue(any(f.code == "duplicate_summaries" for f in report.findings))

    def test_report_referencing_missing_artifact_is_critical(self):
        report = run_consistency_checks(
            self.tmp,
            report_text="See chart in `churn_by_contract.png` and data in `model_comparison.csv`.",
            persist=False,
        )
        codes = [f.code for f in report.critical_findings]
        self.assertIn("report_references_missing_artifacts", codes)

    def test_findings_persisted_to_consistency_report_json(self):
        _write_csv(os.path.join(self.tmp, "cleaned_dataset.csv"), ["a"], n_rows=3)
        run_consistency_checks(self.tmp, persist=True)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "consistency_report.json")))


if __name__ == "__main__":
    unittest.main()
