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


class TestCleaningIntegrityCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_raw_and_cleaned(self, raw_col_values, cleaned_col_values):
        raw = os.path.join(self.tmp, "raw_upload.csv")
        with open(raw, "w", encoding="utf-8") as f:
            f.write("customerID,Senior Citizen,tenure\n")
            for i, v in enumerate(raw_col_values):
                f.write(f"{i},{v},{i % 5}\n")
        with open(os.path.join(self.tmp, "cleaned_dataset.csv"), "w", encoding="utf-8") as f:
            f.write("customerID,Senior Citizen,tenure\n")
            for i, v in enumerate(cleaned_col_values):
                f.write(f"{i},{v},{i % 5}\n")
        return raw

    def test_senior_citizen_style_collapse_is_critical(self):
        raw = self._write_raw_and_cleaned(
            ["No", "Yes"] * 10,  # 2 values in raw
            ["0"] * 20,          # collapsed to constant in cleaned
        )
        report = run_consistency_checks(self.tmp, uploaded_files=[raw], persist=False)
        codes = [f.code for f in report.critical_findings]
        self.assertIn("column_value_collapse", codes)

    def test_legitimate_binary_mapping_is_not_flagged(self):
        raw = self._write_raw_and_cleaned(
            ["No", "Yes"] * 10,
            [0, 1] * 10,  # explicit map keeps both values
        )
        report = run_consistency_checks(self.tmp, uploaded_files=[raw], persist=False)
        codes = [f.code for f in report.critical_findings]
        self.assertNotIn("column_value_collapse", codes)

    def test_no_raw_upload_skips_check(self):
        self._write_raw_and_cleaned(["No", "Yes"] * 10, ["0"] * 20)
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertNotIn("column_value_collapse", [f.code for f in report.findings])


class TestThresholdAnalysisCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_threshold(self, rec_precision, rec_recall, curve_prec, curve_rec):
        _write_json(os.path.join(self.tmp, "threshold_analysis.json"), {
            "applicable": True,
            "default_threshold": 0.5,
            "default_metrics": {"precision": 0.51, "recall": 0.79, "f1": 0.62},
            "recommended_threshold": 0.0033,
            "recommended_metrics": {"precision": rec_precision, "recall": rec_recall, "f1": 0.42},
            "selection_rationale": "test",
            "precision_recall_curve": {
                "precision": curve_prec,
                "recall": curve_rec,
                "thresholds": [0.6, 0.32, 0.0033][: len(curve_prec)],
            },
        })

    def test_degenerate_all_positive_threshold_is_critical(self):
        self._write_threshold(
            rec_precision=0.2654, rec_recall=1.0,
            curve_prec=[0.45, 0.30, 0.2654], curve_rec=[0.85, 0.95, 1.0],
        )
        report = run_consistency_checks(self.tmp, persist=False)
        codes = [f.code for f in report.critical_findings]
        self.assertIn("degenerate_threshold_recommendation", codes)

    def test_sane_operating_point_is_not_flagged(self):
        self._write_threshold(
            rec_precision=0.45, rec_recall=0.90,
            curve_prec=[0.45, 0.2654], curve_rec=[0.90, 1.0],
        )
        report = run_consistency_checks(self.tmp, persist=False)
        codes = [f.code for f in report.findings]
        self.assertNotIn("degenerate_threshold_recommendation", codes)
        self.assertNotIn("threshold_metrics_mismatch", codes)

    def test_missing_curve_skips_degeneracy_check(self):
        _write_json(os.path.join(self.tmp, "threshold_analysis.json"), {
            "applicable": True,
            "recommended_threshold": 0.0033,
            "recommended_metrics": {"precision": 0.2654, "recall": 1.0, "f1": 0.42},
        })
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertNotIn("degenerate_threshold_recommendation", [f.code for f in report.findings])


class TestVisualizationDescriptionCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_reversed_mapping_description_is_flagged(self):
        _write_json(os.path.join(self.tmp, "feature_engineering_summary.json"), {
            "engineered_features": [
                {"feature": "contract_risk", "formula": "Month-to-month=2, One year=1, Two year=0"},
            ],
        })
        _write_json(os.path.join(self.tmp, "analyst_summary_task_3.json"), {
            "generated_visualizations": [
                {"file": "churn_by_contract.png",
                 "description": "Bar chart of churn rate by contract risk (0=Low/Month-to-month, 1=Medium/One year, 2=High/Two year)"},
            ],
        })
        report = run_consistency_checks(self.tmp, persist=False)
        mismatches = [f for f in report.findings if f.code == "viz_description_mapping_mismatch"]
        self.assertTrue(mismatches)
        self.assertIn("Month-to-month", mismatches[0].message)

    def test_correct_mapping_description_is_not_flagged(self):
        _write_json(os.path.join(self.tmp, "feature_engineering_summary.json"), {
            "engineered_features": [
                {"feature": "contract_risk", "formula": "Month-to-month=2, One year=1, Two year=0"},
            ],
        })
        _write_json(os.path.join(self.tmp, "analyst_summary_task_3.json"), {
            "generated_visualizations": [
                {"file": "churn_by_contract.png",
                 "description": "Bar chart of churn rate (0=Two year, 1=One year, 2=Month-to-month)"},
            ],
        })
        report = run_consistency_checks(self.tmp, persist=False)
        self.assertNotIn("viz_description_mapping_mismatch", [f.code for f in report.findings])

    def test_dummy_classifier_champion_emits_warning(self):
        with open(os.path.join(self.tmp, "dummy_p0.pkl"), "wb") as f:
            f.write(b"dummy")
        _write_json(os.path.join(self.tmp, "champion_model.json"), {
            "model": "dummy_p0.pkl",
            "artifact_filename": "dummy_p0.pkl",
            "model_name": "DummyClassifier",
        })
        _write_json(os.path.join(self.tmp, "model_registry.json"), {
            "models": [{"model_id": "dummy_p0", "artifact_filename": "dummy_p0.pkl", "status": "champion"}]
        })
        report = run_consistency_checks(self.tmp, persist=False)
        dummy_findings = [f for f in report.findings if f.code == "dummy_classifier_champion"]
        self.assertTrue(dummy_findings)
        self.assertEqual(dummy_findings[0].severity, "minor")
        self.assertIn("cannot be deployed or promoted as the champion", dummy_findings[0].message)


if __name__ == "__main__":
    unittest.main()

