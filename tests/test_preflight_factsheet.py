"""Tests for the preflight gate, champion selector, fact sheet, and blank filling."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.champion_selector import select_champion
from agents.preflight_gate import run_preflight
from agents.fact_sheet import build_facts, fill_blanks, extract_blanks, replace_unresolved_blanks


@pytest.fixture
def run_dir(tmp_path):
    """A minimal but realistic churn run directory."""
    d = tmp_path / "run"
    d.mkdir()

    pd.DataFrame({
        "model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [0.80, 0.78],
        "Precision": [0.65, 0.62],
        "Recall": [0.56, 0.48],
        "F1": [0.60, 0.54],
        "ROC_AUC": [0.852, 0.833],
        "model_artifact": ["logistic_regression_p0.pkl", "random_forest_p0.pkl"],
    }).to_csv(d / "model_comparison.csv", index=False)

    (d / "logistic_regression_p0.pkl").write_bytes(b"fake")
    (d / "random_forest_p0.pkl").write_bytes(b"fake")

    # Confusion matrix consistent with Logistic Regression (recall .5615, precision .6542)
    (d / "confusion_matrix.json").write_text(json.dumps({
        "matrix": [[924, 111], [164, 210]],
        "true_negatives": 924, "false_positives": 111,
        "false_negatives": 164, "true_positives": 210,
        "labels": ["No Churn", "Churn"],
    }))

    pd.DataFrame({
        "actual": [1, 1, 0, 0, 1, 0],
        "predicted": [1, 0, 0, 0, 1, 1],
    }).to_csv(d / "predictions.csv", index=False)

    pd.DataFrame({
        "feature": ["cat__Contract_Month-to-month", "num__Tenure Months", "num__Monthly Charges"],
        "coefficient": [0.69, -0.79, 0.31],
        "absolute_importance": [0.69, 0.79, 0.31],
        "direction": ["increases_churn", "reduces_churn", "increases_churn"],
    }).to_csv(d / "feature_importance.csv", index=False)

    (d / "champion_model.json").write_text(json.dumps({
        "model": "random_forest_p0.pkl",  # WRONG on purpose: gate must fix it
        "primary_metric": "roc_auc", "score": 0.833,
    }))

    (d / "model_registry.json").write_text(json.dumps({"models": [
        {"model_id": "random_forest_p0", "artifact_filename": "random_forest_p0.pkl", "status": "champion"},
        {"model_id": "logistic_regression_p0", "artifact_filename": "logistic_regression_p0.pkl", "status": "evaluated"},
    ]}))

    (d / "ml_summary.json").write_text(json.dumps({
        "best_model": {"name": "Random Forest", "test_metrics": {}},
        "model_selection": {"best_model": "Random Forest"},
        "generated_artifacts": [{"file": "missing_chart.png"}],  # phantom file, on purpose
    }))

    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(d / "cleaned_dataset.csv", index=False)
    (d / "dataset_profile.json").write_text(json.dumps({"datasets": {
        "cleaned": {"file": "cleaned_dataset.csv", "rows": 999, "cols": 42},  # wrong on purpose
    }}))

    return d


class TestChampionSelector:
    def test_recall_goal_prefers_recall(self, run_dir):
        champ = select_champion(str(run_dir), goal_text="catch as many churners as possible")
        assert champ is not None
        # RF has lower ROC-AUC but neither beats LR on recall; LR wins on both here
        assert champ["model_name"] == "Logistic Regression"
        assert champ["artifact_filename"] == "logistic_regression_p0.pkl"

    def test_champion_file_rewritten(self, run_dir):
        select_champion(str(run_dir), goal_text="reduce churn")
        champ = json.loads((run_dir / "champion_model.json").read_text())
        assert champ["model"] == "logistic_regression_p0.pkl"
        assert champ["selected_by"].startswith("code")

    def test_registry_champion_updated(self, run_dir):
        select_champion(str(run_dir), goal_text="reduce churn")
        reg = json.loads((run_dir / "model_registry.json").read_text())
        statuses = {e["artifact_filename"]: e.get("status") for e in reg["models"]}
        assert statuses["logistic_regression_p0.pkl"] == "champion"
        assert statuses["random_forest_p0.pkl"] != "champion"


class TestPreflightGate:
    def test_repairs_and_never_stops(self, run_dir):
        result = run_preflight(str(run_dir), goal_text="predict churn")
        # workflow continues regardless
        assert isinstance(result["warnings"], list)
        # champion disagreement + phantom file + shape mismatch were repaired or warned
        champ = json.loads((run_dir / "champion_model.json").read_text())
        assert champ["model"] == "logistic_regression_p0.pkl"
        ml = json.loads((run_dir / "ml_summary.json").read_text())
        assert ml["best_model"]["name"] == "Logistic Regression"
        profile = json.loads((run_dir / "dataset_profile.json").read_text())
        assert profile["datasets"]["cleaned"]["rows"] == 3
        assert profile["datasets"]["cleaned"]["cols"] == 2
        # residual findings appear as warnings, not as a stop
        residual = [w for w in result["warnings"] if "missing_chart.png" in w]
        assert residual, "phantom file should surface as a warning"

    def test_confusion_matrix_mismatch_repaired_from_predictions(self, tmp_path):
        d = tmp_path
        pd.DataFrame({
            "model": ["Logistic Regression"], "Recall": [0.99], "Precision": [0.99],
            "ROC_AUC": [0.9], "model_artifact": ["lr.pkl"],
        }).to_csv(d / "model_comparison.csv", index=False)
        (d / "lr.pkl").write_bytes(b"x")
        (d / "confusion_matrix.json").write_text(json.dumps({
            "matrix": [[10, 80], [90, 20]],
            "true_negatives": 10, "false_positives": 80,
            "false_negatives": 90, "true_positives": 20,
        }))
        pd.DataFrame({"actual": [1, 0, 1, 0], "predicted": [1, 0, 1, 1]}).to_csv(d / "predictions.csv", index=False)
        result = run_preflight(str(d), goal_text="churn")
        cm = json.loads((d / "confusion_matrix.json").read_text())
        assert cm["true_positives"] == 2 and cm["false_negatives"] == 0
        assert cm.get("recomputed_by")

    def test_run_level_warnings(self, run_dir):
        result = run_preflight(str(run_dir), goal_text="churn")
        # test set of 1409 is not small; no imbalance (<10%) — expect at most near-tie/none
        assert all(isinstance(w, str) for w in result["warnings"])


class TestFactSheet:
    def test_build_and_format(self, run_dir):
        pre = run_preflight(str(run_dir), goal_text="predict churn")
        vm = {"dataset": {"rows": 7043, "cols": 33, "source_file": "cleaned_dataset.csv", "missing_pct": 0.0},
              "confusion_matrix": {"true_positives": 210, "false_positives": 111, "false_negatives": 164, "true_negatives": 924}}
        facts = build_facts(str(run_dir), vm, pre)
        assert facts["best_model"] == "Logistic Regression"
        assert facts["best_model_recall"] == "56.0%"
        assert "7,043 rows" in facts["dataset_rows"]
        assert "1. Contract: Month-to-month" in facts["drivers_increasing_outcome"]
        assert "1. Tenure Months" in facts["drivers_reducing_outcome"]
        assert "churn" not in facts["drivers_increasing_outcome"].lower().split(":")[0]  # leaky churn columns filtered
        assert "contacts per true positive" in facts["contacts_per_true_positive"]

    def test_leaky_features_excluded(self, run_dir):
        pd.DataFrame({
            "feature": ["churn_score", "customerid", "num__Tenure Months"],
            "coefficient": [5.0, 4.0, -0.5],
            "absolute_importance": [5.0, 4.0, 0.5],
            "direction": ["increases_churn", "increases_churn", "reduces_churn"],
        }).to_csv(run_dir / "feature_importance.csv", index=False)
        facts = build_facts(str(run_dir), {}, {})
        joined = facts.get("drivers_increasing_outcome", "") + facts.get("drivers_reducing_outcome", "")
        assert "churn_score" not in joined and "customerid" not in joined
        assert "Tenure Months" in joined

    def test_mandatory_warnings_included(self, run_dir):
        facts = build_facts(str(run_dir), {}, {"warnings": ["small test set", "near-tie models"]})
        assert "- small test set" in facts["mandatory_warnings"]


class TestBlankFilling:
    def test_fill_known(self):
        out = fill_blanks("Model {{best_model}} recall {{best_model_recall}}", {"best_model": "LR", "best_model_recall": "56.1%"})
        assert out["report"] == "Model LR recall 56.1%"
        assert out["unknown_blanks"] == []

    def test_unknown_blank_detected(self):
        out = fill_blanks("x {{made_up_key}}", {"best_model": "LR"})
        assert out["unknown_blanks"] == ["made_up_key"]
        assert "{{made_up_key}}" in out["report"]

    def test_replace_unresolved(self):
        assert replace_unresolved_blanks("a {{b}} c") == "a (not verified in current run) c"

    def test_extract_blanks_dedup(self):
        assert extract_blanks("{{a}} {{b}} {{a}}") == ["a", "b"]
