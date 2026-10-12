"""Tests for Post-ML Verification and Champion Consistency Gate."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.preflight_gate import verify_champion_consistency, normalize_model_family


def test_normalize_model_family():
    assert normalize_model_family("logistic_regression_p0.pkl") == "logistic_regression"
    assert normalize_model_family("logistic_regression_p1.pkl") == "logistic_regression"
    assert normalize_model_family("logistic_regression_p1_tuned.pkl") == "logistic_regression"
    assert normalize_model_family("Logistic Regression (Tuned)") == "logistic_regression"
    assert normalize_model_family("random_forest_p0.pkl") == "random_forest"
    assert normalize_model_family("Random Forest") == "random_forest"
    assert normalize_model_family("histgradboost_p0.pkl") == "histgradboost"
    assert normalize_model_family("histgradboost_p1_tuned.pkl") == "histgradboost"
    assert normalize_model_family("HistGradientBoostingClassifier") == "histgradboost"


def test_post_ml_verification_corrupted_model_registry(tmp_path):
    output_dir = str(tmp_path)
    # Write a truncated/corrupted model_registry.json
    with open(os.path.join(output_dir, "model_registry.json"), "w", encoding="utf-8") as f:
        f.write('{"models": {"lr": {"metrics": { "tn": ')

    (tmp_path / "logistic_regression_p0.pkl").write_bytes(b"dummy")
    with open(os.path.join(output_dir, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "logistic_regression_p0.pkl",
            "artifact_filename": "logistic_regression_p0.pkl",
            "primary_metric": "recall",
            "score": 0.89
        }, f)

    res = verify_champion_consistency(output_dir, task_type="model_comparison", repair=False)
    assert not res["consistent"]
    codes = [f.code for f in res["findings"]]
    assert "CORRUPTED_MODEL_REGISTRY" in codes


def test_post_ml_verification_hpt_family_mismatch(tmp_path):
    output_dir = str(tmp_path)
    prev_champ = {
        "artifact_filename": "logistic_regression_p0.pkl",
        "model_name": "Logistic Regression",
        "model_family": "logistic_regression",
        "primary_metric": "recall",
        "score": 0.8947
    }

    (tmp_path / "logistic_regression_p0.pkl").write_bytes(b"dummy")
    (tmp_path / "histgradboost_p1_tuned.pkl").write_bytes(b"dummy")

    with open(os.path.join(output_dir, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "histgradboost_p1_tuned.pkl",
            "artifact_filename": "histgradboost_p1_tuned.pkl",
            "primary_metric": "recall",
            "score": 0.6667
        }, f)

    with open(os.path.join(output_dir, "hyperparameter_tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "parent_model": {"artifact": "logistic_regression_p0.pkl"},
            "challenger_model": {"artifact": "histgradboost_p1_tuned.pkl"},
            "comparison": {
                "baseline": {"recall": 0.1228},
                "tuned": {"recall": 0.6667}
            },
            "champion_decision": {
                "previous_champion": "logistic_regression_p0.pkl",
                "new_champion": "histgradboost_p1_tuned.pkl",
                "champion_changed": True
            }
        }, f)

    res = verify_champion_consistency(
        output_dir,
        task_type="hyperparameter_tuning",
        pre_task_champion=prev_champ,
        primary_metric="recall",
        repair=False
    )
    assert not res["consistent"]
    codes = [f.code for f in res["findings"]]
    assert "CHAMPION_FAMILY_MISMATCH" in codes
    assert any("HPT can ONLY tune the existing champion model family" in f.message for f in res["findings"])


def test_post_ml_verification_hpt_false_improvement_claim(tmp_path):
    output_dir = str(tmp_path)
    prev_champ = {
        "artifact_filename": "logistic_regression_p0.pkl",
        "model_name": "Logistic Regression",
        "model_family": "logistic_regression",
        "primary_metric": "recall",
        "score": 0.8947
    }

    (tmp_path / "logistic_regression_p0.pkl").write_bytes(b"dummy")
    (tmp_path / "logistic_regression_p1.pkl").write_bytes(b"dummy")

    # Agent incorrectly claimed challenger won despite worse score (0.6667 vs 0.8947)
    with open(os.path.join(output_dir, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "logistic_regression_p1.pkl",
            "artifact_filename": "logistic_regression_p1.pkl",
            "primary_metric": "recall",
            "score": 0.6667
        }, f)

    with open(os.path.join(output_dir, "hyperparameter_tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "parent_model": {"artifact": "logistic_regression_p0.pkl"},
            "challenger_model": {"artifact": "logistic_regression_p1.pkl"},
            "comparison": {
                "baseline": {"recall": 0.8947},
                "tuned": {"recall": 0.6667}
            },
            "champion_decision": {
                "previous_champion": "logistic_regression_p0.pkl",
                "new_champion": "logistic_regression_p1.pkl",
                "champion_changed": True
            }
        }, f)

    res = verify_champion_consistency(
        output_dir,
        task_type="hyperparameter_tuning",
        pre_task_champion=prev_champ,
        primary_metric="recall",
        repair=False
    )
    assert not res["consistent"]
    codes = [f.code for f in res["findings"]]
    assert "FALSE_CHAMPION_IMPROVEMENT_CLAIM" in codes


def test_post_ml_verification_hpt_valid_improvement(tmp_path):
    output_dir = str(tmp_path)
    prev_champ = {
        "artifact_filename": "logistic_regression_p0.pkl",
        "model_name": "Logistic Regression",
        "model_family": "logistic_regression",
        "primary_metric": "recall",
        "score": 0.8500
    }

    (tmp_path / "logistic_regression_p0.pkl").write_bytes(b"dummy")
    (tmp_path / "logistic_regression_p1.pkl").write_bytes(b"dummy")

    # Tuned version improved (0.9200 vs 0.8500)
    with open(os.path.join(output_dir, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "logistic_regression_p1.pkl",
            "artifact_filename": "logistic_regression_p1.pkl",
            "primary_metric": "recall",
            "score": 0.9200
        }, f)

    with open(os.path.join(output_dir, "model_registry.json"), "w", encoding="utf-8") as f:
        json.dump({"models": [
            {"model_id": "logistic_regression_p0", "artifact_filename": "logistic_regression_p0.pkl", "status": "evaluated"},
            {"model_id": "logistic_regression_p1", "artifact_filename": "logistic_regression_p1.pkl", "status": "champion"}
        ]}, f)

    with open(os.path.join(output_dir, "hyperparameter_tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "parent_model": {"artifact": "logistic_regression_p0.pkl"},
            "challenger_model": {"artifact": "logistic_regression_p1.pkl"},
            "comparison": {
                "baseline": {"recall": 0.8500},
                "tuned": {"recall": 0.9200}
            },
            "champion_decision": {
                "previous_champion": "logistic_regression_p0.pkl",
                "new_champion": "logistic_regression_p1.pkl",
                "champion_changed": True
            }
        }, f)

    res = verify_champion_consistency(
        output_dir,
        task_type="hyperparameter_tuning",
        pre_task_champion=prev_champ,
        primary_metric="recall",
        repair=False
    )
    assert res["consistent"]
    assert len([f for f in res["findings"] if f.severity == "error"]) == 0


def test_post_ml_verification_hpt_retains_champion_when_not_improved(tmp_path):
    output_dir = str(tmp_path)
    prev_champ = {
        "artifact_filename": "logistic_regression_p0.pkl",
        "model_name": "Logistic Regression",
        "model_family": "logistic_regression",
        "primary_metric": "recall",
        "score": 0.8947
    }

    (tmp_path / "logistic_regression_p0.pkl").write_bytes(b"dummy")
    (tmp_path / "logistic_regression_p1.pkl").write_bytes(b"dummy")

    # Tuned version did not beat previous champion (0.8000 vs 0.8947), so old champion retained
    with open(os.path.join(output_dir, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "logistic_regression_p0.pkl",
            "artifact_filename": "logistic_regression_p0.pkl",
            "primary_metric": "recall",
            "score": 0.8947
        }, f)

    with open(os.path.join(output_dir, "model_registry.json"), "w", encoding="utf-8") as f:
        json.dump({"models": [
            {"model_id": "logistic_regression_p0", "artifact_filename": "logistic_regression_p0.pkl", "status": "champion"},
            {"model_id": "logistic_regression_p1", "artifact_filename": "logistic_regression_p1.pkl", "status": "evaluated"}
        ]}, f)

    with open(os.path.join(output_dir, "hyperparameter_tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "parent_model": {"artifact": "logistic_regression_p0.pkl"},
            "challenger_model": {"artifact": "logistic_regression_p1.pkl"},
            "comparison": {
                "baseline": {"recall": 0.8947},
                "tuned": {"recall": 0.8000}
            },
            "champion_decision": {
                "previous_champion": "logistic_regression_p0.pkl",
                "new_champion": "logistic_regression_p0.pkl",
                "champion_changed": False
            }
        }, f)

    res = verify_champion_consistency(
        output_dir,
        task_type="hyperparameter_tuning",
        pre_task_champion=prev_champ,
        primary_metric="recall",
        repair=False
    )
    assert res["consistent"]
    assert len([f for f in res["findings"] if f.severity == "error"]) == 0


def test_post_ml_verification_no_false_positive_on_unhyphenated_naming(tmp_path):
    output_dir = str(tmp_path)
    (tmp_path / "logisticregression_p0.pkl").write_bytes(b"dummy")

    with open(os.path.join(output_dir, "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "logistic_regression_p0.pkl",
            "artifact_filename": "logistic_regression_p0.pkl",
            "model_name": "Logistic Regression",
            "primary_metric": "recall",
            "score": 0.8947
        }, f)

    with open(os.path.join(output_dir, "model_registry.json"), "w", encoding="utf-8") as f:
        json.dump({"models": [
            {"model_id": "logistic_regression", "artifact_filename": "logisticregression_p0.pkl", "status": "champion"}
        ]}, f)

    with open(os.path.join(output_dir, "ml_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "champion": {"model_artifact": "logisticregression_p0.pkl"},
            "best_model": {"name": "LogisticRegression"}
        }, f)

    res = verify_champion_consistency(
        output_dir,
        task_type="model_comparison",
        repair=False
    )
    assert res["consistent"]
    assert len([f for f in res["findings"] if f.severity == "error"]) == 0
