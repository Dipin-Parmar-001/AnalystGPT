"""Focused regression checks for champion selection and preflight behavior."""

import json

import pandas as pd

from agents import champion_selector
from agents.preflight_gate import run_preflight


def _selection_dir(tmp_path):
    pd.DataFrame({
        "model": ["Logistic Regression"],
        "ROC_AUC": [0.9],
        "model_artifact": ["lr.pkl"],
    }).to_csv(tmp_path / "model_comparison.csv", index=False)
    (tmp_path / "lr.pkl").write_bytes(b"model")
    (tmp_path / "model_registry.json").write_text('{"models": []}', encoding="utf-8")
    return tmp_path


def test_registry_sync_failure_keeps_successful_champion(tmp_path, monkeypatch):
    run_dir = _selection_dir(tmp_path)

    def fail_registry(*_args, **_kwargs):
        raise OSError("registry unavailable")

    monkeypatch.setattr(champion_selector, "_update_registry_with_champion", fail_registry)
    result = champion_selector.select_initial_champion(str(run_dir))

    assert result.status == champion_selector.ChampionSelectionStatus.SUCCESS
    assert result.champion["artifact_filename"] == "lr.pkl"
    assert result.error_code == "REGISTRY_SYNC_FAILED"
    saved = json.loads((run_dir / "champion_model.json").read_text(encoding="utf-8"))
    assert saved["artifact_filename"] == "lr.pkl"


def test_preflight_does_not_reselect_or_repair_by_default(tmp_path, monkeypatch):
    run_dir = _selection_dir(tmp_path)
    original_champion = {"model": "lr.pkl", "artifact_filename": "lr.pkl", "model_name": "Logistic Regression"}
    original_registry = {"models": [{"artifact_filename": "other.pkl", "status": "champion"}]}
    (run_dir / "champion_model.json").write_text(json.dumps(original_champion), encoding="utf-8")
    (run_dir / "model_registry.json").write_text(json.dumps(original_registry), encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("preflight must not invoke champion selection")

    monkeypatch.setattr(champion_selector, "select_champion", fail_if_called)
    result = run_preflight(str(run_dir))

    assert result["champion"] == original_champion
    assert result["repairs_applied"] == []
    assert json.loads((run_dir / "model_registry.json").read_text(encoding="utf-8")) == original_registry
    assert any(f["code"] == "CHAMPION_REGISTRY_DISAGREEMENT" for f in result["findings"])


def test_preflight_reports_missing_champion_as_selection_failure(tmp_path):
    _selection_dir(tmp_path)
    result = run_preflight(str(tmp_path))
    assert result["consistency_state"] == "selection_failed"
    assert any(f["code"] == "SELECTION_FAILED" for f in result["findings"])


def test_preflight_classifies_incomplete_champion_schema(tmp_path):
    run_dir = _selection_dir(tmp_path)
    (run_dir / "champion_model.json").write_text(json.dumps({"model": "lr.pkl", "score": 0.9}), encoding="utf-8")
    result = run_preflight(str(run_dir))
    assert result["consistency_state"] == "inconsistent"
    assert any(f["code"] == "SCHEMA_ERROR" for f in result["findings"])
