"""
Champion Selector - AnalystGPT.

Deterministic, code-owned best-model selection. After ML/tuning tasks finish,
the orchestrator calls `select_initial_champion()` once after initial model comparison,
then `evaluate_challenger()` after every HPT task. The ML agent never decides
(or overwrites) the champion.

The selection rule is derived from the user's goal text:
- Goals about catching churners / recall ("catch as many churners as possible",
  "don't miss churners") prioritize Recall, then F1, then ROC-AUC.
- Everything else prioritizes ROC-AUC, then PR-AUC, then F1, then Accuracy.
- Regression metrics (RMSE/MAE/MSE) are minimized when no classification
  metric exists.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from agents.logger import setup_logfire
from agents.schema import ModelRegistry, ChampionModel, ModelEntry, ModelStatus, ModelStage, load_registry, save_registry, load_champion, save_champion

try:
    import logfire
except ImportError:
    logfire = None

setup_logfire()

RECALL_PRIORITY = ["recall", "f1", "roc_auc", "roc-auc", "pr_auc", "pr-auc", "accuracy"]
DEFAULT_PRIORITY = ["roc_auc", "roc-auc", "pr_auc", "pr-auc", "f1", "recall", "accuracy"]
ASCENDING_METRICS = {"rmse", "mae", "mse", "log_loss"}

_MODEL_NAME_KEYS = ["model", "algorithm", "name", "model_name"]
_ARTIFACT_KEYS = ["model_artifact", "artifact", "artifact_filename", "model_file"]


def _norm_metric_key(col: str) -> str:
    return col.strip().lower().replace(" ", "").replace("-", "_")


def _norm_model_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace("(tuned)", "").replace("(baseline)", "").strip("_")


def _extract_phase_and_family(model_name: str) -> tuple[str, str]:
    """
    Extract model family and phase from model name.

    Examples:
        "Logistic Regression" -> ("logistic_regression", "p0")
        "Logistic Regression (Tuned)" -> ("logistic_regression", "p1")
        "logistic_regression_p0" -> ("logistic_regression", "p0")
        "logistic_regression_p1_tuned" -> ("logistic_regression", "p1")
        "SVM p0" -> ("svm", "p0")
        "SVM p1" -> ("svm", "p1")

    Returns:
        (family, phase) where phase is "p0", "p1", "p2", etc.
    """
    name = str(model_name).strip()
    lower = name.lower()

    # Check for p0, p1, p2 suffixes (e.g., "logistic_regression_p1_tuned")
    phase_match = re.search(r'_p(\d+)(?:_tuned)?$', lower)
    if phase_match:
        phase = f"p{phase_match.group(1)}"
        family = lower[:phase_match.start()].rstrip('_')
        return family, phase

    # Check for (Tuned) / (Baseline) markers
    if "(tuned)" in lower:
        family = lower.replace("(tuned)", "").strip().replace(" ", "_").replace("-", "_").strip("_")
        return family, "p1"
    if "(baseline)" in lower:
        family = lower.replace("(baseline)", "").strip().replace(" ", "_").replace("-", "_").strip("_")
        return family, "p0"

    # Check for " p0", " p1" suffixes (e.g., "SVM p0")
    phase_match = re.search(r'\s+p(\d+)$', lower)
    if phase_match:
        phase = f"p{phase_match.group(1)}"
        family = lower[:phase_match.start()].strip().replace(" ", "_").replace("-", "_")
        return family, phase

    # Default: assume p0 (initial comparison)
    family = lower.replace(" ", "_").replace("-", "_")
    return family, "p0"


def _goal_wants_recall(goal_text: str) -> bool:
    g = (goal_text or "").lower()
    return any(k in g for k in ("churn", "recall", "catch", "miss", "identify", "detect", "fraud", "attrition"))


def _load_comparison_rows(output_dir: str) -> List[Dict[str, Any]]:
    """Merged rows from model_comparison.csv + tuned_model_comparison.csv.

    Key is now f"{model_family}_{phase}" (e.g., "svm_p0", "svm_p1") to prevent
    p1 from overwriting p0. Both rows are preserved as separate entries.
    """
    import pandas as pd

    rows_by_key: Dict[str, Dict[str, Any]] = {}

    for fname in ("model_comparison.csv", "tuned_model_comparison.csv"):
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, row in df.iterrows():
            name = next((str(row[k]) for k in _MODEL_NAME_KEYS if k in row and pd.notna(row.get(k))), None)
            if not name:
                continue
            artifact = next((str(row[k]) for k in _ARTIFACT_KEYS if k in row and pd.notna(row.get(k))), None)
            metrics = {}
            for col in df.columns:
                if col in _MODEL_NAME_KEYS or col in _ARTIFACT_KEYS:
                    continue
                if col.lower() in ("configuration", "notes"):
                    continue
                try:
                    val = float(row[col])
                    metrics[_norm_metric_key(col)] = val
                except (TypeError, ValueError):
                    continue

            # Extract family and phase for unique key
            family, phase = _extract_phase_and_family(name)
            key = f"{family}_{phase}"

            rec = {
                "model": name,
                "model_family": family,
                "phase": phase,
                "artifact": artifact,
                "metrics": metrics,
                "source_file": fname,
            }
            # Use family_phase as key to preserve both p0 and p1
            rows_by_key[key] = rec

    return list(rows_by_key.values())


def _pick_metric(metrics: Dict[str, float], goal_text: str, primary_metric: Optional[str] = None) -> Optional[str]:
    """Pick the metric to use for ranking.

    If primary_metric is explicitly provided (from planner/task spec), use it.
    Otherwise fall back to goal-aware priority.
    """
    if primary_metric:
        pm = _norm_metric_key(primary_metric)
        if pm in metrics:
            return pm
        # Try common variations
        for k in metrics:
            if k == pm or k.replace("_", "") == pm.replace("_", ""):
                return k

    priority = RECALL_PRIORITY if _goal_wants_recall(goal_text) else DEFAULT_PRIORITY
    for cand in priority:
        if cand in metrics:
            return cand
    # regression fallback: minimize error metrics
    for cand in ("rmse", "mae", "mse"):
        if cand in metrics:
            return cand
    if metrics:
        return sorted(metrics.keys())[0]
    return None


def _validate_artifact_exists(output_dir: str, artifact: Optional[str]) -> bool:
    """Check if artifact file exists on disk."""
    if not artifact:
        return False
    artifact_path = Path(output_dir) / artifact
    return artifact_path.exists()


def _build_champion_record(
    winner: Dict[str, Any],
    metric: str,
    goal_text: str,
    output_dir: str,
    primary_metric: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Build champion record with validation."""
    artifact = winner.get("artifact")

    # Validation gate: artifact must exist
    if not _validate_artifact_exists(output_dir, artifact):
        return None

    ascending = metric in ASCENDING_METRICS
    basis = (
        f"highest {metric.upper()} among all evaluated models"
        if not ascending
        else f"lowest {metric.upper()} among all evaluated models"
    )
    if _goal_wants_recall(goal_text) and metric == "recall":
        basis = "recall-first selection rule derived from the user goal (catch as many positives as possible), tie-broken by F1 then ROC-AUC"

    champ = {
        "model": artifact or winner["model"],
        "model_name": winner["model"],
        "artifact_filename": artifact,
        "primary_metric": primary_metric or metric,
        "score": round(float(winner["metrics"][metric]), 6),
        "selection_basis": basis,
        "selected_by": "code (champion_selector)",
    }
    return champ


def _update_registry_with_champion(output_dir: str, champ: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Update model_registry.json to mark the champion."""
    reg_path = os.path.join(output_dir, "model_registry.json")
    try:
        # Load existing registry with schema validation
        registry = load_registry(reg_path)
    except ValueError as e:
        # Schema mismatch - raise hard error instead of silently defaulting to {}
        raise ValueError(f"model_registry.json schema validation failed: {e}. Fix the schema or delete the file to start fresh.")

    champ_art = champ.get("artifact_filename")

    def _entry_art(entry: ModelEntry) -> Optional[str]:
        return entry.artifact_filename

    matched = False
    for entry in registry.models:
        is_champ = bool(champ_art) and _entry_art(entry) == champ_art
        if is_champ:
            entry.status = ModelStatus.CHAMPION
            matched = True
        elif entry.status == ModelStatus.CHAMPION:
            # Revert former champion to evaluated (completed evaluation)
            entry.status = ModelStatus.EVALUATED

    # Register champion artifact if not already registered
    if not matched and champ_art:
        winner_row = next(
            (r for r in rows if (r.get("artifact") or r["model"]) == champ_art),
            None,
        )
        entry = ModelEntry(
            model_id=winner_row["model_family"] if winner_row else _norm_model_name(champ["model_name"]),
            model_family=winner_row["model"] if winner_row else champ["model_name"],
            artifact_filename=champ_art,
            version=winner_row.get("phase", "p0") if winner_row else "p0",
            stage=ModelStage.INITIAL_COMPARISON if (winner_row and winner_row.get("source_file") == "model_comparison.csv") else ModelStage.HYPERPARAMETER_TUNING,
            task_id=0,  # Will be filled by orchestrator
            dataset_source="",
            target_column="",
            feature_count=0,
            evaluation_metrics=winner_row["metrics"] if winner_row else {},
            primary_selection_metric=champ["primary_metric"],
            secondary_metrics=[],
            status=ModelStatus.CHAMPION,
        )
        registry.models.append(entry)
        champ["registry_entry_added"] = True

    save_registry(registry, reg_path)


def select_initial_champion(
    output_dir: str,
    goal_text: str = "",
    primary_metric: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Deterministically choose the best model from the run's comparison tables,
    rewrite champion_model.json, and mark the champion in model_registry.json.

    Called ONCE after the initial model comparison task completes.

    Returns the champion record, or None when no comparison data exists.
    """
    rows = _load_comparison_rows(output_dir)
    if not rows:
        return None

    candidates = [r for r in rows if r["metrics"]]
    if not candidates:
        return None

    metric = _pick_metric(candidates[0]["metrics"], goal_text, primary_metric)
    if metric is None:
        return None
    ascending = metric in ASCENDING_METRICS
    ranked = sorted(
        candidates,
        key=lambda r: r["metrics"].get(metric, float("inf") if ascending else float("-inf")),
        reverse=not ascending,
    )
    winner = ranked[0]

    champ = _build_champion_record(winner, metric, goal_text, output_dir, primary_metric)
    if champ is None:
        # No valid artifact - return error info
        return {
            "error": "no_valid_artifact",
            "message": f"Champion candidate {winner['model']} has no valid artifact file on disk",
            "candidate": winner["model"]
        }

    champ_path = os.path.join(output_dir, "champion_model.json")
    try:
        # Validate with schema before saving
        champion_model = ChampionModel(**champ)
        save_champion(champion_model, champ_path)
    except Exception as e:
        print(f"[ChampionSelector] could not write champion_model.json: {e}")
        return None

    # Mark the champion in the model registry
    _update_registry_with_champion(output_dir, champ, rows)

    # Report ranking (also consumed for the near-tie warning)
    champ["ranking"] = [
        {"model": r["model"], "artifact": r.get("artifact"), metric: round(r["metrics"].get(metric, float("nan")), 6)}
        for r in ranked[:5]
    ]
    champ["primary_metric_name"] = metric
    champ["_winner_metrics"] = winner["metrics"]

    if logfire:
        logfire.info(
            "Initial champion selected by code",
            model=champ["model_name"],
            artifact=champ.get("artifact_filename"),
            metric=metric,
            score=champ["score"],
        )
    return champ


def evaluate_challenger(
    output_dir: str,
    goal_text: str = "",
    primary_metric: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Compare current champion vs. the one tuned model (challenger).

    Called AFTER each HPT task. Reads current champion from champion_model.json,
    finds the challenger from tuned_model_comparison.csv, and returns whichever wins.

    Returns the champion record (either retained or updated), or None on error.
    """
    # Load current champion
    champ_path = os.path.join(output_dir, "champion_model.json")
    current_champ = load_champion(champ_path)
    if current_champ is None:
        # No existing champion - fall back to initial selection
        return select_initial_champion(output_dir, goal_text, primary_metric)

    # Load comparison rows to find the challenger (p1/p2 from tuned_model_comparison.csv)
    rows = _load_comparison_rows(output_dir)
    if not rows:
        return current_champ.dict()

    # Find challenger rows (phase p1, p2, etc. from tuned_model_comparison.csv)
    challengers = [r for r in rows if r.get("phase") != "p0" and r.get("source_file") == "tuned_model_comparison.csv"]
    if not challengers:
        return current_champ.dict()

    # For now, assume single challenger per HPT run (the one just tuned)
    # In future, could handle multiple by picking the best among them
    challenger = challengers[0]

    # Get metric from primary_metric or current champion
    metric = primary_metric or current_champ.primary_metric
    if metric not in challenger["metrics"] or metric not in current_champ.dict().get("_winner_metrics", {}):
        # Fall back to goal-aware pick
        metric = _pick_metric(challenger["metrics"], goal_text, primary_metric)
        if metric is None:
            return current_champ.dict()

    ascending = metric in ASCENDING_METRICS
    current_score = current_champ.score
    challenger_score = challenger["metrics"].get(metric, float("inf") if ascending else float("-inf"))

    # Determine winner
    if (not ascending and challenger_score > current_score) or (ascending and challenger_score < current_score):
        # Challenger wins
        winner = challenger
        basis = f"challenger {winner['model']} improved {metric.upper()} from {current_score:.6f} to {challenger_score:.6f}"
    else:
        # Current champion retained
        winner = next(r for r in rows if r.get("artifact") == current_champ.artifact_filename or r["model"] == current_champ.model_name)
        basis = f"current champion {current_champ.model_name} retained ({metric.upper()}={current_score:.6f} vs challenger {challenger_score:.6f})"

    champ = _build_champion_record(winner, metric, goal_text, output_dir, primary_metric)
    if champ is None:
        # Challenger has no valid artifact - retain current
        return current_champ.dict()

    # Update champion_model.json
    try:
        champion_model = ChampionModel(**champ)
        save_champion(champion_model, champ_path)
    except Exception as e:
        print(f"[ChampionSelector] could not write champion_model.json: {e}")
        return None

    # Update registry
    _update_registry_with_champion(output_dir, champ, rows)

    # Ranking for reporting
    all_candidates = [r for r in rows if r["metrics"]]
    ranked = sorted(
        all_candidates,
        key=lambda r: r["metrics"].get(metric, float("inf") if ascending else float("-inf")),
        reverse=not ascending,
    )
    champ["ranking"] = [
        {"model": r["model"], "artifact": r.get("artifact"), metric: round(r["metrics"].get(metric, float("nan")), 6)}
        for r in ranked[:5]
    ]
    champ["primary_metric_name"] = metric
    champ["_winner_metrics"] = winner["metrics"]

    if logfire:
        logfire.info(
            "Challenger evaluated",
            model=champ["model_name"],
            artifact=champ.get("artifact_filename"),
            metric=metric,
            score=champ["score"],
            changed=champ["model_name"] != current_champ.model_name
        )
    return champ


# Backward compatibility
def select_champion(output_dir: str, goal_text: str = "") -> Optional[Dict[str, Any]]:
    """Legacy function - calls select_initial_champion for backward compatibility."""
    return select_initial_champion(output_dir, goal_text)