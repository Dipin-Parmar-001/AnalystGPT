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
from dataclasses import dataclass
from enum import Enum

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


class ChampionSelectionStatus(Enum):
    """Explicit status for champion selection results."""
    SUCCESS = "success"
    NO_COMPARISON_DATA = "no_comparison_data"
    NO_VALID_CANDIDATES = "no_valid_candidates"
    NO_VALID_ARTIFACT = "no_valid_artifact"
    NO_METRIC_FOUND = "no_metric_found"
    SAVE_FAILED = "save_failed"
    REGISTRY_SYNC_FAILED = "registry_sync_failed"
    EXCEPTION = "exception"


@dataclass
class ChampionSelectionResult:
    """Structured result for champion selection to avoid None ambiguity."""
    status: ChampionSelectionStatus
    champion: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stage: str = "selection"
    candidate: Optional[str] = None
    ranking: Optional[List[Dict[str, Any]]] = None
    primary_metric_name: Optional[str] = None
    _winner_metrics: Optional[Dict[str, float]] = None

    def is_success(self) -> bool:
        return self.status == ChampionSelectionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status": self.status.value,
            "stage": self.stage,
        }
        if self.champion:
            result["champion"] = self.champion
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message
        if self.candidate:
            result["candidate"] = self.candidate
        if self.ranking:
            result["ranking"] = self.ranking
        if self.primary_metric_name:
            result["primary_metric_name"] = self.primary_metric_name
        if self._winner_metrics:
            result["_winner_metrics"] = self._winner_metrics
        return result


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
    """Determine if the user's goal explicitly prioritizes recall over ROC-AUC.

    Checks for:
    1. Explicit metric specification: "primary metric: recall", "optimize for f1", "metric=roc_auc"
    2. Business context keywords: churn, fraud, "catch as many", "don't miss"
    """
    g = (goal_text or "").lower()

    # First: Check for explicit primary metric specification (highest priority)
    # Patterns like: "primary metric: recall", "optimize for f1", "metric=roc_auc", "target: precision"
    import re
    explicit_metric_patterns = [
        r"primary\s+metric\s*[:\-]\s*(recall|f1|roc[_\-]?auc|pr[_\-]?auc|precision|accuracy)",
        r"optimize\s+for\s+(recall|f1|roc[_\-]?auc|pr[_\-]?auc|precision|accuracy)",
        r"metric\s*[=\:]\s*(recall|f1|roc[_\-]?auc|pr[_\-]?auc|precision|accuracy)",
        r"target\s*[:\-]\s*(recall|f1|roc[_\-]?auc|pr[_\-]?auc|precision|accuracy)",
        r"maximize\s+(recall|f1|roc[_\-]?auc|pr[_\-]?auc|precision|accuracy)",
    ]
    for pattern in explicit_metric_patterns:
        match = re.search(pattern, g)
        if match:
            metric = match.group(1).replace("-", "_")
            if metric in ("recall", "f1"):
                return True
            elif metric in ("roc_auc", "pr_auc", "precision", "accuracy"):
                return False

    # Second: Business context keywords that imply recall priority
    recall_keywords = (
        "churn", "fraud", "attrition",
        "catch as many", "don't miss", "do not miss",
        "identify all", "detect all", "miss no",
        "minimize false negative", "reduce false negative",
        "recall", "sensitivity", "true positive rate"
    )
    return any(k in g for k in recall_keywords)


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

            # Attempt artifact resolution if artifact not provided or not directly found on disk
            if not _validate_artifact_exists(output_dir, artifact):
                resolved = _resolve_model_artifact(output_dir, artifact, name, phase)
                if resolved:
                    artifact = resolved

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


def _clean_str(s: Any) -> str:
    """Lowercase string with only alphanumeric characters, stripping extension if any."""
    if not s:
        return ""
    stem = Path(str(s)).stem
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _resolve_model_artifact(
    output_dir: str,
    candidate_artifact: Optional[str] = None,
    model_name: Optional[str] = None,
    phase: str = "p0",
) -> Optional[str]:
    """Resolve the actual on-disk .pkl artifact for a model candidate.

    Handles naming variations:
    - underscores vs no underscores (logistic_regression_p0.pkl vs logisticregression_p0.pkl)
    - case differences (LogisticRegression.pkl vs logisticregression.pkl)
    - missing phase or extension (LogisticRegression -> logisticregression_p0.pkl)
    - missing artifact column in comparison CSV (resolves from model_name)
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None

    # 1. Exact match on disk
    if candidate_artifact:
        cand_str = str(candidate_artifact).strip()
        if cand_str and os.path.isfile(os.path.join(output_dir, cand_str)):
            return cand_str

    pkl_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".pkl")]
    if not pkl_files:
        return None

    clean_cand = _clean_str(candidate_artifact) if candidate_artifact else ""
    clean_model = _clean_str(model_name) if model_name else ""
    norm_phase = (phase or "p0").lower().strip()

    # 2. Try matching candidate_artifact by clean stem (e.g. logisticregressionp0)
    if clean_cand:
        for f in pkl_files:
            if _clean_str(f) == clean_cand:
                return f

    # 3. Try matching via normalize_model_family from preflight_gate
    try:
        from agents.preflight_gate import normalize_model_family
        target_family = None
        if candidate_artifact:
            target_family = normalize_model_family(candidate_artifact)
        if not target_family or target_family == "unknown":
            if model_name:
                target_family = normalize_model_family(model_name)

        if target_family:
            matching = [f for f in pkl_files if normalize_model_family(f) == target_family]
            if len(matching) == 1:
                return matching[0]
            elif len(matching) > 1:
                # Disambiguate by phase (e.g. p0, p1)
                phase_matching = [f for f in matching if norm_phase in f.lower()]
                if phase_matching:
                    return phase_matching[0]
                return matching[0]
    except Exception:
        pass

    # 4. Try matching model_name + phase by clean stem
    if clean_model:
        target_clean = f"{clean_model}{norm_phase}"
        for f in pkl_files:
            clean_f = _clean_str(f)
            if clean_f == target_clean:
                return f
        for f in pkl_files:
            clean_f = _clean_str(f)
            if clean_f.startswith(clean_model) and norm_phase in clean_f:
                return f
        for f in pkl_files:
            clean_f = _clean_str(f)
            if clean_f.startswith(clean_model):
                return f

    return None


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

    # 3. Verify metric exists in comparison data, fall back if not
    for cand in priority:
        if cand in metrics:
            return cand

    # 4. Regression fallback: minimize error metrics
    for cand in ("rmse", "mae", "mse"):
        if cand in metrics:
            return cand

    # 5. Last resort: first available metric
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

    # If artifact is missing or not directly on disk, attempt resolution
    if not _validate_artifact_exists(output_dir, artifact):
        resolved = _resolve_model_artifact(
            output_dir,
            candidate_artifact=artifact,
            model_name=winner.get("model"),
            phase=winner.get("phase", "p0"),
        )
        if resolved:
            artifact = resolved
            winner["artifact"] = resolved

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
        entry_art = _entry_art(entry)
        is_champ = bool(champ_art) and (
            entry_art == champ_art
            or (entry_art and _clean_str(entry_art) == _clean_str(champ_art))
        ) and not matched
        if is_champ:
            entry.artifact_filename = champ_art
            entry.status = ModelStatus.CHAMPION
            matched = True
        elif entry.status == ModelStatus.CHAMPION:
            entry.status = ModelStatus.EVALUATED

    # Register champion artifact if not already registered
    if not matched and champ_art:
        winner_row = next(
            (r for r in rows if (r.get("artifact") or r["model"]) == champ_art or _clean_str(r.get("artifact")) == _clean_str(champ_art)),
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
) -> ChampionSelectionResult:
    """Deterministically choose the best model from the run's comparison tables,
    rewrite champion_model.json, and mark the champion in model_registry.json.

    Called ONCE after the initial model comparison task completes.

    Returns a ChampionSelectionResult with status, champion data, or error info.
    """
    try:
        rows = _load_comparison_rows(output_dir)
        if not rows:
            return ChampionSelectionResult(ChampionSelectionStatus.NO_COMPARISON_DATA, error_code="NO_COMPARISON_DATA", error_message="No model comparison CSV files found", stage="load_data")
        candidates = [row for row in rows if row["metrics"]]
        if not candidates:
            return ChampionSelectionResult(ChampionSelectionStatus.NO_VALID_CANDIDATES, error_code="NO_VALID_CANDIDATES", error_message="No candidates with metrics found")
        from agents.preflight_gate import is_dummy_classifier
        viable_candidates = [
            row for row in candidates
            if not is_dummy_classifier(row.get("model")) and not is_dummy_classifier(row.get("artifact"))
        ]
        if viable_candidates:
            candidates = viable_candidates
        metric = _pick_metric(candidates[0]["metrics"], goal_text, primary_metric)
        if metric is None:
            return ChampionSelectionResult(ChampionSelectionStatus.NO_METRIC_FOUND, error_code="NO_METRIC_FOUND", error_message="No suitable metric found")
        ascending = metric in ASCENDING_METRICS
        ranked = sorted(candidates, key=lambda row: row["metrics"].get(metric, float("inf") if ascending else float("-inf")), reverse=not ascending)
        winner = ranked[0]
        champ = _build_champion_record(winner, metric, goal_text, output_dir, primary_metric)
        if champ is None:
            return ChampionSelectionResult(ChampionSelectionStatus.NO_VALID_ARTIFACT, error_code="NO_VALID_ARTIFACT", error_message=f"Champion candidate {winner['model']} has no valid artifact", candidate=winner["model"])
        champ_path = os.path.join(output_dir, "champion_model.json")
        try:
            save_champion(ChampionModel(**champ), champ_path)
            verified = load_champion(champ_path)
            if verified is None or verified.artifact_filename != champ.get("artifact_filename"):
                raise ValueError("written champion does not match selected artifact")
        except Exception as exc:
            return ChampionSelectionResult(ChampionSelectionStatus.SAVE_FAILED, stage="verify_write", error_code="ARTIFACT_VERIFY_FAILED", error_message=str(exc), candidate=winner["model"])
        registry_error = None
        try:
            _update_registry_with_champion(output_dir, champ, rows)
        except Exception as exc:
            registry_error = exc
        champ["ranking"] = [{"model": row["model"], "artifact": row.get("artifact"), metric: round(row["metrics"].get(metric, float("nan")), 6)} for row in ranked[:5]]
        champ["primary_metric_name"] = metric
        champ["_winner_metrics"] = winner["metrics"]
        result = ChampionSelectionResult(ChampionSelectionStatus.SUCCESS, champ, stage="registry_sync" if registry_error else "completed", ranking=champ["ranking"], primary_metric_name=metric, _winner_metrics=winner["metrics"])
        if registry_error:
            result.error_code = "REGISTRY_SYNC_FAILED"
            result.error_message = f"Champion selected and saved, but registry sync failed: {registry_error}"
        return result
    except Exception as e:
        print(f"[ChampionSelector] unhandled exception during select_initial_champion: {e}")
        return ChampionSelectionResult(
            status=ChampionSelectionStatus.EXCEPTION,
            stage="initial_selection",
            error_code="SELECTION_EXCEPTION",
            error_message=f"Exception during champion selection: {e}"
        )


def evaluate_challenger(output_dir: str, goal_text: str = "", primary_metric: Optional[str] = None) -> ChampionSelectionResult:
    """Evaluate the latest tuned challenger against the persisted champion."""
    try:
        champ_path = os.path.join(output_dir, "champion_model.json")
        current = load_champion(champ_path)
        if current is None:
            return select_initial_champion(output_dir, goal_text, primary_metric)
        rows = _load_comparison_rows(output_dir)
        challengers = [r for r in rows if r.get("phase") != "p0" and r.get("source_file") == "tuned_model_comparison.csv"]
        if not challengers:
            return ChampionSelectionResult(ChampionSelectionStatus.SUCCESS, champion=current.dict(), primary_metric_name=current.primary_metric)
        challenger = max(challengers, key=lambda r: int(r.get("phase", "p0")[1:]) if r.get("phase", "p0")[1:].isdigit() else 0)
        metric = primary_metric or current.primary_metric
        if metric not in challenger["metrics"]:
            metric = _pick_metric(challenger["metrics"], goal_text, primary_metric)
        if metric is None:
            return ChampionSelectionResult(ChampionSelectionStatus.NO_METRIC_FOUND, champion=current.dict(), error_code="NO_METRIC_FOUND", error_message="No comparable metric found", stage="challenger")
        ascending = metric in ASCENDING_METRICS
        current_score = current.score
        challenger_score = challenger["metrics"].get(metric, float("inf") if ascending else float("-inf"))
        challenger_wins = challenger_score < current_score if ascending else challenger_score > current_score
        if challenger_wins:
            winner = challenger
        else:
            winner = next((
                r for r in rows
                if r.get("artifact") == current.artifact_filename
                or r["model"] == current.model_name
                or (r.get("artifact") and _clean_str(r.get("artifact")) == _clean_str(current.artifact_filename))
            ), None)
            if winner is None:
                return ChampionSelectionResult(ChampionSelectionStatus.NO_VALID_CANDIDATES, champion=current.dict(), error_code="CURRENT_CHAMPION_NOT_IN_COMPARISON", error_message="Current champion has no comparison row", stage="challenger")
        champ = _build_champion_record(winner, metric, goal_text, output_dir, primary_metric)
        if champ is None:
            return ChampionSelectionResult(ChampionSelectionStatus.SUCCESS, champion=current.dict(), primary_metric_name=current.primary_metric)
        try:
            save_champion(ChampionModel(**champ), champ_path)
            saved = load_champion(champ_path)
            if saved is None or saved.artifact_filename != champ.get("artifact_filename"):
                raise ValueError("written champion does not match selected artifact")
        except Exception as exc:
            return ChampionSelectionResult(ChampionSelectionStatus.SAVE_FAILED, stage="verify_write", error_code="ARTIFACT_VERIFY_FAILED", error_message=str(exc))
        registry_error = None
        try:
            _update_registry_with_champion(output_dir, champ, rows)
        except Exception as exc:
            registry_error = exc
        ranked = sorted([r for r in rows if r["metrics"]], key=lambda r: r["metrics"].get(metric, float("inf") if ascending else float("-inf")), reverse=not ascending)
        champ["ranking"] = [{"model": r["model"], "artifact": r.get("artifact"), metric: round(r["metrics"].get(metric, float("nan")), 6)} for r in ranked[:5]]
        champ["primary_metric_name"] = metric
        champ["_winner_metrics"] = winner["metrics"]
        result = ChampionSelectionResult(ChampionSelectionStatus.SUCCESS, champion=champ, stage="registry_sync" if registry_error else "completed", ranking=champ["ranking"], primary_metric_name=metric, _winner_metrics=winner["metrics"])
        if registry_error:
            result.error_code = "REGISTRY_SYNC_FAILED"
            result.error_message = f"Champion selected and saved, but registry sync failed: {registry_error}"
        return result
    except Exception as exc:
        return ChampionSelectionResult(ChampionSelectionStatus.EXCEPTION, stage="evaluate_challenger", error_code="SELECTION_EXCEPTION", error_message=f"Exception during challenger evaluation: {exc}")

# Backward compatibility
def select_champion(output_dir: str, goal_text: str = "") -> Optional[Dict[str, Any]]:
    """Legacy function - calls select_initial_champion for backward compatibility.

    Returns the champion dict if successful, None otherwise.
    For new code, use select_initial_champion which returns ChampionSelectionResult.
    """
    result = select_initial_champion(output_dir, goal_text)
    if result.is_success() and result.champion:
        return result.champion
    return None
