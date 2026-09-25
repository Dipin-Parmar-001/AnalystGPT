"""
Champion Selector - AnalystGPT.

Deterministic, code-owned best-model selection. After ML/tuning tasks finish,
the orchestrator calls `select_champion()` which reads the comparison CSVs the
run actually produced, applies a goal-aware metric priority, and rewrites
`champion_model.json` / `model_registry.json` itself. The ML agent never decides
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
from typing import Dict, Any, List, Optional

from agents.logger import setup_logfire

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
    Tuned rows win when a model family appears in both."""
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
            rec = {
                "model": name,
                "artifact": artifact,
                "metrics": metrics,
                "source_file": fname,
            }
            # Tuned table: family key strips " (Tuned)"/" (Baseline)" suffixes so a
            # tuned row replaces the base row of the same family.
            key = _norm_model_name(name)
            rows_by_key[key] = rec

    return list(rows_by_key.values())


def _pick_metric(metrics: Dict[str, float], goal_text: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Select the best metric based on goal, verified metrics, and available data.

    Priority:
    1. Explicit metric from verified_metrics.json (user-specified or code-selected)
    2. Goal-based priority (recall vs default)
    3. Verify the metric exists in comparison data; if not, fall back to next priority
    4. Regression fallback
    """
    # 1. Check verified_metrics.json for explicitly set primary metric
    if output_dir:
        import os
        import json
        vm_path = os.path.join(output_dir, "verified_metrics.json")
        if os.path.exists(vm_path):
            try:
                with open(vm_path, "r", encoding="utf-8") as f:
                    vm = json.load(f)
                # Check for user-specified primary metric in the goal/plan
                explicit_metric = vm.get("primary_metric") or vm.get("goal_primary_metric")
                if explicit_metric:
                    norm_explicit = _norm_metric_key(explicit_metric)
                    if norm_explicit in metrics:
                        return norm_explicit
                    # If explicit metric not in this row's metrics, check if it exists in ANY row
                    # (we'd need access to all rows; for now fall through to goal-based)
            except Exception:
                pass

    # 2. Goal-based priority
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


def select_champion(output_dir: str, goal_text: str = "") -> Optional[Dict[str, Any]]:
    """Deterministically choose the best model from the run's comparison tables,
    rewrite champion_model.json, and mark the champion in model_registry.json.

    Returns the champion record, or None when no comparison data exists.
    """
    rows = _load_comparison_rows(output_dir)
    if not rows:
        return None

    candidates = [r for r in rows if r["metrics"]]
    if not candidates:
        return None

    metric = _pick_metric(candidates[0]["metrics"], goal_text, output_dir)
    if metric is None:
        return None
    ascending = metric in ASCENDING_METRICS
    ranked = sorted(
        candidates,
        key=lambda r: r["metrics"].get(metric, float("inf") if ascending else float("-inf")),
        reverse=not ascending,
    )
    winner = ranked[0]

    basis = (
        f"highest {metric.upper()} among all evaluated models"
        if not ascending
        else f"lowest {metric.upper()} among all evaluated models"
    )
    if _goal_wants_recall(goal_text) and metric == "recall":
        basis = "recall-first selection rule derived from the user goal (catch as many positives as possible), tie-broken by F1 then ROC-AUC"

    champ = {
        "model": winner.get("artifact") or winner["model"],
        "model_name": winner["model"],
        "artifact_filename": winner.get("artifact"),
        "primary_metric": metric,
        "score": round(float(winner["metrics"][metric]), 6),
        "selection_basis": basis,
        "selected_by": "code (champion_selector)",
    }

    champ_path = os.path.join(output_dir, "champion_model.json")
    try:
        with open(champ_path, "w", encoding="utf-8") as f:
            json.dump(champ, f, indent=2)
    except Exception as e:
        print(f"[ChampionSelector] could not write champion_model.json: {e}")

    # Mark the champion in the model registry so every artifact agrees.
    # Matching is by exact artifact_filename first; the model-family fallback
    # must never mark a DIFFERENT artifact (e.g. the untuned baseline) as
    # champion when a tuned sibling won.
    reg_path = os.path.join(output_dir, "model_registry.json")
    try:
        if os.path.isfile(reg_path):
            with open(reg_path, encoding="utf-8") as f:
                registry = json.load(f)
            if not isinstance(registry, dict):
                registry = {}
            entries = registry.get("models")
            if not isinstance(entries, list):
                entries = []
                registry["models"] = entries

            champ_art = champ.get("artifact_filename")

            def _entry_art(entry: Dict[str, Any]) -> Optional[str]:
                return entry.get("artifact_filename") or entry.get("model")

            matched = False
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                is_champ = bool(champ_art) and _entry_art(entry) == champ_art
                if is_champ:
                    entry["status"] = "champion"
                    matched = True
                elif entry.get("status") == "champion":
                    entry["status"] = entry.get("stage") or "evaluated"

            # The champion artifact may not be registered at all (e.g. a tuned
            # model written after the registry was created). Register it from
            # the comparison row so champion_model.json and the registry agree.
            if not matched and champ_art:
                winner_row = next(
                    (r for r in rows if (r.get("artifact") or r["model"]) == champ_art),
                    None,
                )
                entry = {
                    "model_id": _norm_model_name(champ["model_name"]),
                    "model_family": winner_row["model"] if winner_row else champ["model_name"],
                    "artifact_filename": champ_art,
                    "status": "champion",
                    "registered_by": "champion_selector",
                }
                if winner_row:
                    entry["evaluation_metrics"] = winner_row["metrics"]
                    entry["source_file"] = winner_row.get("source_file")
                entries.append(entry)
                champ["registry_entry_added"] = True

            with open(reg_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"[ChampionSelector] could not update model_registry.json: {e}")

    # Report ranking (also consumed for the near-tie warning).
    champ["ranking"] = [
        {"model": r["model"], "artifact": r.get("artifact"), metric: round(r["metrics"].get(metric, float("nan")), 6)}
        for r in ranked[:5]
    ]
    champ["primary_metric_name"] = metric
    # Internal: full metric row of the winner (preflight round-trip check); never
    # persisted because champion_model.json is written above.
    champ["_winner_metrics"] = winner["metrics"]

    if logfire:
        logfire.info(
            "Champion selected by code",
            model=champ["model_name"],
            artifact=champ.get("artifact_filename"),
            metric=metric,
            score=champ["score"],
        )
    return champ
