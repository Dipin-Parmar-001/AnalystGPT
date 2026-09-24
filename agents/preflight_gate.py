"""
Preflight Gate - AnalystGPT.

Runs AFTER training and BEFORE the reporter starts. Asks "do all the facts
agree?" across the run's artifacts:

1. Exactly one best model everywhere it is mentioned (champion_model.json,
   model_registry.json, ml_summary.json, comparison tables).
2. The confusion matrix round-trips to the champion's reported recall/precision.
3. File lists referenced in summaries exist in the run folder.
4. Row/column counts in dataset_profile.json match the actual CSVs on disk.

Failure policy (user-confirmed): NEVER stop the workflow. First attempt an
automatic code repair (reselect the champion, recompute the confusion matrix
from predictions.csv, rewrite profile counts and file lists), re-run the
checks once, and whatever still fails is downgraded to a WARNING that is
passed to the reporter and must appear in the report.

Also emits machine-computed run-level warnings: small test set, class
imbalance, two nearly tied models.

The whole gate is fail-safe: any internal error is swallowed and reported as
a warning so it can never block report generation.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from agents.logger import setup_logfire
from agents.champion_selector import select_champion, _norm_model_name

try:
    import logfire
except ImportError:
    logfire = None

setup_logfire()

TOL = 0.02          # tolerance for confusion-matrix round-trip vs reported metrics
NEAR_TIE_GAP = 0.01  # top-2 models closer than this on the primary metric
SMALL_TEST_MAX = 300  # test set smaller than this triggers a warning


def _load_json_safe(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def _dump_json(path: str, data: Any) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[PreflightGate] could not write {path}: {e}")
        return False


def _finding(code: str, message: str, artifacts: List[str]) -> Dict[str, Any]:
    return {"severity": "warning", "code": code, "message": message, "artifacts": artifacts}


# ----------------------------------------------------------------------------
# Checks (each returns a list of findings)
# ----------------------------------------------------------------------------

def _check_champion(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    pkl_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".pkl")) if os.path.isdir(output_dir) else []

    if champ is None:
        if os.path.exists(os.path.join(output_dir, "model_comparison.csv")):
            findings.append(_finding("champion_not_selected", "A model comparison exists but no champion could be selected by code.", ["model_comparison.csv"]))
        return findings

    art = champ.get("artifact_filename") or champ.get("model")
    if art and art not in pkl_files:
        findings.append(_finding(
            "stale_champion_pointer",
            f"champion_model.json points to '{art}', which does not exist on disk. Available: {', '.join(pkl_files) or 'none'}.",
            ["champion_model.json"],
        ))

    # ml_summary.json agreement
    ml = _load_json_safe(os.path.join(output_dir, "ml_summary.json"))
    if isinstance(ml, dict):
        summary_best = (ml.get("best_model") or {}).get("name") if isinstance(ml.get("best_model"), dict) else ml.get("best_model")
        if summary_best and _norm_model_name(str(summary_best)) != _norm_model_name(champ.get("model_name") or ""):
            findings.append(_finding(
                "champion_disagreement_ml_summary",
                f"ml_summary.json names '{summary_best}' as best model but code-selected champion is '{champ.get('model_name')}'.",
                ["ml_summary.json", "champion_model.json"],
            ))
    return findings


def _check_confusion_matrix(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    cm = _load_json_safe(os.path.join(output_dir, "confusion_matrix.json"))
    if not isinstance(cm, dict):
        return findings
    tp = cm.get("true_positives")
    fp = cm.get("false_positives")
    fn = cm.get("false_negatives")
    if not all(isinstance(v, (int, float)) for v in (tp, fp, fn)):
        return findings
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    if recall is None and precision is None:
        return findings

    # Reported metrics for the champion come from the selector's ranking; if the
    # selector didn't run, fall back to the champion file's own record.
    reported: Dict[str, float] = {}
    if champ and isinstance(champ.get("_winner_metrics"), dict):
        reported = champ["_winner_metrics"]
    else:
        champ_file = _load_json_safe(os.path.join(output_dir, "champion_model.json")) or {}
        for k, v in champ_file.items():
            if isinstance(v, (int, float)):
                reported[k.lower()] = float(v)

    for name, derived in (("recall", recall), ("precision", precision)):
        rep = reported.get(name)
        if rep is not None and derived is not None and abs(rep - derived) > TOL:
            findings.append(_finding(
                "confusion_matrix_mismatch",
                f"confusion_matrix.json implies {name}={derived:.4f} but the champion's reported {name} is {rep:.4f} — the matrix may belong to a different model.",
                ["confusion_matrix.json", "champion_model.json"],
            ))
    return findings


def _check_file_lists(output_dir: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not os.path.isdir(output_dir):
        return findings
    on_disk = set(os.listdir(output_dir))

    ml = _load_json_safe(os.path.join(output_dir, "ml_summary.json"))
    if isinstance(ml, dict):
        for gf in (ml.get("generated_artifacts") or []):
            fname = gf.get("file") if isinstance(gf, dict) else gf
            if fname and fname not in on_disk:
                findings.append(_finding("phantom_file_reference", f"ml_summary.json lists artifact '{fname}' which does not exist in the run folder.", ["ml_summary.json"]))
    return findings


def _check_shape_counts(output_dir: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    profile = _load_json_safe(os.path.join(output_dir, "dataset_profile.json"))
    if not isinstance(profile, dict):
        return findings
    import pandas as pd

    datasets = profile.get("datasets") or {}
    for key, meta in datasets.items():
        if not isinstance(meta, dict):
            continue
        fname = meta.get("file")
        if not fname or not os.path.isfile(os.path.join(output_dir, fname)):
            continue
        try:
            actual_rows, actual_cols = pd.read_csv(os.path.join(output_dir, fname)).shape
        except Exception:
            continue
        if meta.get("rows") not in (None, actual_rows) or meta.get("cols") not in (None, actual_cols):
            findings.append(_finding(
                "shape_mismatch",
                f"dataset_profile.json says dataset '{key}' ({fname}) is {meta.get('rows')}x{meta.get('cols')} but the file on disk is {actual_rows}x{actual_cols}.",
                ["dataset_profile.json", fname],
            ))
    return findings


# ----------------------------------------------------------------------------
# Repairs
# ----------------------------------------------------------------------------

def _repair_confusion_matrix(output_dir: str) -> bool:
    """Recompute confusion_matrix.json from predictions.csv (actual vs predicted)."""
    import pandas as pd

    pred_path = os.path.join(output_dir, "predictions.csv")
    if not os.path.isfile(pred_path):
        return False
    try:
        df = pd.read_csv(pred_path)
    except Exception:
        return False
    cols = {c.strip().lower(): c for c in df.columns}
    actual_col = next((cols[c] for c in ("actual", "y_true", "y_actual", "true") if c in cols), None)
    pred_col = next((cols[c] for c in ("predicted", "y_pred", "prediction") if c in cols), None)
    if not actual_col or not pred_col:
        return False
    try:
        y = df[actual_col].astype(int)
        p = df[pred_col].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
    except Exception:
        return False
    return _dump_json(os.path.join(output_dir, "confusion_matrix.json"), {
        "matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
        "labels": ["negative", "positive"],
        "recomputed_by": "preflight_gate from predictions.csv",
    })


def _repair_dataset_profile(output_dir: str) -> bool:
    """Rewrite dataset_profile.json shape counts from the files on disk."""
    profile = _load_json_safe(os.path.join(output_dir, "dataset_profile.json"))
    if not isinstance(profile, dict):
        return False
    import pandas as pd

    changed = False
    datasets = profile.get("datasets") or {}
    for key, meta in datasets.items():
        if not isinstance(meta, dict) or not meta.get("file"):
            continue
        fpath = os.path.join(output_dir, meta["file"])
        if not os.path.isfile(fpath):
            continue
        try:
            shape = pd.read_csv(fpath).shape
        except Exception:
            continue
        if meta.get("rows") != shape[0] or meta.get("cols") != shape[1]:
            meta["rows"], meta["cols"] = int(shape[0]), int(shape[1])
            meta["shape_recomputed_by"] = "preflight_gate"
            changed = True
    if changed:
        return _dump_json(os.path.join(output_dir, "dataset_profile.json"), profile)
    return changed


def _repair_ml_summary_champion(output_dir: str, champ: Dict[str, Any]) -> bool:
    """Point ml_summary.json's best-model fields at the code-selected champion."""
    path = os.path.join(output_dir, "ml_summary.json")
    ml = _load_json_safe(path)
    if not isinstance(ml, dict):
        return False
    name = champ.get("model_name") or champ.get("model")
    changed = False
    if isinstance(ml.get("best_model"), dict):
        if ml["best_model"].get("name") != name:
            ml["best_model"]["name"] = name
            changed = True
    elif ml.get("best_model") and ml["best_model"] != name:
        ml["best_model"] = name
        changed = True
    if isinstance(ml.get("model_selection"), dict) and ml["model_selection"].get("best_model") != name:
        ml["model_selection"]["best_model"] = name
        changed = True
    if changed:
        return _dump_json(path, ml)
    return changed


# ----------------------------------------------------------------------------
# Post-task champion consistency (runs after EVERY ML task, not just pre-report)
# ----------------------------------------------------------------------------

def _repair_ml_champion_pointer(path: str, champ: Dict[str, Any]) -> List[str]:
    """Rewrite one ml_summary JSON's champion pointer to the code-selected champion.

    Fixes the `champion` block, `best_model` / `model_selection.best_model`, and
    the `model_artifacts[].status` flags. Returns a list of human-readable repairs.
    """
    ml = _load_json_safe(path)
    if not isinstance(ml, dict):
        return []
    repairs: List[str] = []
    fname = os.path.basename(path)
    champ_art = champ.get("artifact_filename") or champ.get("model")
    champ_name = champ.get("model_name") or champ.get("model")
    metric = champ.get("primary_metric") or champ.get("primary_metric_name")

    ch = ml.get("champion")
    if isinstance(ch, dict):
        if ch.get("model_artifact") not in (None, champ_art):
            ch["model_artifact"] = champ_art
            ch["repaired_to"] = "code-selected champion (champion_selector)"
            repairs.append(f"{fname}: champion.model_artifact repointed to {champ_art}")
        if metric and ch.get("primary_metric") != metric:
            ch["primary_metric"] = metric
            repairs.append(f"{fname}: champion.primary_metric set to {metric}")
        if metric and champ.get("score") is not None and ch.get("score") != champ.get("score"):
            ch["score"] = champ.get("score")
            repairs.append(f"{fname}: champion.score set to {champ.get('score')}")
    elif ml.get("champion") not in (None, champ_art):
        ml["champion"] = champ_art
        repairs.append(f"{fname}: champion repointed to {champ_art}")

    best = ml.get("best_model")
    best_name = best.get("name") if isinstance(best, dict) else best
    if best_name and _norm_model_name(str(best_name)) != _norm_model_name(str(champ_name)):
        if isinstance(best, dict):
            best["name"] = champ_name
        else:
            ml["best_model"] = champ_name
        repairs.append(f"{fname}: best_model repointed to {champ_name}")
    if isinstance(ml.get("model_selection"), dict):
        sel = ml["model_selection"]
        cur = sel.get("best_model") or sel.get("best_model_by_primary_metric")
        if cur and _norm_model_name(str(cur)) != _norm_model_name(str(champ_name)):
            sel["best_model"] = champ_name
            sel["best_model_by_primary_metric"] = champ_name
            repairs.append(f"{fname}: model_selection.best_model repointed to {champ_name}")

    arts = ml.get("model_artifacts")
    if isinstance(arts, list):
        for entry in arts:
            if not isinstance(entry, dict):
                continue
            entry_art = entry.get("artifact") or entry.get("artifact_filename") or entry.get("model")
            if champ_art and entry_art == champ_art:
                if entry.get("status") != "champion":
                    entry["status"] = "champion"
                    repairs.append(f"{fname}: {entry_art} marked champion in model_artifacts")
            elif entry.get("status") == "champion":
                entry["status"] = entry.get("stage") or "superseded"
                repairs.append(f"{fname}: {entry_art} demoted from champion in model_artifacts")

    if repairs:
        _dump_json(path, ml)
    return repairs


def _check_stale_threshold_analysis(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag threshold_analysis.json when it no longer describes the champion.

    The analysis stores its at-0.5 default metrics but no model reference, so
    staleness is detected two ways: (a) the metrics don't match the champion's
    reported metrics, or (b) they match a superseded/rejected registry entry
    instead — needed when the champion and the superseded baseline scored
    nearly identically. Also warns when the threshold itself was tuned on the
    held-out test set (selection on the test set inflates the reported gain).
    """
    findings: List[Dict[str, Any]] = []
    ta = _load_json_safe(os.path.join(output_dir, "threshold_analysis.json"))
    if not isinstance(ta, dict) or ta.get("applicable") is False:
        return findings
    dm = ta.get("default_metrics") or {}
    if not isinstance(dm, dict) or champ is None:
        return findings
    winner = (champ.get("_winner_metrics") or {})

    def _matches(metrics: Dict[str, Any], tol: float = 0.01) -> bool:
        compared = [m for m in ("recall", "precision", "f1")
                    if isinstance(dm.get(m), (int, float)) and isinstance(metrics.get(m), (int, float))]
        return bool(compared) and all(abs(float(dm[m]) - float(metrics[m])) <= tol for m in compared)

    champ_match = bool(winner) and _matches(winner)

    # (b) match against non-champion registry entries (superseded baselines).
    superseded_match = False
    reg = _load_json_safe(os.path.join(output_dir, "model_registry.json")) or {}
    for entry in (reg.get("models") or []):
        if not isinstance(entry, dict) or entry.get("status") == "champion":
            continue
        entry_art = entry.get("artifact_filename") or entry.get("model")
        if champ.get("artifact_filename") and entry_art == champ.get("artifact_filename"):
            continue
        if _matches(entry.get("evaluation_metrics") or {}):
            superseded_match = True
            break

    if not champ_match or superseded_match:
        findings.append(_finding(
            "stale_threshold_analysis",
            "threshold_analysis.json default metrics "
            f"(recall={dm.get('recall')}, precision={dm.get('precision')}) match a "
            f"superseded model rather than the current champion '{champ.get('model_name')}' "
            f"(recall={winner.get('recall')}, precision={winner.get('precision')}). The "
            "threshold recommendation belongs to a superseded model and must not be quoted "
            "for the champion.",
            ["threshold_analysis.json", "champion_model.json"],
        ))

    rationale = str(ta.get("selection_rationale") or "").lower()
    if ta.get("recommended_threshold") is not None and ("test set" in rationale or "held-out" in rationale):
        findings.append(_finding(
            "threshold_tuned_on_test_set",
            "threshold_analysis.json selected its recommended threshold by maximizing a metric "
            "on the held-out test set. A threshold optimized on the test set is optimistically "
            "biased; select it via cross-validation on training data and report test-set "
            "performance at that threshold as confirmation only.",
            ["threshold_analysis.json"],
        ))
    return findings


def annotate_stale_threshold_analysis(output_dir: str, champ: Optional[Dict[str, Any]]) -> bool:
    """Stamp threshold_analysis.json with a staleness banner when repairs ran."""
    ta = _load_json_safe(os.path.join(output_dir, "threshold_analysis.json"))
    if not isinstance(ta, dict) or champ is None:
        return False
    stamp = {
        "stale_for_current_champion": True,
        "current_champion_artifact": champ.get("artifact_filename") or champ.get("model"),
        "annotated_by": "champion consistency check",
        "note": "Metrics in this file were computed for a previously selected model; "
                "do not quote them for the current champion.",
    }
    ta["staleness_warning"] = stamp
    return _dump_json(os.path.join(output_dir, "threshold_analysis.json"), ta)


def verify_champion_consistency(output_dir: str, goal_text: str = "", repair: bool = True) -> Dict[str, Any]:
    """Post-ML-task gate: every artifact that names a champion must name THE champion.

    Re-selects the champion by code (which rewrites champion_model.json and
    model_registry.json), repairs every ml_summary*.json champion pointer, and
    reports anything that still disagrees. Never raises.
    """
    result: Dict[str, Any] = {
        "checked_at": datetime.now().isoformat(),
        "output_dir": output_dir,
        "consistent": True,
        "repairs": [],
        "findings": [],
        "champion": None,
    }
    try:
        champ = select_champion(output_dir, goal_text=goal_text)
    except Exception as e:
        champ = None
        result["findings"].append(_finding("champion_selection_error", f"champion selection crashed: {e}", []))
    result["champion"] = champ

    if champ is None:
        # No comparison data yet — nothing to verify (e.g. before model training).
        return result

    champ_art = champ.get("artifact_filename") or champ.get("model")

    # 1. Repair champion pointers in every ml_summary file.
    if repair:
        for fname in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
            if fname.startswith("ml_summary") and fname.endswith(".json"):
                result["repairs"].extend(_repair_ml_champion_pointer(os.path.join(output_dir, fname), champ))
        reg = _load_json_safe(os.path.join(output_dir, "model_registry.json")) or {}
        reg_champs = [m.get("artifact_filename") or m.get("model")
                      for m in (reg.get("models") or [])
                      if isinstance(m, dict) and m.get("status") == "champion"]
        if len(reg_champs) == 1 and reg_champs[0] == champ_art:
            pass  # registry agrees
        else:
            # select_champion already rewrote the registry; re-run it defensively
            try:
                select_champion(output_dir, goal_text=goal_text)
                result["repairs"].append("model_registry.json champion flags rewritten by champion_selector")
            except Exception:
                pass
        if any("stale_threshold" in f["code"] for f in result["findings"]):
            pass  # findings computed below; annotation happens after checks

    # 2. Re-check agreement (registry, ml summaries, comparison tables).
    ml_files = [f for f in (sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else [])
                if f.startswith("ml_summary") and f.endswith(".json")]
    for fname in ml_files:
        ml = _load_json_safe(os.path.join(output_dir, fname)) or {}
        ch = ml.get("champion")
        ch_art = ch.get("model_artifact") if isinstance(ch, dict) else ch
        if ch_art and ch_art != champ_art:
            result["findings"].append(_finding(
                "champion_pointer_disagreement",
                f"{fname} points to '{ch_art}' but the code-selected champion is '{champ_art}'.",
                [fname, "champion_model.json"],
            ))

    reg = _load_json_safe(os.path.join(output_dir, "model_registry.json")) or {}
    reg_champs = [m.get("artifact_filename") or m.get("model")
                  for m in (reg.get("models") or [])
                  if isinstance(m, dict) and m.get("status") == "champion"]
    if reg_champs != [champ_art]:
        result["findings"].append(_finding(
            "champion_registry_disagreement",
            f"model_registry.json marks {reg_champs or 'nothing'} as champion but champion_model.json "
            f"points to '{champ_art}'. All artifacts must agree on the champion.",
            ["model_registry.json", "champion_model.json"],
        ))

    # 3. Stale threshold analysis (belongs to a superseded model / test-set tuned).
    stale_findings = _check_stale_threshold_analysis(output_dir, champ)
    result["findings"].extend(stale_findings)
    if repair and any(f["code"] == "stale_threshold_analysis" for f in stale_findings):
        if annotate_stale_threshold_analysis(output_dir, champ):
            result["repairs"].append("threshold_analysis.json annotated as stale for the current champion")

    result["consistent"] = len(result["findings"]) == 0

    try:
        _dump_json(os.path.join(output_dir, "champion_consistency_report.json"), result)
    except Exception:
        pass

    if logfire:
        if result["findings"]:
            logfire.warn(
                "Champion consistency check: unresolved disagreements after ML task",
                findings=len(result["findings"]),
                repairs=len(result["repairs"]),
                detail=[f["message"] for f in result["findings"]],
            )
        elif result["repairs"]:
            logfire.info("Champion consistency check: pointers repaired to agree",
                         repairs=result["repairs"])
        else:
            logfire.info("Champion consistency check passed: all artifacts agree on the champion",
                         champion=champ_art)
    return result


# ----------------------------------------------------------------------------
# Machine-computed run warnings
# ----------------------------------------------------------------------------

def _run_level_warnings(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []

    cm = _load_json_safe(os.path.join(output_dir, "confusion_matrix.json")) or {}
    tp, fp, fn, tn = (cm.get(k) for k in ("true_positives", "false_positives", "false_negatives", "true_negatives"))
    if all(isinstance(v, (int, float)) for v in (tp, fp, fn, tn)):
        total = tp + fp + fn + tn
        if 0 < total < SMALL_TEST_MAX:
            warnings.append(f"The test set is small ({int(total)} samples); performance metrics may vary considerably on new data.")
        positives = tp + fn
        if positives and positives / total < 0.10:
            warnings.append(f"The classes are unbalanced (positive class is {positives / total:.1%} of the test set); accuracy alone is misleading.")
    else:
        # fall back to dataset-level imbalance if no matrix exists
        try:
            import pandas as pd
            ds_path = os.path.join(output_dir, "cleaned_dataset.csv")
            if os.path.isfile(ds_path):
                df = pd.read_csv(ds_path)
                bool_cols = df.select_dtypes(include=["bool", "int64", "float64"]).nunique()
                bin_cols = [c for c in bool_cols.index if bool_cols[c] == 2]
                if bin_cols:
                    rate = df[bin_cols[0]].mean()
                    if 0 < rate < 0.10 or rate > 0.90:
                        warnings.append(f"The classes are unbalanced (minority outcome is {min(rate, 1 - rate):.1%} of the data); accuracy alone is misleading.")
        except Exception:
            pass

    ranking = (champ or {}).get("ranking") or []
    metric = (champ or {}).get("primary_metric_name")
    if len(ranking) >= 2 and metric and ranking[0].get(metric) is not None and ranking[1].get(metric) is not None:
        gap = abs(float(ranking[0][metric]) - float(ranking[1][metric]))
        if gap < NEAR_TIE_GAP:
            warnings.append(
                f"The top two models are nearly tied on {metric.upper()} "
                f"({ranking[0]['model']}: {ranking[0][metric]:.4f} vs {ranking[1]['model']}: {ranking[1][metric]:.4f}); "
                "the choice between them should not be treated as decisive."
            )
    return warnings


# ----------------------------------------------------------------------------
# Gate entry point
# ----------------------------------------------------------------------------

def run_preflight(output_dir: str, goal_text: str = "") -> Dict[str, Any]:
    """Run consistency checks + one automatic repair pass before the reporter.

    Returns {"valid": bool, "findings": [...], "warnings": [...], "repairs": [...],
             "champion": {...}|None}. Never raises and never stops the workflow:
    residual findings are converted into warnings for the reporter.
    """
    result: Dict[str, Any] = {
        "checked_at": datetime.now().isoformat(),
        "output_dir": output_dir,
        "valid": True,
        "findings": [],
        "warnings": [],
        "repairs": [],
        "champion": None,
    }

    # Pass 1: select champion by code, then run all checks.
    try:
        champ = select_champion(output_dir, goal_text=goal_text)
    except Exception as e:
        print(f"[PreflightGate] champion selection error: {e}")
        champ = None
    result["champion"] = champ

    try:
        findings = _run_all(output_dir, champ)
    except Exception as e:
        print(f"[PreflightGate] initial check pass error: {e}")
        findings = [_finding("gate_error", f"initial checks crashed: {e}", [])]

    # Repair pass
    if findings:
        repairs: List[str] = []
        if champ is not None:
            if _repair_ml_summary_champion(output_dir, champ):
                repairs.append("ml_summary.json best-model pointer rewritten to the code-selected champion")
            for fname in sorted(os.listdir(output_dir)):
                if fname.startswith("ml_summary") and fname.endswith(".json"):
                    repairs.extend(_repair_ml_champion_pointer(os.path.join(output_dir, fname), champ))
        if any(f["code"] == "confusion_matrix_mismatch" for f in findings):
            if _repair_confusion_matrix(output_dir):
                repairs.append("confusion_matrix.json recomputed from predictions.csv")
        if any(f["code"] == "shape_mismatch" for f in findings):
            if _repair_dataset_profile(output_dir):
                repairs.append("dataset_profile.json shape counts rewritten from the files on disk")
        if any(f["code"] == "stale_threshold_analysis" for f in findings):
            if annotate_stale_threshold_analysis(output_dir, champ):
                repairs.append("threshold_analysis.json annotated as stale for the current champion")
        result["repairs"] = repairs

        # Pass 2: re-run checks once after repairs
        try:
            findings = _run_all(output_dir, champ)
        except Exception as e:
            findings = [_finding("gate_error", f"post-repair checks crashed: {e}", [])]

    # Warn-not-stop: residual findings become warnings carried into the report.
    result["findings"] = findings
    result["warnings"] = _run_level_warnings(output_dir, champ) + [
        f"Consistency check not resolved automatically: {f['message']}" for f in findings
    ]
    result["valid"] = len(findings) == 0

    try:
        _dump_json(os.path.join(output_dir, "preflight_report.json"), result)
    except Exception:
        pass

    if logfire:
        if findings:
            logfire.warn(
                "Preflight gate: unresolved consistency findings (workflow continues, warnings passed to reporter)",
                findings=len(findings),
                warnings=len(result["warnings"]),
                repairs=len(result["repairs"]),
                detail=[f["message"] for f in findings],
            )
        elif result["warnings"]:
            logfire.info(
                "Preflight gate passed all consistency checks (informational run warnings present)",
                warnings=len(result["warnings"]),
                detail=result["warnings"],
            )
        else:
            logfire.info("Preflight gate passed all consistency checks (no warnings)")
    return result


def _run_all(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    findings.extend(_check_champion(output_dir, champ))
    findings.extend(_check_confusion_matrix(output_dir, champ))
    findings.extend(_check_file_lists(output_dir))
    findings.extend(_check_shape_counts(output_dir))
    findings.extend(_check_stale_threshold_analysis(output_dir, champ))
    return findings
