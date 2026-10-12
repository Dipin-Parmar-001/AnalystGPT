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

Key Design Principles:
- Preflight is READ-ONLY by default (does not modify artifacts)
- Champion selection is done ONCE by orchestrator; preflight only validates
- Validation and repair are separate stages
- Explicit error codes for different failure types
- champion_model.json is the canonical selection record
- Structured results passed to reporter
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agents.logger import setup_logfire
from agents.champion_selector import _load_comparison_rows, _norm_model_name, ChampionSelectionResult, ChampionSelectionStatus
from agents.schema import ChampionModel
import re

try:
    import logfire
except ImportError:
    logfire = None

setup_logfire()

TOL = 0.02          # tolerance for confusion-matrix round-trip vs reported metrics
NEAR_TIE_GAP = 0.01  # top-2 models closer than this on the primary metric
SMALL_TEST_MAX = 300  # test set smaller than this triggers a warning


class ConsistencyState(Enum):
    """Explicit consistency states - no ambiguity."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    SELECTION_FAILED = "selection_failed"
    REPAIR_FAILED = "repair_failed"
    NO_CHAMPION = "no_champion"


@dataclass
class PreflightFinding:
    """Structured finding with explicit error code."""
    severity: str  # "warning" | "error" | "critical"
    code: str  # NO_COMPARISON_DATA, INVALID_ARTIFACT, REGISTRY_SYNC_FAILED, etc.
    message: str
    artifacts: List[str]
    stage: str = "validation"  # "validation" | "repair"
    original_finding: Optional[Dict[str, Any]] = None  # For tracking repairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "artifacts": self.artifacts,
            "stage": self.stage,
            "original_finding": self.original_finding
        }


@dataclass
class PreflightResult:
    """Complete preflight result with all findings, repairs, and state."""
    valid: bool
    consistency_state: ConsistencyState
    findings: List[PreflightFinding] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repairs_applied: List[str] = field(default_factory=list)
    original_findings: List[PreflightFinding] = field(default_factory=list)
    champion: Optional[Dict[str, Any]] = None
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    output_dir: str = ""

    def add_finding(self, finding: PreflightFinding):
        self.findings.append(finding)
        if finding.severity in ("error", "critical"):
            self.valid = False

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_repair(self, repair: str):
        self.repairs_applied.append(repair)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "consistency_state": self.consistency_state.value,
            "findings": [f.to_dict() for f in self.findings],
            "warnings": self.warnings,
            "repairs_applied": self.repairs_applied,
            "original_findings": [f.to_dict() for f in self.original_findings],
            "champion": self.champion,
            "checked_at": self.checked_at,
            "output_dir": self.output_dir
        }


def _load_json_with_error(path: str) -> Tuple[Optional[Any], Optional[str]]:
    """Loads JSON returning (data, error_str). Differentiates missing vs corrupted JSON."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f), None
    except Exception as exc:
        return None, str(exc)


def normalize_model_family(name_or_file: str) -> str:
    """Normalize model artifact filename or model display name to a canonical model family string."""
    s = str(name_or_file or "").strip()
    s = re.sub(r"\.pkl$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-_]p\d+(?:[-_]tuned)?$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+p\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(tuned\)|\(baseline\)", "", s, flags=re.IGNORECASE)
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "histgradientboostingclassifier": "histgradboost",
        "histgradientboosting": "histgradboost",
        "hist_gradient_boosting": "histgradboost",
        "gradientboostingclassifier": "gradient_boosting",
        "gradientboosting": "gradient_boosting",
        "randomforestclassifier": "random_forest",
        "randomforest": "random_forest",
        "logisticregression": "logistic_regression",
        "svc": "svm",
        "linearsvc": "svm",
    }
    return aliases.get(s, s)


def _clean_str(s: Any) -> str:
    """Lowercase string with only alphanumeric characters, stripping extension if any."""
    if not s:
        return ""
    stem = os.path.splitext(str(s))[0]
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _match_artifact(art1: Optional[str], art2: Optional[str]) -> bool:
    """Check if two model artifact references refer to the same model artifact."""
    if not art1 or not art2:
        return False
    s1, s2 = str(art1).strip(), str(art2).strip()
    if s1 == s2:
        return True
    c1, c2 = _clean_str(s1), _clean_str(s2)
    if c1 and c1 == c2:
        return True
    fam1 = normalize_model_family(s1)
    fam2 = normalize_model_family(s2)
    if fam1 and fam1 != "unknown" and fam1 == fam2:
        p1 = re.search(r'p\d+', s1.lower())
        p2 = re.search(r'p\d+', s2.lower())
        if p1 and p2:
            return p1.group(0) == p2.group(0)
        return True
    return False


def _match_model_names(name1: Optional[str], name2: Optional[str]) -> bool:
    """Check if two model names refer to the same model family/name."""
    if not name1 or not name2:
        return False
    s1, s2 = str(name1).strip(), str(name2).strip()
    if s1 == s2:
        return True
    c1, c2 = _clean_str(s1), _clean_str(s2)
    if c1 and c1 == c2:
        return True
    fam1 = normalize_model_family(s1)
    fam2 = normalize_model_family(s2)
    if fam1 and fam1 != "unknown" and fam1 == fam2:
        return True
    return False


def _find_matching_artifact(name: Optional[str], pkl_files: Any) -> Optional[str]:
    """Find a matching on-disk .pkl file from candidate string."""
    if not name or not pkl_files:
        return None
    name_str = str(name).strip()
    if name_str in pkl_files:
        return name_str
    clean_name = _clean_str(name_str)
    for p in pkl_files:
        if _clean_str(p) == clean_name:
            return p
    target_fam = normalize_model_family(name_str)
    if target_fam and target_fam != "unknown":
        target_phase = re.search(r'p\d+', name_str.lower())
        for p in pkl_files:
            if normalize_model_family(p) == target_fam:
                if target_phase:
                    p_phase = re.search(r'p\d+', p.lower())
                    if p_phase and p_phase.group(0) != target_phase.group(0):
                        continue
                return p
    return None


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


def _finding(severity: str, code: str, message: str, artifacts: List[str], stage: str = "validation") -> PreflightFinding:
    return PreflightFinding(severity=severity, code=code, message=message, artifacts=artifacts, stage=stage)


def _extract_registry_models(reg: Any) -> List[Dict[str, Any]]:
    """Safely extracts the list of model entries whether model_registry.json is a list or dict."""
    if isinstance(reg, list):
        return [m for m in reg if isinstance(m, dict)]
    elif isinstance(reg, dict):
        models = reg.get("models")
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict)]
        entries = [v for v in reg.values() if isinstance(v, dict) and any(k in v for k in ("model_id", "artifact_filename", "artifact", "model", "status"))]
        if entries:
            return entries
    return []


# ----------------------------------------------------------------------------
# Checks (each returns a list of PreflightFinding)
# ----------------------------------------------------------------------------

def _check_champion(output_dir: str, champ: Optional[Dict[str, Any]], selector_result: Optional[ChampionSelectionResult] = None) -> List[PreflightFinding]:
    findings: List[PreflightFinding] = []
    pkl_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".pkl")) if os.path.isdir(output_dir) else []

    # Handle case where champion selection failed - check why
    if champ is None:
        if selector_result and selector_result.status != ChampionSelectionStatus.SUCCESS:
            # Map selector status to explicit error code
            code_map = {
                ChampionSelectionStatus.NO_COMPARISON_DATA: "NO_COMPARISON_DATA",
                ChampionSelectionStatus.NO_VALID_CANDIDATES: "NO_VALID_CANDIDATES",
                ChampionSelectionStatus.NO_VALID_ARTIFACT: "INVALID_ARTIFACT",
                ChampionSelectionStatus.NO_METRIC_FOUND: "NO_METRIC_FOUND",
                ChampionSelectionStatus.SAVE_FAILED: "CHAMPION_SAVE_FAILED",
                ChampionSelectionStatus.REGISTRY_SYNC_FAILED: "REGISTRY_SYNC_FAILED",
                ChampionSelectionStatus.EXCEPTION: "SELECTION_EXCEPTION",
            }
            code = code_map.get(selector_result.status, "CHAMPION_SELECTION_FAILED")
            findings.append(_finding(
                "error",
                code,
                selector_result.error_message or f"Champion selection failed: {selector_result.status.value}",
                ["champion_model.json", "model_comparison.csv"],
            ))
        elif os.path.exists(os.path.join(output_dir, "model_comparison.csv")):
            findings.append(_finding(
                "error",
                "INVALID_ARTIFACT" if os.path.exists(os.path.join(output_dir, "champion_model.json")) else "SELECTION_FAILED",
                "Comparison data exists but champion_model.json is missing or invalid; selection outcome is unknown because preflight does not rerun selection.",
                ["model_comparison.csv"]
            ))
        else:
            findings.append(_finding(
                "warning", "NO_COMPARISON_DATA",
                "No canonical champion or model comparison data is available to validate yet.",
                ["champion_model.json", "model_comparison.csv"],
            ))
        return findings

    try:
        ChampionModel.model_validate(champ)
    except Exception as exc:
        return [_finding("error", "SCHEMA_ERROR", f"champion_model.json does not match the champion schema: {exc}", ["champion_model.json"])]

    art = champ.get("artifact_filename") or champ.get("model")
    matched_pkl = _find_matching_artifact(art, pkl_files)
    if art and not matched_pkl:
        findings.append(_finding(
            "error",
            "INVALID_ARTIFACT",
            f"champion_model.json points to '{art}', which does not exist on disk. Available: {', '.join(pkl_files) or 'none'}.",
            ["champion_model.json"],
        ))
    elif matched_pkl and matched_pkl != art:
        art = matched_pkl
        champ["model"] = matched_pkl
        champ["artifact_filename"] = matched_pkl

    # champion_model.json is authoritative; the registry must point to the same
    # artifact with exactly one champion status.
    registry = _load_json_safe(os.path.join(output_dir, "model_registry.json"))
    if registry is not None:
        models = _extract_registry_models(registry)
        registry_champions = [
            item.get("artifact_filename") or item.get("model")
            for item in models
            if isinstance(item, dict) and str(item.get("status", "")).lower() == "champion"
        ]
        has_match = (
            len(registry_champions) == 1
            and _match_artifact(registry_champions[0], art)
        )
        if not has_match:
            findings.append(_finding(
                "error", "CHAMPION_REGISTRY_DISAGREEMENT",
                f"model_registry.json champion entries {registry_champions or 'none'} do not match canonical champion '{art}'.",
                ["model_registry.json", "champion_model.json"],
            ))

    # ml_summary.json agreement
    ml = _load_json_safe(os.path.join(output_dir, "ml_summary.json"))
    if isinstance(ml, dict):
        summary_best = (ml.get("best_model") or {}).get("name") if isinstance(ml.get("best_model"), dict) else ml.get("best_model")
        if summary_best and not _match_model_names(str(summary_best), champ.get("model_name") or champ.get("model") or ""):
            findings.append(_finding(
                "warning",
                "CHAMPION_DISAGREEMENT_ML_SUMMARY",
                f"ml_summary.json names '{summary_best}' as best model but code-selected champion is '{champ.get('model_name')}'.",
                ["ml_summary.json", "champion_model.json"],
            ))
    return findings


def _check_confusion_matrix(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[PreflightFinding]:
    findings: List[PreflightFinding] = []
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
                "error",
                "CONFUSION_MATRIX_MISMATCH",
                f"confusion_matrix.json implies {name}={derived:.4f} but the champion's reported {name} is {rep:.4f} — the matrix may belong to a different model.",
                ["confusion_matrix.json", "champion_model.json"],
            ))
    return findings


def _check_file_lists(output_dir: str) -> List[PreflightFinding]:
    findings: List[PreflightFinding] = []
    if not os.path.isdir(output_dir):
        return findings
    on_disk = set(os.listdir(output_dir))

    ml = _load_json_safe(os.path.join(output_dir, "ml_summary.json"))
    if isinstance(ml, dict):
        for gf in (ml.get("generated_artifacts") or []):
            fname = gf.get("file") if isinstance(gf, dict) else gf
            if fname and fname not in on_disk:
                findings.append(_finding(
                    "warning",
                    "PHANTOM_FILE_REFERENCE",
                    f"ml_summary.json lists artifact '{fname}' which does not exist in the run folder.",
                    ["ml_summary.json"]
                ))
    return findings


def _check_shape_counts(output_dir: str) -> List[PreflightFinding]:
    findings: List[PreflightFinding] = []
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
                "error",
                "SHAPE_MISMATCH",
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


def _check_stale_threshold_analysis(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[PreflightFinding]:
    """Flag threshold_analysis.json when it no longer describes the champion.

    Model identity is checked first. Legacy files without identity fall back
    to metric comparisons against the champion and superseded registry entries.
    Also warns when the threshold was tuned on the held-out test set.
    """
    findings: List[PreflightFinding] = []
    ta = _load_json_safe(os.path.join(output_dir, "threshold_analysis.json"))
    if not isinstance(ta, dict) or ta.get("applicable") is False:
        return findings
    dm = ta.get("default_metrics") or {}
    if not isinstance(dm, dict) or champ is None:
        return findings
    champion_artifact = champ.get("artifact_filename") or champ.get("model")
    # A persisted identity is authoritative. Metric-only matching can falsely
    # accept a different model when two models happen to have similar scores.
    recorded_artifact = ta.get("model_artifact") or ta.get("artifact_filename")
    recorded_id = ta.get("model_id")
    champion_id = os.path.splitext(str(champion_artifact or ""))[0]
    if recorded_artifact and recorded_artifact != champion_artifact:
        findings.append(_finding(
            "warning", "STALE_THRESHOLD_ANALYSIS",
            f"threshold_analysis.json belongs to '{recorded_artifact}', but the canonical champion is '{champion_artifact}'. Its operating points must not be reported for the champion.",
            ["threshold_analysis.json", "champion_model.json"],
        ))
        return findings
    if recorded_id and str(recorded_id) not in {champion_id, str(champ.get("model_id") or "")}:
        findings.append(_finding(
            "warning", "STALE_THRESHOLD_ANALYSIS",
            f"threshold_analysis.json model_id '{recorded_id}' does not match champion '{champion_id}'. Its operating points must not be reported for the champion.",
            ["threshold_analysis.json", "champion_model.json"],
        ))
        return findings
    winner = (champ.get("_winner_metrics") or {})

    def _matches(metrics: Any, tol: float = 0.01) -> bool:
        if not isinstance(metrics, dict) or not isinstance(dm, dict):
            return False
        compared = [m for m in ("recall", "precision", "f1")
                    if isinstance(dm.get(m), (int, float)) and isinstance(metrics.get(m), (int, float))]
        return bool(compared) and all(abs(float(dm[m]) - float(metrics[m])) <= tol for m in compared)

    champ_match = bool(winner) and _matches(winner)

    # (b) match against non-champion registry entries (superseded baselines).
    superseded_match = False
    reg = _load_json_safe(os.path.join(output_dir, "model_registry.json")) or {}
    for entry in _extract_registry_models(reg):
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
            "warning",
            "STALE_THRESHOLD_ANALYSIS",
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
            "warning",
            "THRESHOLD_TUNED_ON_TEST_SET",
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
    # Keep the old measurements for audit, but prevent report prompts from
    # treating them as the current champion's operating points.
    ta["applicable"] = False
    ta["stale_for_current_champion"] = True
    ta["stale_reason"] = stamp["note"]
    return _dump_json(os.path.join(output_dir, "threshold_analysis.json"), ta)


def _sync_threshold_analysis_ml_summary(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[str]:
    """Check threshold_analysis and ml_summary.json alignment.

    If they are improving the same key metric (or threshold analysis belongs to the champion),
    sync ml_summary.json's threshold block to match threshold_analysis.json.
    If there are discrepancies, log a warning to logs/logfire ONLY without giving a warning
    to the next step (reporter).
    """
    repairs: List[str] = []
    ta_path = os.path.join(output_dir, "threshold_analysis.json")
    ta = _load_json_safe(ta_path)
    if not isinstance(ta, dict) or ta.get("applicable") is False:
        return repairs

    champ_art = (champ.get("artifact_filename") or champ.get("model")) if isinstance(champ, dict) else None
    ta_art = ta.get("model_artifact") or ta.get("artifact_filename")
    if champ_art and ta_art and ta_art != champ_art:
        return repairs

    primary_metric = champ.get("primary_metric") if isinstance(champ, dict) else "recall"

    for fname in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
        if fname.startswith("ml_summary") and fname.endswith(".json"):
            path = os.path.join(output_dir, fname)
            summary = _load_json_safe(path)
            if not isinstance(summary, dict):
                continue
            ml_ta = summary.get("threshold_analysis")
            if isinstance(ml_ta, dict):
                ta_rec = ta.get("recommended_threshold")
                ml_rec = ml_ta.get("recommended_threshold")
                if ta_rec is not None and ml_rec is not None and abs(float(ta_rec) - float(ml_rec)) > 0.001:
                    # Log warning to logs only, do NOT raise or propagate warning to next step
                    if logfire:
                        logfire.warn(
                            f"{fname} and threshold_analysis.json recommended threshold mismatch",
                            threshold_analysis=ta_rec,
                            ml_summary=ml_rec,
                            primary_metric=primary_metric
                        )
                    else:
                        print(f"[PreflightGate] Note: {fname} (threshold {ml_rec}) and threshold_analysis.json (threshold {ta_rec}) differ. Syncing ml_summary to threshold_analysis.")
                    # Sync ml_summary.json to match threshold_analysis.json
                    summary["threshold_analysis"] = {
                        "applicable": ta.get("applicable", True),
                        "reason_if_not_applicable": ta.get("reason_if_not_applicable"),
                        "default_threshold": ta.get("default_threshold", 0.5),
                        "default_metrics": ta.get("default_metrics", {}),
                        "recommended_threshold": ta.get("recommended_threshold"),
                        "recommended_metrics": ta.get("recommended_metrics", {}),
                        "selection_rationale": ta.get("selection_rationale", "")
                    }
                    if _dump_json(path, summary):
                        repairs.append(f"{fname}: threshold_analysis synced to match threshold_analysis.json")
    return repairs


def _repair_registry_champion(output_dir: str, champ: Dict[str, Any]) -> bool:
    """Make the registry reflect champion_model.json, the canonical pointer."""
    path = os.path.join(output_dir, "model_registry.json")
    registry = _load_json_safe(path)
    if not isinstance(registry, (dict, list)):
        return False
    artifact = champ.get("artifact_filename") or champ.get("model")
    models = _extract_registry_models(registry)
    if not models and not isinstance(registry, list):
        return False
    matched = False
    for entry in models:
        if not isinstance(entry, dict):
            continue
        is_target = (entry.get("artifact_filename") == artifact or _match_artifact(entry.get("artifact_filename"), artifact)) and not matched
        entry["status"] = "champion" if is_target else ("evaluated" if entry.get("status") == "champion" else entry.get("status", "candidate"))
        if is_target:
            entry["artifact_filename"] = artifact
        matched = matched or is_target
    if not matched:
        if not artifact or not os.path.isfile(os.path.join(output_dir, str(artifact))):
            return False
        model_id = os.path.splitext(os.path.basename(str(artifact)))[0]
        family = model_id.rsplit("_p", 1)[0] if "_p" in model_id else model_id
        new_entry = {
            "model_id": model_id,
            "model_family": family,
            "artifact_filename": artifact,
            "version": model_id[len(family) + 1:] if model_id.startswith(family + "_") else "p0",
            "stage": "hyperparameter_tuning" if "_p1" in model_id or "_tuned" in model_id else "initial_comparison",
            "evaluation_metrics": {},
            "primary_selection_metric": champ.get("primary_metric") or "",
            "status": "champion",
        }
        if isinstance(registry, list):
            registry.append(new_entry)
        elif isinstance(registry, dict):
            registry.setdefault("models", []).append(new_entry)
    return _dump_json(path, registry)


def verify_champion_consistency(
    output_dir: str,
    goal_text: str = "",
    repair: bool = False,
    task_type: Optional[str] = None,
    pre_task_champion: Optional[Dict[str, Any]] = None,
    primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Post-ML-task gate: every artifact that names a champion must name THE champion.

    Reads champion_model.json as the canonical selection record and reports
    disagreements. Optional repairs do not re-run model selection. Never raises.
    """
    result: Dict[str, Any] = {
        "checked_at": datetime.now().isoformat(),
        "output_dir": output_dir,
        "consistent": True,
        "repairs": [],
        "findings": [],
        "champion": None,
    }
    # 1. Model registry corruption check
    reg_path = os.path.join(output_dir, "model_registry.json")
    if os.path.isfile(reg_path):
        reg_data, reg_err = _load_json_with_error(reg_path)
        if reg_err is not None:
            result["findings"].append(_finding(
                "error", "CORRUPTED_MODEL_REGISTRY",
                f"model_registry.json exists on disk but is corrupted and unparseable as JSON: {reg_err}. Fix model_registry.json so it is valid JSON.",
                ["model_registry.json"]
            ))

    # 2. Champion model JSON validation
    champ_path = os.path.join(output_dir, "champion_model.json")
    champ, champ_err = _load_json_with_error(champ_path)
    if os.path.isfile(champ_path) and champ_err is not None:
        result["findings"].append(_finding(
            "error", "INVALID_CHAMPION_MODEL",
            f"champion_model.json is corrupted and unparseable as JSON: {champ_err}. Rewrite with valid JSON.",
            ["champion_model.json"]
        ))
    elif not os.path.isfile(champ_path) and task_type in ("model_comparison", "hyperparameter_tuning"):
        result["findings"].append(_finding(
            "error", "MISSING_CHAMPION_MODEL",
            f"champion_model.json was not created after {task_type}. An authoritative champion must be recorded in champion_model.json.",
            ["champion_model.json"]
        ))
    result["champion"] = champ

    champ_art = champ.get("artifact_filename") or champ.get("model") if isinstance(champ, dict) else None
    pkl_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".pkl")] if os.path.isdir(output_dir) else []
    matched_art = _find_matching_artifact(champ_art, pkl_files) if champ_art else None
    if champ_art and not matched_art:
        result["findings"].append(_finding(
            "error", "MISSING_CHAMPION_ARTIFACT",
            f"champion_model.json designates '{champ_art}' as champion, but that .pkl file does not exist on disk.",
            ["champion_model.json", str(champ_art)]
        ))
    elif matched_art and matched_art != champ_art:
        champ_art = matched_art
        if isinstance(champ, dict):
            champ["model"] = matched_art
            champ["artifact_filename"] = matched_art

    # 3. Hyperparameter tuning specific verification
    tuning_path = os.path.join(output_dir, "hyperparameter_tuning_summary.json")
    if task_type == "hyperparameter_tuning" or os.path.isfile(tuning_path):
        if not os.path.isfile(tuning_path):
            result["findings"].append(_finding(
                "error", "MISSING_TUNING_SUMMARY",
                "hyperparameter_tuning_summary.json was not generated by hyperparameter tuning.",
                ["hyperparameter_tuning_summary.json"]
            ))
        else:
            tuning, tuning_err = _load_json_with_error(tuning_path)
            if tuning_err is not None:
                result["findings"].append(_finding(
                    "error", "CORRUPTED_TUNING_SUMMARY",
                    f"hyperparameter_tuning_summary.json is corrupted and unparseable: {tuning_err}.",
                    ["hyperparameter_tuning_summary.json"]
                ))
            elif isinstance(tuning, dict):
                prev_champ = pre_task_champion or {}
                prev_art = (prev_champ.get("artifact_filename") or prev_champ.get("model")
                            or (tuning.get("parent_model") or {}).get("artifact")
                            or (tuning.get("champion_decision") or {}).get("previous_champion"))
                prev_family = normalize_model_family(prev_art or prev_champ.get("model_family") or "")
                prev_score = prev_champ.get("score")
                metric = (primary_metric or prev_champ.get("primary_metric")
                          or (tuning.get("comparison_basis") or {}).get("primary_metric") or "recall")

                challenger_art = ((tuning.get("challenger_model") or {}).get("artifact")
                                  or (tuning.get("frozen_model") or {}).get("artifact"))
                if not challenger_art:
                    tmc_path = os.path.join(output_dir, "tuned_model_comparison.csv")
                    if os.path.isfile(tmc_path):
                        try:
                            import pandas as pd
                            tmc_df = pd.read_csv(tmc_path)
                            c_col = next((c for c in tmc_df.columns if c.strip().lower() in ("artifact", "model_artifact")), None)
                            if c_col:
                                non_p0 = [str(x) for x in tmc_df[c_col].dropna() if "_p0" not in str(x).lower()]
                                if non_p0:
                                    challenger_art = non_p0[0]
                        except Exception:
                            pass
                challenger_family = normalize_model_family(challenger_art or (tuning.get("challenger_model") or {}).get("model_name") or "")

                # Check model family consistency: challenger must belong to the same family as previous champion
                if prev_family and challenger_family and prev_family != challenger_family:
                    result["findings"].append(_finding(
                        "error", "CHAMPION_FAMILY_MISMATCH",
                        f"Hyperparameter tuning evaluated or proposed challenger '{challenger_art}' from model family '{challenger_family}', "
                        f"but the previous champion is '{prev_art}' (model family '{prev_family}'). "
                        f"HPT can ONLY tune the existing champion model family to produce a tuned version (e.g. '{prev_family}_p1.pkl'). "
                        f"It cannot switch the champion to a different model family ('{challenger_family}'). "
                        f"Please re-run hyperparameter tuning targeting the '{prev_family}' model.",
                        ["hyperparameter_tuning_summary.json", "champion_model.json"]
                    ))

                comp = tuning.get("comparison") or {}
                tuned_dict = comp.get("tuned") or {}
                base_dict = comp.get("baseline") or {}

                def _extract_metric_val(m_dict: Dict[str, Any], target_m: str) -> Optional[float]:
                    if not isinstance(m_dict, dict):
                        return None
                    norm_target = str(target_m).strip().lower().replace("-", "_").replace(" ", "_")
                    for k, v in m_dict.items():
                        if str(k).strip().lower().replace("-", "_").replace(" ", "_") == norm_target:
                            try:
                                return float(v)
                            except (ValueError, TypeError):
                                pass
                    return None

                tuned_val = _extract_metric_val(tuned_dict, metric)
                base_val = _extract_metric_val(base_dict, metric)
                if prev_score is not None:
                    try:
                        base_val = float(prev_score)
                    except (ValueError, TypeError):
                        pass

                if tuned_val is not None and base_val is not None:
                    is_ascending = str(metric).lower() in ("loss", "log_loss", "mse", "rmse", "mae")
                    challenger_improved = (tuned_val < base_val) if is_ascending else (tuned_val > base_val)

                    champ_decision = tuning.get("champion_decision") or {}
                    claimed_changed = champ_decision.get("champion_changed", False)
                    claimed_new = champ_decision.get("new_champion", "")

                    if (claimed_changed or claimed_new == challenger_art) and not challenger_improved:
                        result["findings"].append(_finding(
                            "error", "FALSE_CHAMPION_IMPROVEMENT_CLAIM",
                            f"hyperparameter_tuning_summary.json declared '{claimed_new}' as new champion (champion_changed=True), "
                            f"but on {metric}, tuned score ({tuned_val:.4f}) did NOT improve over existing champion '{prev_art}' ({base_val:.4f}). "
                            f"When tuning does not beat the existing champion, the existing champion must be retained with champion_changed: false.",
                            ["hyperparameter_tuning_summary.json", "champion_model.json"]
                        ))
                    elif not claimed_changed and champ_art and prev_art and not _match_artifact(champ_art, prev_art):
                        result["findings"].append(_finding(
                            "error", "CHAMPION_DECISION_DISAGREEMENT",
                            f"Tuning did not improve the champion (champion_changed=False), but champion_model.json points to '{champ_art}'. "
                            f"It must remain '{prev_art}'.",
                            ["champion_model.json", "hyperparameter_tuning_summary.json"]
                        ))
                    elif claimed_changed and challenger_improved and prev_family == challenger_family:
                        if champ_art and challenger_art and not _match_artifact(champ_art, challenger_art):
                            result["findings"].append(_finding(
                                "error", "CHAMPION_DECISION_DISAGREEMENT",
                                f"Tuning improved the champion to '{challenger_art}', but champion_model.json still points to '{champ_art}'. "
                                f"champion_model.json must be updated to the new champion '{challenger_art}'.",
                                ["champion_model.json", "hyperparameter_tuning_summary.json"]
                            ))

    if champ is None and not result["findings"]:
        # No comparison data yet — nothing to verify (e.g. before model training).
        return result
        pass
    if False:
        champ = None
        pass
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

    # 2. Re-check agreement (registry, ml summaries, comparison tables).
    ml_files = [f for f in (sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else [])
                if f.startswith("ml_summary") and f.endswith(".json")]
    for fname in ml_files:
        ml = _load_json_safe(os.path.join(output_dir, fname)) or {}
        ch = ml.get("champion")
        ch_art = ch.get("model_artifact") if isinstance(ch, dict) else ch
        if ch_art and not _match_artifact(ch_art, champ_art):
            result["findings"].append(_finding(
                "error", "CHAMPION_POINTER_DISAGREEMENT",
                f"{fname} points to '{ch_art}' but the code-selected champion is '{champ_art}'.",
                [fname, "champion_model.json"],
            ))

    reg_path = os.path.join(output_dir, "model_registry.json")
    if os.path.isfile(reg_path):
        reg = _load_json_safe(reg_path) or {}
        reg_champs = [m.get("artifact_filename") or m.get("model")
                      for m in _extract_registry_models(reg)
                      if isinstance(m, dict) and m.get("status") == "champion"]
        if not (len(reg_champs) == 1 and _match_artifact(reg_champs[0], champ_art)):
            result["findings"].append(_finding(
                "error", "CHAMPION_REGISTRY_DISAGREEMENT",
                f"model_registry.json marks {reg_champs or 'nothing'} as champion but champion_model.json "
                f"points to '{champ_art}'. All artifacts must agree on the champion.",
                ["model_registry.json", "champion_model.json"],
            ))
            if repair and _repair_registry_champion(output_dir, champ):
                result["repairs"].append("model_registry.json synchronized to champion_model.json")
    # [legacy registry check removed]
    # for m in _extract_registry_models(reg)
    # if isinstance(m, dict) and m.get("status") == "champion"]
    if False:
        result["findings"].append(_finding(
            "error", "CHAMPION_REGISTRY_DISAGREEMENT",
            f"model_registry.json marks {reg_champs or 'nothing'} as champion but champion_model.json "
            f"points to '{champ_art}'. All artifacts must agree on the champion.",
            ["model_registry.json", "champion_model.json"],
        ))
        if repair and _repair_registry_champion(output_dir, champ):
            result["repairs"].append("model_registry.json synchronized to champion_model.json")

    # 3. Stale threshold analysis (belongs to a superseded model / test-set tuned).
    stale_findings = _check_stale_threshold_analysis(output_dir, champ)
    result["findings"].extend(stale_findings)
    if repair and any(f.code == "STALE_THRESHOLD_ANALYSIS" for f in stale_findings):
        if annotate_stale_threshold_analysis(output_dir, champ):
            result["repairs"].append("threshold_analysis.json annotated as stale for the current champion")
        for fname in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
            if fname.startswith("ml_summary") and fname.endswith(".json"):
                path = os.path.join(output_dir, fname)
                summary = _load_json_safe(path)
                block = summary.get("threshold_analysis") if isinstance(summary, dict) else None
                if isinstance(block, dict):
                    block["applicable"] = False
                    block["stale_for_current_champion"] = True
                    block["stale_reason"] = "Threshold analysis belongs to a different model artifact than champion_model.json."
                    _dump_json(path, summary)

    # 4. Sync threshold analysis to ml_summary.json if applicable
    if repair:
        result["repairs"].extend(_sync_threshold_analysis_ml_summary(output_dir, champ))

    result["consistent"] = len([f for f in result["findings"] if getattr(f, "severity", "") == "error"]) == 0

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
                detail=[f.message for f in result["findings"]],
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
    if (
        os.path.isfile(os.path.join(output_dir, "unsupervised_metrics.json"))
        and not os.path.isfile(os.path.join(output_dir, "model_comparison.csv"))
        and not os.path.isfile(os.path.join(output_dir, "champion_model.json"))
    ):
        return warnings

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

def run_preflight(output_dir: str, goal_text: str = "", repair: bool = False) -> Dict[str, Any]:
    """Run consistency checks + one automatic repair pass before the reporter.

    Returns structured PreflightResult with explicit consistency state,
    findings, warnings, and repairs. Never raises and never stops the workflow:
    residual findings are converted into warnings for the reporter.

    Key design:
    - Preflight is READ-ONLY by default (does NOT call select_champion)
    - Champion selection is done ONCE by orchestrator; preflight only validates
    - Validation and repair are separate stages
    - Explicit error codes for different failure types
    - champion_model.json is the canonical selection record
    """
    # Load existing champion from disk (read-only validation)
    champ_path = os.path.join(output_dir, "champion_model.json")
    champ = _load_json_safe(champ_path)
    canonical_champion_valid = False
    if isinstance(champ, dict):
        try:
            ChampionModel.model_validate(champ)
            canonical_champion_valid = True
        except Exception:
            pass
    if isinstance(champ, dict):
        # Enrich only the in-memory view for reporter facts and metric checks.
        # champion_model.json remains the canonical selection record.
        try:
            artifact = champ.get("artifact_filename") or champ.get("model")
            row = next((item for item in _load_comparison_rows(output_dir) if item.get("artifact") == artifact), None)
            if row:
                champ.setdefault("model_name", row.get("model"))
                champ["_winner_metrics"] = row.get("metrics") or {}
                champ.setdefault("primary_metric_name", champ.get("primary_metric"))
        except Exception:
            pass

    # Initialize result
    result = PreflightResult(
        valid=True,
        consistency_state=ConsistencyState.CONSISTENT,
        output_dir=output_dir,
        champion=champ
    )

    # STAGE 1: Validation (read-only)
    try:
        validation_findings = _run_validation(output_dir, champ)
        result.original_findings = validation_findings
        result.findings = validation_findings
        for f in validation_findings:
            if f.severity in ("error", "critical"):
                result.valid = False
    except Exception as e:
        print(f"[PreflightGate] validation pass error: {e}")
        error_finding = _finding("critical", "GATE_ERROR", f"validation checks crashed: {e}", [])
        result.findings = [error_finding]
        result.valid = False

    # Determine consistency state after validation
    if champ is None:
        if os.path.exists(os.path.join(output_dir, "champion_model.json")):
            result.consistency_state = ConsistencyState.INCONSISTENT
        elif os.path.exists(os.path.join(output_dir, "model_comparison.csv")):
            result.consistency_state = ConsistencyState.SELECTION_FAILED
        else:
            result.consistency_state = ConsistencyState.NO_CHAMPION
    elif any(f.severity in ("error", "critical") for f in result.findings):
        result.consistency_state = ConsistencyState.INCONSISTENT

    # STAGE 2: Repair (only if validation found issues)
    if repair and result.findings:
        repairs: List[str] = []

        # Only attempt repairs if we have a valid champion
        if canonical_champion_valid:
            if any(f.code == "CHAMPION_REGISTRY_DISAGREEMENT" for f in result.findings):
                if _repair_registry_champion(output_dir, champ):
                    repairs.append("model_registry.json synchronized to champion_model.json")
            # Repair ml_summary.json champion pointers
            if _repair_ml_summary_champion(output_dir, champ):
                repairs.append("ml_summary.json best-model pointer rewritten to the code-selected champion")
            for fname in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
                if fname.startswith("ml_summary") and fname.endswith(".json"):
                    repairs.extend(_repair_ml_champion_pointer(os.path.join(output_dir, fname), champ))

            # Repair confusion matrix
            if any(f.code == "CONFUSION_MATRIX_MISMATCH" for f in result.findings):
                if _repair_confusion_matrix(output_dir):
                    repairs.append("confusion_matrix.json recomputed from predictions.csv")

            # Repair dataset profile
            if any(f.code == "SHAPE_MISMATCH" for f in result.findings):
                if _repair_dataset_profile(output_dir):
                    repairs.append("dataset_profile.json shape counts rewritten from the files on disk")

            # Annotate stale threshold analysis
            if any(f.code == "STALE_THRESHOLD_ANALYSIS" for f in result.findings):
                if annotate_stale_threshold_analysis(output_dir, champ):
                    repairs.append("threshold_analysis.json annotated as stale for the current champion")
                # A copied summary can otherwise reintroduce the stale threshold
                # block to the reporter even after the source artifact is marked.
                for fname in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
                    if fname.startswith("ml_summary") and fname.endswith(".json"):
                        path = os.path.join(output_dir, fname)
                        summary = _load_json_safe(path)
                        block = summary.get("threshold_analysis") if isinstance(summary, dict) else None
                        if isinstance(block, dict):
                            block["applicable"] = False
                            block["stale_for_current_champion"] = True
                            block["stale_reason"] = "Threshold analysis belongs to a different model artifact than champion_model.json."
                            _dump_json(path, summary)

            # Sync threshold analysis to ml_summary.json if applicable
            repairs.extend(_sync_threshold_analysis_ml_summary(output_dir, champ))

        result.repairs_applied = repairs

        # STAGE 3: Re-validate after repairs
        try:
            post_repair_findings = _run_validation(output_dir, champ)
            result.findings = post_repair_findings
            # Update consistency state
            if champ is None:
                result.consistency_state = ConsistencyState.NO_CHAMPION
            elif any(f.severity in ("error", "critical") for f in result.findings):
                result.consistency_state = ConsistencyState.REPAIR_FAILED
            elif result.repairs_applied:
                result.consistency_state = ConsistencyState.CONSISTENT  # Repaired successfully
            result.valid = len([f for f in post_repair_findings if f.severity in ("error", "critical")]) == 0
        except Exception as e:
            print(f"[PreflightGate] post-repair validation error: {e}")
            error_finding = _finding("critical", "GATE_ERROR", f"post-repair checks crashed: {e}", [])
            result.findings = [error_finding]
            result.valid = False
            result.consistency_state = ConsistencyState.REPAIR_FAILED

    # Add run-level warnings
    result.warnings = _run_level_warnings(output_dir, champ)

    if any("threshold_analysis.json annotated as stale" in repair for repair in result.repairs_applied):
        result.add_warning(
            "threshold_analysis.json belongs to a different model than champion_model.json. "
            "Its thresholds and metrics were excluded from the report; recompute analysis for the current champion."
        )

    # Add warning messages for residual findings
    for f in result.findings:
        if f.severity in ("error", "critical", "warning"):
            result.add_warning(f"Consistency check not resolved automatically: {f.message}")

    # Persist
    try:
        _dump_json(os.path.join(output_dir, "preflight_report.json"), result.to_dict())
    except Exception:
        pass

    if logfire:
        if result.findings:
            logfire.warn(
                "Preflight gate: unresolved consistency findings (workflow continues, warnings passed to reporter)",
                findings=len(result.findings),
                warnings=len(result.warnings),
                repairs=len(result.repairs_applied),
                consistency_state=result.consistency_state.value,
                detail=[f.message for f in result.findings],
            )
        elif result.warnings:
            logfire.info(
                "Preflight gate passed all consistency checks (informational run warnings present)",
                warnings=len(result.warnings),
                detail=result.warnings,
            )
        else:
            logfire.info("Preflight gate passed all consistency checks (no warnings)",
                         consistency_state=result.consistency_state.value)

    return result.to_dict()


def _run_validation(output_dir: str, champ: Optional[Dict[str, Any]]) -> List[PreflightFinding]:
    """Run all validation checks (read-only, no repairs)."""
    findings: List[PreflightFinding] = []
    unsupervised_only = (
        os.path.isfile(os.path.join(output_dir, "unsupervised_metrics.json"))
        and not os.path.isfile(os.path.join(output_dir, "model_comparison.csv"))
        and not os.path.isfile(os.path.join(output_dir, "champion_model.json"))
    )
    if not unsupervised_only:
        findings.extend(_check_champion(output_dir, champ))
    usable_champ = champ if isinstance(champ, dict) else None
    if not unsupervised_only:
        findings.extend(_check_confusion_matrix(output_dir, usable_champ))
    findings.extend(_check_file_lists(output_dir))
    findings.extend(_check_shape_counts(output_dir))
    if not unsupervised_only:
        findings.extend(_check_stale_threshold_analysis(output_dir, usable_champ))
    return findings
