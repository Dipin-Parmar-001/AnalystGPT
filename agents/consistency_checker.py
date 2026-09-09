"""
Cross-Artifact Consistency Checker - AnalystGPT.

Deterministically reconciles the artifacts of a run against each other and against
the authoritative dataset profile. Catches the failure classes observed in
run_20260906_090233 that artifact existence validation cannot see:

- Contradictory dataset shapes across summaries (30 vs 33 vs 44 columns)
- Phantom columns / features named in summaries but absent from the datasets
- Missing-value contradictions (imputation claimed vs missing cells present)
- Stale/phantom champion model pointers and registry entries
- Mislabeled tuned-comparison rows (e.g. "XGBoost (parent)" vs actual artifact)
- Byte-identical duplicate summaries across tasks
- Report references to artifacts that were never generated

Every finding carries a severity ("critical" or "minor") so the orchestrator can
apply the severity-based failure policy.
"""

import os
import json
import glob
import hashlib
import contextlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from agents.run_context import build_dataset_profile
from agents.logger import setup_logfire

try:
    import logfire
except ImportError:
    logfire = None

setup_logfire()

SEVERITY_CRITICAL = "critical"
SEVERITY_MINOR = "minor"


@dataclass
class Finding:
    severity: str          # "critical" | "minor"
    code: str              # machine-readable check id, e.g. "shape_mismatch"
    message: str
    artifacts: List[str] = field(default_factory=list)


@dataclass
class ConsistencyReport:
    output_dir: str
    checked_at: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def critical_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_CRITICAL]

    @property
    def minor_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_MINOR]

    @property
    def is_valid(self) -> bool:
        return len(self.critical_findings) == 0

    def counts(self) -> Dict[str, int]:
        return {
            "critical": len(self.critical_findings),
            "minor": len(self.minor_findings),
            "total": len(self.findings),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "checked_at": self.checked_at,
            "counts": self.counts(),
            "is_valid": self.is_valid,
            "findings": [asdict(f) for f in self.findings],
        }

    def save(self, filename: str = "consistency_report.json") -> Optional[str]:
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            return path
        except Exception as e:
            print(f"[ConsistencyChecker] Notice: could not save report: {e}")
            return None


def _load_json_safe(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def _walk_json(data: Any, path: str = ""):
    """Yields (path, key, value) for every dict entry in a nested structure."""
    if isinstance(data, dict):
        for k, v in data.items():
            yield path + str(k), k, v
            yield from _walk_json(v, f"{path}{k}.")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            yield from _walk_json(v, f"{path}{i}.")


# ---------------------------------------------------------------- shape checks

def _shape_like(value: Any) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """Recognizes shape dicts like {"rows": .., "columns": ..} or {"rows": .., "cols": ..}."""
    if not isinstance(value, dict):
        return None
    rows = value.get("rows", value.get("n_rows", value.get("num_rows")))
    cols = value.get("columns", value.get("cols", value.get("n_columns", value.get("num_columns"))))
    if isinstance(cols, list):
        cols = len(cols)
    if isinstance(rows, int) or isinstance(cols, int):
        try:
            return (int(rows) if isinstance(rows, int) else None,
                    int(cols) if isinstance(cols, int) else None)
        except (TypeError, ValueError):
            return None
    return None


def check_dataset_shape_claims(output_dir: str, profile: Dict[str, Any]) -> List[Finding]:
    """
    Reconciles every shape claim in summary JSONs against the profiled datasets.

    Lifecycle-stage ambiguity is handled by match-any semantics: a processed-shape
    claim is only wrong if it matches NO dataset artifact on disk (e.g. claiming
    54 columns when raw/cleaned/FE have 33/44/50). initial_shape claims are
    compared strictly against the raw dataset when one is profiled.
    """
    findings: List[Finding] = []
    datasets = profile.get("datasets", {})
    ds_entries = [ds for ds in datasets.values() if ds and not ds.get("error")]

    def _matches_any(claimed: Tuple[Optional[int], Optional[int]]) -> bool:
        c_rows, c_cols = claimed
        for ds in ds_entries:
            r_ok = c_rows is None or ds.get("rows") is None or c_rows == ds["rows"]
            c_ok = c_cols is None or ds.get("cols") is None or c_cols == ds["cols"]
            if r_ok and c_ok and (c_rows is not None or c_cols is not None):
                return True
        return False

    def _compare(claimed: Tuple[Optional[int], Optional[int]], actual_key: str, source: str, label: str):
        actual = datasets.get(actual_key)
        if not actual or actual.get("error"):
            return
        c_rows, c_cols = claimed
        if c_rows is not None and actual.get("rows") is not None and c_rows != actual["rows"]:
            findings.append(Finding(
                SEVERITY_CRITICAL, "shape_mismatch",
                f"{source}: claimed {label} has {c_rows} rows, but '{actual['file']}' actually has {actual['rows']} rows.",
                [source, actual["file"]],
            ))
        if c_cols is not None and actual.get("cols") is not None and c_cols != actual["cols"]:
            findings.append(Finding(
                SEVERITY_CRITICAL, "shape_mismatch",
                f"{source}: claimed {label} has {c_cols} columns, but '{actual['file']}' actually has {actual['cols']} columns.",
                [source, actual["file"]],
            ))

    for summary_path in sorted(glob.glob(os.path.join(output_dir, "*summary*.json"))):
        if os.path.basename(summary_path) == "dataset_profile.json":
            continue
        summary = _load_json_safe(summary_path)
        if summary is None:
            continue
        src = os.path.basename(summary_path)

        for jpath, _key, value in _walk_json(summary):
            shape = _shape_like(value)
            if shape is None:
                continue
            if ".initial_shape" in jpath or jpath.startswith("initial_shape"):
                _compare(shape, "raw", src, f"initial shape ({jpath})")
            else:
                if not _matches_any(shape):
                    actual_desc = "; ".join(
                        f"{ds['file']}={ds['rows']}x{ds['cols']}" for ds in ds_entries if ds.get("rows")
                    ) or "no dataset artifacts profiled"
                    findings.append(Finding(
                        SEVERITY_CRITICAL, "shape_mismatch",
                        f"{src}: claimed shape ({jpath}) {shape[0] or '?'} rows x {shape[1] or '?'} columns matches NO dataset artifact on disk ({actual_desc}).",
                        [src],
                    ))

    # Engineered-feature claims must be a subset of the real engineered set
    # (columns added by FE relative to the cleaned dataset).
    real_engineered = set(profile.get("engineered_features") or [])
    fe_summary = _load_json_safe(os.path.join(output_dir, "feature_engineering_summary.json"))
    if fe_summary and real_engineered:
        claimed: List[str] = []
        _collect_referenced_columns(fe_summary, claimed)
        # Only compare against explicit engineered-feature keys
        claimed = []
        for jpath, key, value in _walk_json(fe_summary):
            if key in ("engineered_features", "features_created", "derived_features") and isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        claimed.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("name"), str):
                        claimed.append(item["name"])
        bogus = [c for c in dict.fromkeys(claimed) if c not in real_engineered and c in
                 set().union(*[set(ds.get("columns") or []) for ds in ds_entries])]
        if bogus:
            findings.append(Finding(
                SEVERITY_CRITICAL, "non_engineered_column_claimed",
                f"feature_engineering_summary.json claims these pre-existing column(s) as engineered: {', '.join(bogus[:8])}. Real engineered features are: {', '.join(sorted(real_engineered))}.",
                ["feature_engineering_summary.json"],
            ))

    return findings


# ------------------------------------------------------------- column checks

_COLUMN_LIST_KEYS = {
    "engineered_features", "features_used", "features_created", "derived_features",
    "numerical_columns", "categorical_columns", "numeric_columns", "feature_columns",
    "columns", "top_features",
}


def _collect_referenced_columns(data: Any, bucket: List[str]):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in _COLUMN_LIST_KEYS and isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and 1 <= len(item) <= 80:
                        bucket.append(item)
                    elif isinstance(item, dict):
                        name = item.get("feature") or item.get("name") or item.get("column")
                        if isinstance(name, str):
                            bucket.append(name)
            else:
                _collect_referenced_columns(v, bucket)
    elif isinstance(data, list):
        for item in data:
            _collect_referenced_columns(item, bucket)


def check_phantom_columns(output_dir: str, profile: Dict[str, Any]) -> List[Finding]:
    """Flags column names referenced in summaries that exist in no dataset on disk."""
    findings: List[Finding] = []
    all_columns: set = set()
    for ds in profile.get("datasets", {}).values():
        all_columns.update(ds.get("columns") or [])
    if not all_columns:
        return findings

    engineered = set(profile.get("engineered_features") or [])

    for summary_path in sorted(glob.glob(os.path.join(output_dir, "*summary*.json"))):
        if os.path.basename(summary_path) == "dataset_profile.json":
            continue
        summary = _load_json_safe(summary_path)
        if summary is None:
            continue
        referenced: List[str] = []
        _collect_referenced_columns(summary, referenced)

        phantom = []
        for col in dict.fromkeys(referenced):
            if col in all_columns:
                continue
            # Skip obvious non-column strings (sentences, file names, metrics)
            if " " in col.strip() and len(col.split()) > 3:
                continue
            if col.endswith((".csv", ".json", ".pkl", ".png")) or col.lower() in {
                "roc-auc", "roc_auc", "pr-auc", "f1", "accuracy", "precision", "recall",
            }:
                continue
            phantom.append(col)

        if phantom:
            sev = SEVERITY_CRITICAL if any(c in engineered for c in phantom) or "feature" in os.path.basename(summary_path).lower() else SEVERITY_MINOR
            findings.append(Finding(
                sev, "phantom_columns",
                f"{os.path.basename(summary_path)} references {len(phantom)} column(s) that exist in no dataset artifact: {', '.join(phantom[:8])}.",
                [os.path.basename(summary_path)],
            ))

    return findings


# ------------------------------------------------------ missing-values checks

def check_missing_values_consistency(output_dir: str, profile: Dict[str, Any]) -> List[Finding]:
    """Flags imputation claims contradicted by actual missing cells (and vice versa)."""
    findings: List[Finding] = []
    cleaned = profile.get("datasets", {}).get("cleaned")
    raw = profile.get("datasets", {}).get("raw")

    cleaning_summary = None
    for cand in ("cleaning_summary.json", "analyst_summary.json"):
        data = _load_json_safe(os.path.join(output_dir, cand))
        if isinstance(data, dict):
            cleaning_summary = (cand, data)
            break
    if not cleaning_summary:
        return findings

    src, summary = cleaning_summary
    cleaning_block = summary.get("data_cleaning") or summary
    handled = cleaning_block.get("missing_values_handled")
    methods = cleaning_block.get("imputation_methods")

    cleaned_missing = cleaned.get("missing_cells") if cleaned and not cleaned.get("error") else None
    raw_missing = raw.get("missing_cells") if raw and not raw.get("error") else None

    if isinstance(methods, dict) and methods and cleaned_missing == 0 and (raw_missing or 0) == 0:
        findings.append(Finding(
            SEVERITY_MINOR, "imputation_claim_without_missing",
            f"{src} describes imputation methods ({methods}) but neither the raw nor cleaned dataset contains missing cells.",
            [src],
        ))

    if cleaned_missing and cleaned_missing > 0 and isinstance(handled, dict) and not handled:
        findings.append(Finding(
            SEVERITY_MINOR, "unreported_remaining_missing",
            f"{src} reports no missing-value handling, but the cleaned dataset still contains {cleaned_missing} missing cells. The report must state this explicitly.",
            [src],
        ))

    return findings


# ------------------------------------------------------- model lineage checks

def check_model_artifact_consistency(output_dir: str) -> List[Finding]:
    """Reconciles champion pointer, model registry, and comparison tables with the pkl files on disk."""
    findings: List[Finding] = []
    pkl_files = {os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "*.pkl"))}

    champion = _load_json_safe(os.path.join(output_dir, "champion_model.json"))
    if isinstance(champion, dict):
        champ_artifact = champion.get("model") or champion.get("artifact_filename")
        if champ_artifact and champ_artifact not in pkl_files:
            findings.append(Finding(
                SEVERITY_CRITICAL, "stale_champion_pointer",
                f"champion_model.json points to '{champ_artifact}', which does not exist on disk. Available: {', '.join(sorted(pkl_files)) or 'none'}.",
                ["champion_model.json"],
            ))

    registry = _load_json_safe(os.path.join(output_dir, "model_registry.json"))
    registry_champion_artifacts = set()
    if registry is not None:
        entries = registry.get("models") if isinstance(registry, dict) else registry
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                artifact = entry.get("artifact_filename") or entry.get("artifact")
                if artifact and artifact not in pkl_files:
                    findings.append(Finding(
                        SEVERITY_CRITICAL, "registry_missing_artifact",
                        f"model_registry.json entry '{entry.get('model_id', '?')}' references '{artifact}', which does not exist on disk.",
                        ["model_registry.json"],
                    ))
                if isinstance(entry.get("status"), str) and entry["status"].lower() == "champion" and artifact:
                    registry_champion_artifacts.add(artifact)

        if isinstance(champion, dict) and registry_champion_artifacts:
            champ_artifact = champion.get("model") or champion.get("artifact_filename")
            if champ_artifact and champ_artifact not in registry_champion_artifacts:
                findings.append(Finding(
                    SEVERITY_CRITICAL, "champion_registry_disagreement",
                    f"champion_model.json points to '{champ_artifact}' but model_registry.json marks {sorted(registry_champion_artifacts)} as champion. All artifacts must agree on the champion.",
                    ["champion_model.json", "model_registry.json"],
                ))

    # tuned_model_comparison.csv rows must reference real, correctly-labeled artifacts
    tuned_path = os.path.join(output_dir, "tuned_model_comparison.csv")
    if os.path.exists(tuned_path):
        try:
            import pandas as pd
            tmc = pd.read_csv(tuned_path)
            model_col = next((c for c in tmc.columns if c.strip().lower() in ("model", "algorithm", "name")), None)
            if model_col:
                for name in tmc[model_col].dropna().astype(str):
                    if "(" in name and ")" in name:
                        findings.append(Finding(
                            SEVERITY_MINOR, "annotated_model_label",
                            f"tuned_model_comparison.csv contains annotated model label '{name}'. Labels must name the actual model family/artifact (e.g. 'logistic_regression_p1_tuned'), not '(parent)'/'(tuned)' annotations.",
                            ["tuned_model_comparison.csv"],
                        ))
                        break
        except Exception:
            pass

    return findings


# ---------------------------------------------------- duplicate summaries check

def check_duplicate_summaries(output_dir: str) -> List[Finding]:
    """Flags distinct per-task summary files whose content is byte-identical."""
    findings: List[Finding] = []
    groups: Dict[str, List[str]] = {}
    for prefix in ("analyst_summary", "ml_summary"):
        hashes: Dict[str, str] = {}
        for path in sorted(glob.glob(os.path.join(output_dir, f"{prefix}*.json"))):
            try:
                with open(path, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                continue
            fname = os.path.basename(path)
            if fname.lower() == f"{prefix}.json":
                continue  # main summary legitimately mirrors one task summary
            hashes.setdefault(digest, []).append(fname)
        for digest, files in hashes.items():
            if len(files) > 1:
                groups[digest] = files
        if groups:
            findings.append(Finding(
                SEVERITY_MINOR, "duplicate_summaries",
                f"Multiple per-task {prefix} files are byte-identical ({groups}); each task must produce its own summary content.",
                [f for fl in groups.values() for f in fl],
            ))
            groups = {}
    return findings


# ------------------------------------------------------------ report references

def check_report_artifact_references(output_dir: str, report_text: str) -> List[Finding]:
    """Flags artifact files referenced in the final report that were never generated."""
    import re
    findings: List[Finding] = []
    if not report_text:
        return findings
    referenced = set(re.findall(r"\b[\w\-.]+\.(?:png|csv|json|pkl|md)\b", report_text))
    referenced = {r for r in referenced if r != "final_report.md"}
    on_disk = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    missing = sorted(r for r in referenced if r not in on_disk)
    if missing:
        findings.append(Finding(
            SEVERITY_CRITICAL, "report_references_missing_artifacts",
            f"The final report references {len(missing)} artifact(s) that do not exist in the run output: {', '.join(missing[:10])}.",
            missing[:10],
        ))
    return findings


# ------------------------------------------------------------------- entry point

def run_consistency_checks(
    output_dir: str,
    report_text: Optional[str] = None,
    uploaded_files: Optional[List[str]] = None,
    persist: bool = True,
) -> ConsistencyReport:
    """
    Runs all cross-artifact consistency checks for a run directory and returns
    (and by default persists) a ConsistencyReport.
    """
    profile = build_dataset_profile(output_dir, uploaded_files, persist=False)
    findings: List[Finding] = []

    checks = (
        ("dataset_shape_claims", lambda: check_dataset_shape_claims(output_dir, profile)),
        ("phantom_columns", lambda: check_phantom_columns(output_dir, profile)),
        ("missing_values_consistency", lambda: check_missing_values_consistency(output_dir, profile)),
        ("model_artifact_consistency", lambda: check_model_artifact_consistency(output_dir)),
        ("duplicate_summaries", lambda: check_duplicate_summaries(output_dir)),
        ("report_artifact_references", lambda: check_report_artifact_references(output_dir, report_text or "")),
    )

    span_ctx = logfire.span(
        "ConsistencyChecker: Run Cross-Artifact Checks", output_dir=output_dir, num_checks=len(checks)
    ) if logfire else contextlib.nullcontext()

    with span_ctx:
        for check_name, check in checks:
            try:
                check_findings = check()
                findings.extend(check_findings)
                if logfire:
                    logfire.info(
                        f"ConsistencyChecker: {check_name} finished",
                        check=check_name,
                        findings_count=len(check_findings),
                        critical_count=sum(1 for f in check_findings if f.severity == SEVERITY_CRITICAL),
                        codes=[f.code for f in check_findings],
                    )
            except Exception as e:
                findings.append(Finding(
                    SEVERITY_MINOR, "check_error",
                    f"Consistency check raised an error and was skipped: {e}",
                    [],
                ))
                if logfire:
                    logfire.warn(f"ConsistencyChecker: {check_name} raised an error; check skipped",
                                 check=check_name, error=str(e))

    report = ConsistencyReport(
        output_dir=output_dir,
        checked_at=datetime.now().isoformat(timespec="seconds"),
        findings=findings,
    )
    if logfire:
        logfire.info(
            "ConsistencyChecker: consistency report complete",
            output_dir=output_dir,
            is_valid=report.is_valid,
            **report.counts(),
        )
    if persist:
        saved_path = report.save()
        if logfire:
            if saved_path:
                logfire.info("ConsistencyChecker: report persisted", path=saved_path)
            else:
                logfire.warn("ConsistencyChecker: could not persist consistency report", output_dir=output_dir)
    return report


def count_by_severity(report: ConsistencyReport) -> Dict[str, int]:
    return report.counts()


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    rep = run_consistency_checks(target, persist=False)
    print(json.dumps(rep.to_dict(), indent=2, default=str))
