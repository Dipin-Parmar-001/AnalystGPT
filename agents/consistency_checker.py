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
- Fabricated segment statistics: report sentences that mention a real dataset
  category together with a rate that contradicts the recomputed ground truth
  (dataset-driven — works for any uploaded dataset's vocabulary)
- Drift between the machine-computed `stats_findings.json` and a fresh
  recomputation from the dataset artifacts

Every finding carries a severity ("critical" or "minor") so the orchestrator can
apply the severity-based failure policy.
"""

import os
import re
import json
import glob
import hashlib
import contextlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from agents.run_context import build_dataset_profile
from agents.logger import setup_logfire
from agents.preflight_gate import _match_artifact, _find_matching_artifact

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

    Match-any semantics for BOTH initial and processed shapes: a claim is only a
    hallucination when it matches NO dataset artifact on disk (raw upload,
    cleaned, or FE). 'initial' is task-relative — the raw upload for the cleaning
    task but cleaned/FE for downstream tasks — so a strict raw comparison would
    false-positive on legitimate downstream input shapes.
    """
    findings: List[Finding] = []
    datasets = profile.get("datasets", {})
    ds_entries = [ds for ds in datasets.values() if ds and not ds.get("error")]

    def _matches_any(claimed: Tuple[Optional[int], Optional[int]], target_file: Optional[str] = None) -> bool:
        c_rows, c_cols = claimed
        candidates = ds_entries
        if target_file:
            target = os.path.basename(str(target_file)).lower()
            derived = next((ds for name, ds in (profile.get("derived_artifacts") or {}).items()
                            if str(name).lower() == target), None)
            named = next((ds for ds in ds_entries if str(ds.get("file", "")).lower() == target), None)
            candidates = [ds for ds in ([named] if named else [derived]) if ds and not ds.get("error")]
        for ds in candidates:
            r_ok = c_rows is None or ds.get("rows") is None or c_rows == ds["rows"]
            c_ok = c_cols is None or ds.get("cols") is None or c_cols == ds["cols"]
            if r_ok and c_ok and (c_rows is not None or c_cols is not None):
                return True
        return False

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
            # Bind explicitly named output shapes to the artifact registered
            # under the same output role. This prevents a coincidentally equal
            # shape in an unrelated CSV from validating the claim.
            target_file = None
            shape_key = jpath.rsplit(".", 1)[-1]
            output_files = summary.get("output_files") or {}
            if shape_key.endswith("_output_shape") and isinstance(output_files, dict):
                role = shape_key[:-len("_output_shape")]
                target_file = output_files.get(role)
            if shape_key == "output_shape":
                target_file = summary.get("output_file") or target_file
            # 'initial' is task-relative: the raw upload for the cleaning task but
            # cleaned/FE for downstream tasks — so both initial and processed shape
            # claims use match-any semantics. A claim is only a hallucination when
            # it matches NO dataset artifact on disk (uploads included).
            is_initial = ".initial_shape" in jpath or jpath.startswith("initial_shape")
            if not _matches_any(shape, target_file):
                actual_desc = "; ".join(
                    f"{ds['file']}={ds['rows']}x{ds['cols']}" for ds in ds_entries if ds.get("rows")
                ) or "no dataset artifacts profiled"
                label = "initial shape" if is_initial else "shape"
                findings.append(Finding(
                    SEVERITY_CRITICAL, "shape_mismatch",
                    f"{src}: claimed {label} ({jpath}) {shape[0] or '?'} rows x {shape[1] or '?'} columns matches NO dataset artifact on disk ({actual_desc}).",
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

    # Reconcile row counts for every explicitly linked input/output pair. A
    # grain change is acceptable only when its metadata explains the operation
    # and the output is unique at the declared grouping key(s).
    artifact_entries = {
        str(ds.get("file", "")).lower(): ds
        for ds in list(datasets.values()) + list((profile.get("derived_artifacts") or {}).values())
        if isinstance(ds, dict) and ds.get("file") and not ds.get("error")
    }

    def _summary_outputs(summary: Dict[str, Any]):
        outputs = []
        seen = set()
        output_file = summary.get("output_file")
        if output_file:
            outputs.append((None, output_file))
            seen.add(os.path.basename(str(output_file)).lower())
        output_files = summary.get("output_files")
        if isinstance(output_files, dict):
            for role, filename in output_files.items():
                key = os.path.basename(str(filename)).lower() if filename else ""
                if filename and key not in seen:
                    outputs.append((str(role), filename))
                    seen.add(key)
        return outputs

    for summary_path in sorted(glob.glob(os.path.join(output_dir, "*summary*.json"))):
        if os.path.basename(summary_path) == "dataset_profile.json":
            continue
        summary = _load_json_safe(summary_path)
        if not isinstance(summary, dict):
            continue
        input_file = summary.get("source_file") or summary.get("input_file")
        if not input_file:
            continue
        source = artifact_entries.get(os.path.basename(str(input_file)).lower())
        if not source:
            continue

        row_grain = summary.get("row_grain") if isinstance(summary.get("row_grain"), dict) else {}
        input_grain = summary.get("input_grain") or row_grain.get("input")
        row_change = summary.get("row_count_change") if isinstance(summary.get("row_count_change"), dict) else {}
        row_changes = summary.get("row_count_changes") if isinstance(summary.get("row_count_changes"), dict) else {}
        row_operations_by_role = summary.get("row_operations_by_role")
        row_operations_by_role = row_operations_by_role if isinstance(row_operations_by_role, dict) else {}
        shared_row_ops = summary.get("row_operations") or summary.get("row_reconciliation")
        output_grains = summary.get("output_grains") if isinstance(summary.get("output_grains"), dict) else {}

        for role, output_file in _summary_outputs(summary):
            target = artifact_entries.get(os.path.basename(str(output_file)).lower())
            if not target:
                continue
            delta = int(target.get("rows", 0)) - int(source.get("rows", 0))
            if not delta:
                continue

            change = row_changes.get(role) if role else None
            change = change if isinstance(change, dict) else row_change if role is None else {}
            row_ops = row_operations_by_role.get(role) if role else shared_row_ops
            output_grain = (output_grains.get(role) if role else None) or summary.get("output_grain") or row_grain.get("output")
            grouping_keys = change.get("grouping_keys")
            transition_ok = bool(
                input_grain and output_grain and change.get("reason")
                and change.get("input_rows") == source.get("rows")
                and change.get("output_rows") == target.get("rows")
                and str(input_grain) == str(source.get("row_grain", "undeclared"))
                and str(output_grain) == str(target.get("row_grain", "undeclared"))
                and isinstance(grouping_keys, list) and grouping_keys
            )
            if transition_ok:
                try:
                    import pandas as pd
                    frame = pd.read_csv(os.path.join(output_dir, target["file"]))
                    transition_ok = all(key in frame.columns for key in grouping_keys)
                    if transition_ok:
                        transition_ok = not frame.duplicated(subset=grouping_keys).any()
                except Exception:
                    transition_ok = False

            operations_ok = False
            if isinstance(row_ops, dict):
                input_rows = row_ops.get("initial_rows", row_ops.get("input_rows"))
                output_rows = row_ops.get("retained", row_ops.get("output_rows"))
                removed = row_ops.get("removed") or []
                added = row_ops.get("added") or []
                if isinstance(removed, list) and isinstance(added, list):
                    try:
                        removed_count = sum(int(item.get("count", 0)) for item in removed if isinstance(item, dict))
                        added_count = sum(int(item.get("count", 0)) for item in added if isinstance(item, dict))
                        complete = all(isinstance(item, dict) and item.get("reason") and "count" in item
                                       for item in removed + added)
                        operations_ok = bool(
                            input_rows == source.get("rows") and output_rows == target.get("rows")
                            and complete and input_rows - removed_count + added_count == output_rows
                        )
                    except (TypeError, ValueError):
                        operations_ok = False

            if not transition_ok and not operations_ok:
                findings.append(Finding(
                    SEVERITY_CRITICAL, "artifact_row_count_change_unreconciled",
                    f"{os.path.basename(str(output_file))} changed rows from {source.get('rows')} to {target.get('rows')} ({delta:+d}) without a verified row operation or declared grain transition with matching counts and unique grouping keys.",
                    [os.path.basename(str(input_file)), os.path.basename(str(output_file)), os.path.basename(summary_path)],
                ))

    return findings


def _canonical_param(value: Any) -> Any:
    """Make JSON numeric strings and equivalent int/float parameters comparable."""
    if isinstance(value, str):
        try:
            number = float(value.strip())
            return int(number) if number.is_integer() else number
        except (ValueError, AttributeError):
            return value.strip().lower()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def check_ml_result_consistency(output_dir: str, report_text: str = "") -> List[Finding]:
    """Cross-check clustering counts and tuning parameters against their artifacts."""
    findings: List[Finding] = []
    tuning = _load_json_safe(os.path.join(output_dir, "hyperparameter_tuning_summary.json"))
    best = _load_json_safe(os.path.join(output_dir, "best_params.json"))
    if isinstance(tuning, dict) and isinstance(best, dict):
        expected = tuning.get("best_params") or (tuning.get("challenger_model") or {})
        if isinstance(expected, dict) and "best_parameters" in expected and isinstance(expected["best_parameters"], dict):
            expected = expected["best_parameters"]
        actual = best.get("best_parameters") or best.get("best_params") or {}
        if isinstance(expected, dict) and isinstance(actual, dict):
            keys = set(expected) | set(actual)
            mismatches = [k for k in sorted(keys) if k not in expected or k not in actual or
                          _canonical_param(expected[k]) != _canonical_param(actual[k])]
            if mismatches:
                findings.append(Finding(
                    SEVERITY_CRITICAL, "tuning_parameter_disagreement",
                    "best_params.json and hyperparameter_tuning_summary.json disagree on "
                    + ", ".join(f"{k} ({expected.get(k, 'missing')} vs {actual.get(k, 'missing')})" for k in mismatches)
                    + ". Use the executed search result and model metadata as ground truth.",
                    ["best_params.json", "hyperparameter_tuning_summary.json"],
                ))

    # The tuned comparison table is another execution artifact; its parameter
    # text and challenger row must agree with the structured search result.
    tuned_csv = os.path.join(output_dir, "tuned_model_comparison.csv")
    if isinstance(tuning, dict) and os.path.isfile(tuned_csv):
        try:
            import pandas as pd
            table = pd.read_csv(tuned_csv)
            challenger = tuning.get("challenger_model") or {}
            expected = {k: _canonical_param(v) for k, v in challenger.items()
                        if k in {"n_clusters", "eps", "min_samples", "init", "n_init", "linkage"}}
            mismatches = []
            for _, row in table.iterrows():
                artifact = str(row.get("artifact", row.get("model_artifact", "")))
                if artifact and artifact != str(challenger.get("artifact", "")):
                    continue
                config = str(row.get("configuration", ""))
                for key, value in expected.items():
                    match = re.search(rf"\b{re.escape(key)}\s*=\s*([^,;]+)", config, re.IGNORECASE)
                    if match and _canonical_param(match.group(1)) != value:
                        mismatches.append(f"{key} ({value} vs {match.group(1).strip()})")
            if mismatches:
                findings.append(Finding(
                    SEVERITY_CRITICAL, "tuned_artifact_parameter_disagreement",
                    "tuned_model_comparison.csv disagrees with the challenger metadata: "
                    + ", ".join(dict.fromkeys(mismatches)) + ".",
                    ["tuned_model_comparison.csv", "hyperparameter_tuning_summary.json"],
                ))
        except Exception as exc:
            findings.append(Finding(SEVERITY_MINOR, "tuned_parameter_check_error",
                                    f"Could not verify tuned comparison parameters: {exc}", ["tuned_model_comparison.csv"]))

    predictions_path = os.path.join(output_dir, "predictions.csv")
    if os.path.isfile(predictions_path):
        try:
            import pandas as pd
            frame = pd.read_csv(predictions_path)
            cluster_col = next((c for c in frame.columns if c.strip().lower() in {"cluster", "cluster_label", "label"}), None)
            if cluster_col:
                counts = frame[cluster_col].value_counts(dropna=False).to_dict()
                actual_counts = {str(k): int(v) for k, v in counts.items()}
                # Validate report statements that explicitly give cluster sizes.
                for label, count in actual_counts.items():
                    pattern = rf"cluster\s*{re.escape(label)}[^\n\r]{{0,100}}?([\d,]+)\s+(?:customers|samples|records)"
                    for match in re.finditer(pattern, report_text, re.IGNORECASE):
                        claimed = int(match.group(1).replace(",", ""))
                        if claimed != count:
                            findings.append(Finding(
                                SEVERITY_CRITICAL, "cluster_count_mismatch",
                                f"Report claims Cluster {label} has {claimed:,} records; predictions.csv contains {count:,}.",
                                ["predictions.csv", "final_report.md"],
                            ))
        except Exception as exc:
            findings.append(Finding(SEVERITY_MINOR, "cluster_count_check_error",
                                    f"Could not verify prediction cluster counts: {exc}", ["predictions.csv"]))
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
    for ds in (profile.get("derived_artifacts") or {}).values():
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
    """Reconciles champion pointer, model registry, and comparison tables with the pkl files on disk.

    Also verifies that the champion artifact can actually be loaded and its metadata matches
    the champion_model.json and model_registry.json entries.
    """
    findings: List[Finding] = []
    pkl_files = {os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "*.pkl"))}

    champion = _load_json_safe(os.path.join(output_dir, "champion_model.json"))
    if isinstance(champion, dict):
        champ_artifact = champion.get("artifact_filename") or champion.get("model")
        if champ_artifact and not _find_matching_artifact(champ_artifact, pkl_files):
            findings.append(Finding(
                SEVERITY_CRITICAL, "stale_champion_pointer",
                f"champion_model.json points to '{champ_artifact}', which does not exist on disk. Available: {', '.join(sorted(pkl_files)) or 'none'}.",
                ["champion_model.json"],
            ))
        from agents.preflight_gate import is_dummy_classifier
        if is_dummy_classifier(champion):
            champ_disp = champion.get("model_name") or champion.get("model") or champ_artifact or "DummyClassifier"
            findings.append(Finding(
                SEVERITY_MINOR, "dummy_classifier_champion",
                f"Dummy classifier model ('{champ_disp}') was designated as champion in champion_model.json. "
                f"The dummy classifier model can be trained and evaluated on the baseline but cannot be deployed or promoted as the champion. "
                f"Please update the related files which are promoting the dummy classifier as champion (e.g. champion_model.json, model_registry.json, ml_summary.json) "
                f"and update artifacts to promote another one best model as champion.",
                ["champion_model.json", "model_registry.json", "ml_summary.json"],
            ))

    registry = _load_json_safe(os.path.join(output_dir, "model_registry.json"))
    registry_champion_artifacts = set()
    if registry is not None:
        entries = registry.get("models") if isinstance(registry, dict) else registry
        if isinstance(entries, list):
            artifact_parameter_map: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                artifact = entry.get("artifact_filename") or entry.get("artifact")
                params = entry.get("model_parameters") or {}
                if artifact and isinstance(params, dict):
                    comparable = {k: _canonical_param(v) for k, v in params.items()
                                  if k in {"n_clusters", "eps", "min_samples", "linkage", "init", "n_init"}
                                  and v is not None}
                    artifact_parameter_map.setdefault(artifact, []).append((str(entry.get("model_id", "?")), comparable))
                if artifact and not _find_matching_artifact(artifact, pkl_files):
                    findings.append(Finding(
                        SEVERITY_CRITICAL, "registry_missing_artifact",
                        f"model_registry.json entry '{entry.get('model_id', '?')}' references '{artifact}', which does not exist on disk.",
                        ["model_registry.json"],
                    ))
                if isinstance(entry.get("status"), str) and entry["status"].lower() == "champion" and artifact:
                    registry_champion_artifacts.add(artifact)

            for artifact, mappings in artifact_parameter_map.items():
                distinct = {json.dumps(params, sort_keys=True, default=str) for _, params in mappings}
                if len(distinct) > 1:
                    findings.append(Finding(
                        SEVERITY_CRITICAL, "artifact_metadata_parameter_collision",
                        f"model_registry.json maps artifact '{artifact}' to conflicting model parameters: "
                        + "; ".join(f"{model_id}={params}" for model_id, params in mappings[:6])
                        + ". Each parameterized model artifact must resolve to the artifact actually trained with those parameters.",
                        ["model_registry.json"],
                    ))

        if isinstance(champion, dict) and registry_champion_artifacts:
            champ_artifact = champion.get("artifact_filename") or champion.get("model")
            if champ_artifact and not any(_match_artifact(champ_artifact, reg_art) for reg_art in registry_champion_artifacts):
                findings.append(Finding(
                    SEVERITY_CRITICAL, "champion_registry_disagreement",
                    f"champion_model.json points to '{champ_artifact}' but model_registry.json marks {sorted(registry_champion_artifacts)} as champion. All artifacts must agree on the champion.",
                    ["champion_model.json", "model_registry.json"],
                ))

            champion_entry = next((e for e in entries if isinstance(e, dict) and
                                   _match_artifact(e.get("artifact_filename") or e.get("artifact"), champ_artifact) and
                                   str(e.get("status", "")).lower() == "champion"), None)
            if champion_entry:
                champ_params = champion_entry.get("model_parameters") or {}
                conflicts = [key for key in ("model_family", "n_clusters", "eps", "linkage")
                             if key in champion and key in (champ_params if key != "model_family" else champion_entry)
                             and _canonical_param(champion[key]) != _canonical_param(
                                 champion_entry.get(key) if key == "model_family" else champ_params.get(key))]
                if conflicts:
                    findings.append(Finding(
                        SEVERITY_CRITICAL, "champion_metadata_disagreement",
                        "champion_model.json and the champion registry entry disagree on "
                        + ", ".join(conflicts) + ".",
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


def _verify_artifact_load(output_dir: str, artifact_name: str, champion: Dict[str, Any], findings: List[Finding]) -> None:
    """Verify the champion artifact can be loaded and its metadata matches expectations."""
    import os
    artifact_path = os.path.join(output_dir, artifact_name)

    try:
        import joblib
        model = joblib.load(artifact_path)

        # Verify the loaded model is a valid sklearn-like estimator
        if not hasattr(model, "predict") and not hasattr(model, "predict_proba"):
            findings.append(Finding(
                SEVERITY_CRITICAL, "artifact_load_verification_failed",
                f"Champion artifact '{artifact_name}' loaded but does not appear to be a valid model (missing predict/predict_proba).",
                ["champion_model.json", artifact_name],
            ))
            return

        # If model has feature_names_in_, verify against expected features
        if hasattr(model, "feature_names_in_"):
            expected_features = champion.get("expected_features")
            if expected_features:
                loaded_features = list(model.feature_names_in_)
                if set(loaded_features) != set(expected_features):
                    findings.append(Finding(
                        SEVERITY_MINOR, "artifact_feature_mismatch",
                        f"Champion artifact '{artifact_name}' has different features than expected. "
                        f"Loaded: {loaded_features}, Expected: {expected_features}",
                        ["champion_model.json", artifact_name],
                    ))

    except Exception as e:
        findings.append(Finding(
            SEVERITY_CRITICAL, "artifact_load_verification_failed",
            f"Champion artifact '{artifact_name}' could not be loaded: {e}",
            ["champion_model.json", artifact_name],
        ))


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


# ------------------------------------------------------ cleaning integrity check

def _load_csv_bounded(path: str):
    """Loads a CSV with pandas, or returns None if unreadable."""
    try:
        import pandas as pd
        try:
            return pd.read_csv(path, nrows=200000)
        except UnicodeDecodeError:
            return pd.read_csv(path, nrows=200000, encoding="latin-1")
    except Exception:
        return None


def check_cleaning_integrity(output_dir: str, profile: Dict[str, Any]) -> List[Finding]:
    """
    Compares shared columns between the raw upload and cleaned_dataset.csv.

    Catches silent data corruption such as a 'Yes'/'No' column coerced with
    pd.to_numeric(errors='coerce') and then filled with 0: every value becomes
    the same constant and the column's signal is destroyed. A shared column
    whose unique-value count collapses (>= 2 raw values -> 1 cleaned value) is
    flagged critical; a > 50% cardinality drop is flagged minor for review.
    """
    findings: List[Finding] = []
    raw_entry = profile.get("datasets", {}).get("raw")
    cleaned_entry = profile.get("datasets", {}).get("cleaned")
    if not cleaned_entry or cleaned_entry.get("error"):
        return findings
    cleaned_path = os.path.join(output_dir, cleaned_entry.get("file", ""))
    if not os.path.exists(cleaned_path):
        cleaned_path = cleaned_entry.get("file", "")
    cleaned_df = _load_csv_bounded(cleaned_path)
    if cleaned_df is None:
        return findings

    # A declared datetime conversion must leave a usable datetime-like column,
    # not a wholly null float column (the observed InvoiceDate regression).
    for summary_name in ("cleaning_summary.json", "analyst_summary.json"):
        summary = _load_json_safe(os.path.join(output_dir, summary_name))
        conversions = summary.get("dtype_conversions", []) if isinstance(summary, dict) else []
        for item in conversions if isinstance(conversions, list) else []:
            if not isinstance(item, dict) or "datetime" not in str(item.get("dtype_after", "")).lower():
                continue
            col = item.get("column")
            if col in cleaned_df.columns:
                series = cleaned_df[col]
                if float(series.isna().mean()) >= 0.99:
                    findings.append(Finding(
                        SEVERITY_CRITICAL, "critical_datetime_column_lost",
                        f"{summary_name} says {col} was converted to {item.get('dtype_after')}, but the cleaned dataset has {series.isna().mean():.1%} null values in that column.",
                        [summary_name, cleaned_entry.get("file", "cleaned_dataset.csv")],
                    ))
    if not raw_entry:
        return findings
    raw_path = raw_entry.get("path") or os.path.join(output_dir, raw_entry.get("file", ""))
    if not os.path.exists(raw_path):
        raw_path = raw_entry.get("file", "")
    raw_df = _load_csv_bounded(raw_path)
    if raw_df is None:
        return findings

    shared = [c for c in raw_df.columns if c in cleaned_df.columns]
    collapsed: List[str] = []
    dropped: List[str] = []
    null_regressions: List[str] = []
    for col in shared:
        try:
            before_null = float(raw_df[col].isna().mean())
            after_null = float(cleaned_df[col].isna().mean())
            if before_null < 0.10 and after_null >= 0.99:
                null_regressions.append(f"'{col}' ({before_null:.1%} -> {after_null:.1%})")
        except Exception:
            pass
        try:
            raw_n = int(raw_df[col].nunique(dropna=True))
            clean_n = int(cleaned_df[col].nunique(dropna=True))
        except Exception:
            continue
        if raw_n >= 2 and clean_n == 1:
            collapsed.append(f"'{col}' ({raw_n} values -> 1)")
        elif raw_n >= 4 and clean_n >= 1 and clean_n <= raw_n / 2:
            dropped.append(f"'{col}' ({raw_n} values -> {clean_n})")

    if collapsed:
        findings.append(Finding(
            SEVERITY_CRITICAL, "column_value_collapse",
            "Cleaned dataset destroyed the variance of shared column(s): "
            + "; ".join(collapsed)
            + ". This usually means a non-numeric text column was coerced with "
            "pd.to_numeric(errors='coerce') and then filled with a constant. Map binary text "
            "columns explicitly (e.g. .map({'No': 0, 'Yes': 1})) and never impute a fully-coerced "
            "column with a constant.",
            [cleaned_entry.get("file", "cleaned_dataset.csv")],
        ))
    if dropped:
        findings.append(Finding(
            SEVERITY_MINOR, "column_cardinality_drop",
            "Shared column(s) lost more than half their unique values during cleaning: "
            + "; ".join(dropped)
            + ". If this is an intentional merge/binning, document it in cleaning_summary.json.",
            [cleaned_entry.get("file", "cleaned_dataset.csv")],
        ))
    if null_regressions:
        findings.append(Finding(
            SEVERITY_CRITICAL, "critical_column_null_rate_regression",
            "Critical shared column(s) became almost entirely null during cleaning (before -> after): "
            + "; ".join(null_regressions[:8]) + ". Preserve source values and validate dtype/null-rate changes after reloading the saved CSV.",
            [raw_entry.get("file", "source dataset"), cleaned_entry.get("file", "cleaned_dataset.csv")],
        ))
    return findings


# ------------------------------------------------------- threshold sanity check

def check_threshold_analysis(output_dir: str, champion_artifact: Optional[str] = None) -> List[Finding]:
    """
    Validates the recommended operating point in threshold_analysis.json.

    A recommendation whose recall is ~1.0 only because precision collapsed to
    the class base rate flags (nearly) every customer and has no targeting
    value; when the stored curve contains a better operating point (recall
    >= 0.90 with materially higher precision), that is flagged critical.
    Also cross-checks the claimed recommended metrics against the curve.
    Verifies the threshold analysis belongs to the current champion model.
    """
    findings: List[Finding] = []
    ta = _load_json_safe(os.path.join(output_dir, "threshold_analysis.json"))
    if not isinstance(ta, dict) or ta.get("applicable") is False or ("recommended_threshold" not in ta and ta.get("applicable") is not True):
        return findings

    # Verify the threshold analysis belongs to the champion model
    if champion_artifact:
        ta_artifact = ta.get("model_artifact") or ta.get("model") or ta.get("artifact_filename")
        if ta_artifact and ta_artifact != champion_artifact:
            findings.append(Finding(
                SEVERITY_CRITICAL, "threshold_analysis_model_mismatch",
                f"threshold_analysis.json was generated for model '{ta_artifact}' but the current "
                f"champion is '{champion_artifact}'. The threshold recommendation does not apply to "
                f"the champion model and must be regenerated.",
                ["threshold_analysis.json", "champion_model.json"],
            ))
            # Don't return early - still validate the curve for completeness

    rec_metrics = ta.get("recommended_metrics") or {}
    rec_recall = rec_metrics.get("recall")
    rec_precision = rec_metrics.get("precision")
    curve = ta.get("precision_recall_curve") or {}
    precisions = curve.get("precision") or []
    recalls = curve.get("recall") or []

    base_rate = None
    if precisions and recalls and len(precisions) == len(recalls):
        try:
            max_recall_idx = max(range(len(recalls)), key=lambda i: recalls[i])
            base_rate = float(precisions[max_recall_idx])
        except Exception:
            base_rate = None

    def _better_operating_point(min_recall: float = 0.90) -> Optional[Tuple[float, float, float]]:
        """Returns (threshold, precision, recall) maximizing precision s.t. recall >= min_recall."""
        thresholds = curve.get("thresholds") or []
        best = None
        for i, r in enumerate(recalls):
            try:
                r_f = float(r)
                p_f = float(precisions[i])
            except Exception:
                continue
            if r_f >= min_recall:
                t_f = float(thresholds[i]) if i < len(thresholds) else 0.0
                if best is None or p_f > best[1]:
                    best = (t_f, p_f, r_f)
        return best

    is_degenerate = (
        rec_recall is not None
        and rec_precision is not None
        and float(rec_recall) >= 0.999
        and base_rate is not None
        and float(rec_precision) <= base_rate * 1.02 + 1e-9
    )
    if is_degenerate:
        alt = _better_operating_point()
        alt_txt = ""
        if alt:
            alt_txt = (
                f" A better operating point exists in the stored curve: threshold ~{alt[0]:.3f} "
                f"(precision={alt[1]:.3f}, recall={alt[2]:.3f})."
            )
        findings.append(Finding(
            SEVERITY_CRITICAL, "degenerate_threshold_recommendation",
            f"threshold_analysis.json recommends a degenerate operating point: recall="
            f"{rec_recall} with precision={rec_precision}, which equals the class base rate — "
            f"i.e. predict positive for every customer. That has no targeting value."
            f"{alt_txt} Recommend max precision subject to a recall constraint (or max F-beta) "
            f"instead, and state the flagged-customer volume.",
            ["threshold_analysis.json"],
        ))

    # Cross-check claimed recommended metrics against the stored curve.
    if precisions and recalls and rec_recall is not None and rec_precision is not None:
        try:
            tol = 0.02
            match_found = any(
                abs(float(precisions[i]) - float(rec_precision)) <= tol
                and abs(float(recalls[i]) - float(rec_recall)) <= tol
                for i in range(min(len(precisions), len(recalls)))
            )
            if not match_found:
                findings.append(Finding(
                    SEVERITY_MINOR, "threshold_metrics_mismatch",
                    "threshold_analysis.json recommended_metrics (precision="
                    f"{rec_precision}, recall={rec_recall}) do not match any point on its own "
                    "stored precision_recall_curve.",
                    ["threshold_analysis.json"],
                ))
        except Exception:
            pass
    return findings


# --------------------------------------------- visualization description check

def _normalize_label(text: str) -> str:
    """Normalizes a category label for fuzzy comparison."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def check_visualization_descriptions(output_dir: str) -> List[Finding]:
    """
    Cross-checks ordinal-mapping claims in chart descriptions (e.g.
    '0=Month-to-month, 2=Two year') against the canonical mappings declared in
    feature_engineering_summary.json. Reversed mappings mislead anyone reading
    the summary JSON without the chart.
    """
    import re

    findings: List[Finding] = []
    fe = _load_json_safe(os.path.join(output_dir, "feature_engineering_summary.json"))
    if not isinstance(fe, dict):
        return findings

    # Canonical mappings from engineered-feature formulas: normalized label -> (raw label, value).
    canonical: Dict[str, Tuple[str, int]] = {}
    for feat in fe.get("engineered_features", []) or []:
        if not isinstance(feat, dict):
            continue
        formula = str(feat.get("formula", ""))
        if "=" not in formula or not re.search(r"=\s*\d+", formula):
            continue
        for match in re.finditer(r"([A-Za-z][\w\s\-()']*?)\s*=\s*(-?\d+)", formula):
            label_raw = match.group(1).strip().rstrip(",")
            try:
                value = int(match.group(2))
            except ValueError:
                continue
            if label_raw:
                canonical[_normalize_label(label_raw)] = (label_raw, value)
    if not canonical:
        return findings

    for path in sorted(glob.glob(os.path.join(output_dir, "analyst_summary*.json"))):
        summary = _load_json_safe(path)
        if not isinstance(summary, dict):
            continue
        src = os.path.basename(path)
        for viz in summary.get("generated_visualizations", []) or []:
            if not isinstance(viz, dict):
                continue
            description = str(viz.get("description", ""))
            # Claims like "0=Low/Month-to-month" or "2 = High/Two year".
            for match in re.finditer(r"(\d+)\s*=\s*([A-Za-z][\w\s/\-()']*)", description):
                try:
                    claimed_value = int(match.group(1))
                except ValueError:
                    continue
                for part in re.split(r"[/,]", match.group(2)):
                    norm = _normalize_label(part)
                    if len(norm) < 3:
                        continue
                    hit = canonical.get(norm)
                    if hit and hit[1] != claimed_value:
                        findings.append(Finding(
                            SEVERITY_MINOR, "viz_description_mapping_mismatch",
                            f"{src} describes a chart with mapping '{claimed_value}={part.strip()}' "
                            f"for an engineered feature, but feature_engineering_summary.json "
                            f"defines '{hit[0]}={hit[1]}'. Fix the description (or the chart) so "
                            f"the stated mapping matches the actual encoding.",
                            [src, "feature_engineering_summary.json"],
                        ))
    return findings


# ------------------------------------------------------------ segment claims

_SEGMENT_SKIP_CATEGORIES = {"0", "1", "<reference>", "no", "yes"}
_RATE_MATCH_TOL_PCT = 0.5  # percentage points; covers 1-decimal rounding of true rates

# Sentences describing model PREDICTION behavior rather than dataset segments.
# Percentages quoted here are holdout prediction rates (flag rate, positive-class
# share, contact volume), which must not be judged against segment churn rates.
_PREDICTION_SENTENCE_RE = re.compile(
    r"flag(?:ged| rate|s)?\b|positive[- ]class|predicted (?:rate|share|volume|count|positive)"
    r"|\bpredictions?\b|test (?:set|predictions)|holdout|contact(?:s|ed)? volume",
    re.IGNORECASE,
)


def _binary_feature_labels(feature: str) -> List[str]:
    """Normalized index labels for a binary feature so prose like 'customers
    without partners' or 'no dependents' resolves to the feature's segments."""
    import re as _re

    feat = _re.sub(r"[\s\-\u2013\u2014_]+", " ", (feature or "").strip()).lower()
    if not feat:
        return []
    labels = [feat, feat + "s"]
    # Common singular/plural variants for features already ending in 's'
    # ('dependents' -> 'dependent') so singular prose still matches.
    if feat.endswith("ies"):
        labels.append(feat[:-3] + "y")
    elif feat.endswith("s"):
        labels.append(feat[:-1])
    return labels


def _load_segment_frame(output_dir: str):
    """Loads the dataset artifact used for segment grounding (FE preferred over cleaned)."""
    try:
        import pandas as pd
    except ImportError:
        return None, None
    for name in ("feature_engineered_dataset.csv", "cleaned_dataset.csv", "cleaned_data.csv"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                return pd.read_csv(path), name
            except Exception:
                continue
    return None, None


def _recompute_segment_stats(output_dir: str, uploaded_files: Optional[List[str]] = None, dataset_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Recomputes ground-truth segment statistics from the run's dataset artifacts.

    Falls back to a name-based segment recomputation from the dataset artifacts when no
    raw uploads are available; raw uploads are merged in so one-hot reference categories
    keep their real names.
    """
    from pathlib import Path
    from sandbox_execution.verification_script import compute_segment_stats, declared_target_column

    if not dataset_type:
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest_data = _load_json_safe(manifest_path)
                if isinstance(manifest_data, dict):
                    dataset_type = manifest_data.get("dataset_type")
            except Exception:
                pass
    if dataset_type and str(dataset_type).lower() in ("regression", "unsupervised"):
        return None

    df, source = _load_segment_frame(output_dir)
    if df is None:
        return None
    try:
        declared = declared_target_column(Path(output_dir))
    except Exception:
        declared = None
    try:
        stats = compute_segment_stats(df, declared_target=declared, dataset_type=dataset_type)
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    stats["source_file"] = source

    # Merge named segments from raw uploads (see build_verified_metrics).
    import pandas as _pd
    for up in uploaded_files or []:
        try:
            if not up or not str(up).lower().endswith(".csv") or not os.path.exists(up):
                continue
            raw_df = _pd.read_csv(up, nrows=500_000)
            if declared not in raw_df.columns:
                continue
            raw_stats = compute_segment_stats(raw_df, declared_target=declared)
            if not raw_stats:
                continue
            by_key = {
                (str(s["feature"]).lower(), str(s["category"]).lower()): s
                for s in stats["rate_by_category"]
            }
            for s in raw_stats.get("rate_by_category", []):
                by_key[(str(s["feature"]).lower(), str(s["category"]).lower())] = s
            stats["rate_by_category"] = list(by_key.values())
            stats["raw_source_file"] = os.path.basename(up)
        except Exception:
            continue
    return stats


def _norm_segment_label(s: str) -> str:
    """Collapses dashes/underscores/whitespace so 'two-year', 'Two year' and '49–72' align."""
    import re as _re
    return _re.sub(r"[\s\-\u2013\u2014_]+", " ", (s or "").strip()).lower()


def check_segment_claims(output_dir: str, report_text: str, uploaded_files: Optional[List[str]] = None, dataset_type: Optional[str] = None) -> List[Finding]:
    """
    Dataset-driven guard against fabricated segment statistics.

    For every category that actually exists in the run's dataset artifacts (any
    categorical level, one-hot dummy level, or tenure-style bucket — whatever the
    uploaded data contains), scans the report for sentences that mention that
    category together with a percentage. If the quoted percentage matches none of
    the rates of the categories named in that same sentence, it is fabricated
    -> critical.

    Semantic rules (to avoid false positives on legitimately derived numbers):

    - Binary yes/no segments (one-hot reference categories) are indexed under
      their FEATURE name as well (e.g. 'partner', 'dependents'), with both the
      yes and no rates as legitimate referents. A sentence like "customers
      without partners (32.96%)" therefore checks against Partner=No's real
      rate instead of being judged against unrelated co-mentioned segments.
    - Sentences describing model PREDICTION behavior (flag rate, positive-class
      share of predictions, test-set volume) are excluded: quoted percentages
      there are model outputs on the holdout, not dataset segment churn rates.
      Value grounding for those numbers remains the numeric verifier's job.

    Matching is deliberately scoped to the categories mentioned in the sentence:
    a fabricated rate must not be excused by colliding with some *other*
    segment's true rate (pool saturation). Fully generic — the vocabulary checked
    comes from the dataset itself, not from any domain-specific word list.
    """
    import re as _re
    from pathlib import Path as _Path

    findings: List[Finding] = []
    if not report_text:
        return findings
    stats = _recompute_segment_stats(output_dir, uploaded_files, dataset_type=dataset_type)
    if not stats:
        return findings

    # Index named segments by normalized category label. Binary yes/no levels are
    # additionally indexed under their feature name so prose like "customers
    # without partners" resolves to the Partner segments.
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for seg in stats.get("rate_by_category", []):
        category = str(seg.get("category", "")).strip()
        try:
            float(seg["rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if not category:
            continue
        # Self-index only distinctive category labels; bare yes/no levels are
        # reachable via their feature-name labels below.
        if len(category) >= 3 and category.lower() not in _SEGMENT_SKIP_CATEGORIES:
            by_label.setdefault(_norm_segment_label(category), []).append(seg)
        if str(seg.get("feature", "")).strip().lower() in {"churn", "churn label", "churn value", "target"}:
            continue
        if category.lower() in {"no", "yes"}:
            for label in _binary_feature_labels(str(seg.get("feature", ""))):
                by_label.setdefault(label, []).append(seg)

    # Ground-truth percentages already verified elsewhere (VERIFIED_METRICS):
    # exempting them keeps this check focused on segment attribution.

    def _norm(s: str) -> str:
        return _norm_segment_label(s)

    text = _re.sub(r"```.*?```", " ", report_text or "", flags=_re.DOTALL)
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
    pct_re = _re.compile(r"(\d+(?:\.\d+)?)\s*%")
    seen: set = set()

    # Label-boundary matcher: alphanumerics around the label defeat the match
    # (so 'DSL' is not found inside 'SDSL'), while punctuation like '(' next to
    # a bucket label ('(0-12') still matches.
    def _label_in(low: str, label: str) -> bool:
        # Trailing 's?' (positive lookahead) lets 'partner' match the plural prose
        # 'partners' while still requiring a word boundary after it.
        return _re.search(rf"(?<![0-9a-z]){ _re.escape(label) }(?=s?(?![0-9a-z]))", low) is not None

    for sentence in sentences:
        low = _norm(sentence)
        # Skip verification-notice blockquotes: runs whose report embedded prior
        # gate output quote finding text (with segment rates) verbatim, which the
        # check must not re-adjudicate as new report claims.
        if low.startswith(">") and ("verification notice" in low or "report quotes" in low):
            continue
        # Model-prediction sentences quote prediction-side rates (flag rate,
        # positive-class share), not dataset segment rates. Judging those against
        # segment ground truth produces false positives.
        if _PREDICTION_SENTENCE_RE.search(low):
            continue
        mentioned: List[Tuple[str, List[Dict[str, Any]]]] = [
            (label, segs) for label, segs in by_label.items() if _label_in(low, label)
        ]
        if not mentioned:
            continue
        mentioned_rates = [float(seg["rate"]) for _, segs in mentioned for seg in segs]
        # The overall target rate is a legitimate referent on a category line.
        try:
            mentioned_rates.append(float(stats["overall"]["rate"]))
        except (KeyError, TypeError, ValueError):
            pass
        for m in pct_re.finditer(sentence):
            claimed = float(m.group(1))
            if any(abs(claimed / 100.0 - r) * 100 <= _RATE_MATCH_TOL_PCT for r in mentioned_rates):
                continue
            key = (tuple(sorted(l for l, _ in mentioned)), claimed)
            if key in seen:
                continue
            seen.add(key)
            refs = "; ".join(
                f"'{segs[0]['feature']}'='{segs[0]['category']}'={float(segs[0]['rate']) * 100:.1f}% "
                f"({segs[0]['positives']}/{segs[0]['count']})"
                for _, segs in mentioned[:3]
            )
            findings.append(Finding(
                SEVERITY_CRITICAL, "segment_rate_mismatch",
                f"Report quotes {claimed:g}% in a sentence about segments that actually have: {refs}. "
                f"Recompute from the dataset artifacts and correct the claim.",
                ["final_report.md", stats.get("source_file") or "dataset"],
            ))
            if len(findings) >= 10:
                return findings
    return findings


def check_stats_findings_consistency(output_dir: str, uploaded_files: Optional[List[str]] = None, dataset_type: Optional[str] = None) -> List[Finding]:
    """
    Diffs the machine-computed `stats_findings.json` (which the analyst's sandbox
    script is required to emit) against a fresh recomputation from the dataset
    artifacts. Any drift proves a summary number did not come from the executing
    code. No-op for runs that predate the stats_findings.json contract.
    """
    findings: List[Finding] = []
    sf = _load_json_safe(os.path.join(output_dir, "stats_findings.json"))

    if not dataset_type:
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest_data = _load_json_safe(manifest_path)
                if isinstance(manifest_data, dict):
                    dataset_type = manifest_data.get("dataset_type")
            except Exception:
                pass

    if dataset_type and str(dataset_type).lower() in ("regression", "unsupervised"):
        return findings

    # Contract-compliance signal: a stats-and-viz task ran (analyst summary with
    # generated visualizations) but the required stats artifact was never written.
    if not isinstance(sf, dict):
        for summary_path in sorted(glob.glob(os.path.join(output_dir, "analyst_summary*.json"))):
            summary = _load_json_safe(summary_path)
            if isinstance(summary, dict) and summary.get("generated_visualizations"):
                findings.append(Finding(
                    SEVERITY_MINOR, "stats_findings_missing",
                    f"{os.path.basename(summary_path)} reports generated visualizations but required "
                    f"`stats_findings.json` (computed statistics artifact) is absent — segment-rate claims "
                    f"cannot be grounded for this run.",
                    [os.path.basename(summary_path)],
                ))
                break
        return findings

    stats = _recompute_segment_stats(output_dir, uploaded_files, dataset_type=dataset_type)
    if not stats:
        return findings

    recomputed = {}
    for seg in stats.get("rate_by_category", []):
        recomputed[(str(seg.get("feature", "")).lower(), str(seg.get("category", "")).lower())] = seg
    recomputed_overall = stats.get("overall") or {}

    sf_overall = sf.get("overall")
    if isinstance(sf_overall, dict) and recomputed_overall:
        for field in ("count", "positives"):
            sf_v, rc_v = sf_overall.get(field), recomputed_overall.get(field)
            if isinstance(sf_v, (int, float)) and rc_v is not None and int(sf_v) != int(rc_v):
                findings.append(Finding(
                    SEVERITY_CRITICAL, "segment_count_mismatch",
                    f"stats_findings.json overall {field}={sf_v} but the dataset artifacts give {int(rc_v)}. "
                    f"The stats artifact was not produced from the loaded data.",
                    ["stats_findings.json", stats.get("source_file") or "dataset"],
                ))
        sf_rate, rc_rate = sf_overall.get("rate"), recomputed_overall.get("rate")
        if isinstance(sf_rate, (int, float)) and rc_rate is not None and abs(float(sf_rate) - float(rc_rate)) > 0.005:
            findings.append(Finding(
                SEVERITY_CRITICAL, "segment_rate_mismatch",
                f"stats_findings.json overall rate={sf_rate} but the dataset artifacts give {rc_rate}.",
                ["stats_findings.json", stats.get("source_file") or "dataset"],
            ))

    df_cache = None
    for seg in sf.get("rate_by_category", []) or []:
        if not isinstance(seg, dict):
            continue
        key = (str(seg.get("feature", "")).lower(), str(seg.get("category", "")).lower())
        rc = recomputed.get(key)
        if rc is None:
            if df_cache is None:
                df_cache, _ = _load_segment_frame(output_dir)
            feat_name = seg.get("feature")
            cat_name = seg.get("category")
            if df_cache is not None and feat_name in df_cache.columns:
                try:
                    unique_vals = set(df_cache[feat_name].dropna().astype(str).str.lower().unique())
                    if str(cat_name).lower() in unique_vals:
                        continue
                except Exception:
                    pass
            findings.append(Finding(
                SEVERITY_MINOR, "stats_findings_unmatched_segment",
                f"stats_findings.json references segment '{seg.get('feature')}'='{seg.get('category')}' which "
                f"does not exist in the dataset artifacts (check for typos or invented categories).",
                ["stats_findings.json"],
            ))
            continue
        sf_rate, rc_rate = seg.get("rate"), rc.get("rate")
        if isinstance(sf_rate, (int, float)) and rc_rate is not None and abs(float(sf_rate) - float(rc_rate)) > 0.005:
            findings.append(Finding(
                SEVERITY_CRITICAL, "segment_rate_mismatch",
                f"stats_findings.json: rate for '{seg.get('feature')}'='{seg.get('category')}' is {sf_rate} but the "
                f"dataset artifacts give {rc_rate} ({rc.get('positives')}/{rc.get('count')}).",
                ["stats_findings.json", stats.get("source_file") or "dataset"],
            ))
        for field in ("count", "positives"):
            sf_v, rc_v = seg.get(field), rc.get(field)
            if isinstance(sf_v, (int, float)) and rc_v is not None and int(sf_v) != int(rc_v):
                findings.append(Finding(
                    SEVERITY_CRITICAL, "segment_count_mismatch",
                    f"stats_findings.json: {field} for '{seg.get('feature')}'='{seg.get('category')}' is {int(sf_v)} "
                    f"but the dataset artifacts give {int(rc_v)}.",
                    ["stats_findings.json", stats.get("source_file") or "dataset"],
                ))
    return findings


# ------------------------------------------------------------ report references

def check_report_artifact_references(
    output_dir: str,
    report_text: str,
    uploaded_files: Optional[List[str]] = None,
) -> List[Finding]:
    """Flags artifact files referenced in the final report that were never generated."""
    import re
    findings: List[Finding] = []
    if not report_text:
        return findings
    referenced = set(re.findall(r"\b[\w\-.]+\.(?:png|csv|json|pkl|md)\b", report_text))
    referenced = {r for r in referenced if r != "final_report.md"}
    on_disk = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    # A wildcard such as `model_p*.pkl` is prose notation, not a reference to
    # a literal `_p0.pkl` file.
    referenced = {r for r in referenced if not r.startswith("_")}
    # Raw uploads live outside the run dir; referencing them is legitimate.
    for f in uploaded_files or []:
        if f:
            on_disk.add(os.path.basename(f))
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
    dataset_type: Optional[str] = None,
) -> ConsistencyReport:
    """
    Runs all cross-artifact consistency checks for a run directory and returns
    (and by default persists) a ConsistencyReport.
    """
    profile = build_dataset_profile(output_dir, uploaded_files, persist=False)
    findings: List[Finding] = []

    # Load champion artifact for threshold analysis verification
    champion_artifact = None
    champion = _load_json_safe(os.path.join(output_dir, "champion_model.json"))
    if isinstance(champion, dict):
        champion_artifact = champion.get("model") or champion.get("artifact_filename")

    checks = (
        ("dataset_shape_claims", lambda: check_dataset_shape_claims(output_dir, profile)),
        ("phantom_columns", lambda: check_phantom_columns(output_dir, profile)),
        ("missing_values_consistency", lambda: check_missing_values_consistency(output_dir, profile)),
        ("cleaning_integrity", lambda: check_cleaning_integrity(output_dir, profile)),
        ("threshold_analysis", lambda: check_threshold_analysis(output_dir, champion_artifact)),
        ("viz_descriptions", lambda: check_visualization_descriptions(output_dir)),
        ("model_artifact_consistency", lambda: check_model_artifact_consistency(output_dir)),
        ("ml_result_consistency", lambda: check_ml_result_consistency(output_dir, report_text or "")),
        ("duplicate_summaries", lambda: check_duplicate_summaries(output_dir)),
        ("segment_claims", lambda: check_segment_claims(output_dir, report_text or "", uploaded_files, dataset_type=dataset_type)),
        ("stats_findings_consistency", lambda: check_stats_findings_consistency(output_dir, uploaded_files, dataset_type=dataset_type)),
        ("report_artifact_references", lambda: check_report_artifact_references(output_dir, report_text or "", uploaded_files)),
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
