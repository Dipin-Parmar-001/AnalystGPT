"""
Run Context Module - AnalystGPT.

Builds the AUTHORITATIVE DATASET PROFILE: a single ground-truth description of every
dataset artifact in the run output directory (raw upload, cleaned_dataset.csv,
feature_engineered_dataset.csv), computed directly from the files on disk.

Root cause this fixes: agents re-derive dataset shapes/columns from memory after
context compaction and state contradictory counts (e.g. 30 vs 33 vs 44 columns,
50 vs 54 columns, 6 vs 10 engineered features, phantom column names). The profile
is rebuilt from disk before every task dispatch and injected into each agent
prompt as the only admissible source for row/column/column-name claims.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from agents.logger import setup_logfire

try:
    import logfire
except ImportError:
    logfire = None

setup_logfire()

# Dataset artifacts in lifecycle order; later entries supersede earlier ones
# when determining the "current dataset".
DATASET_LIFECYCLE = [
    "cleaned_dataset.csv",
    "cleaned_data.csv",
    "feature_engineered_dataset.csv",
]

_PROFILE_FILENAME = "dataset_profile.json"
MAX_PROFILE_ROWS = 10_000_000


def _profile_one_csv(path: str) -> Optional[Dict[str, Any]]:
    """Computes rows/cols/column list/missing summary for one CSV file."""
    try:
        import pandas as pd

        # Profiles must cover large uploads too; nrows here is a hard cap and
        # previously made every dataset over 100k rows report a false shape.
        df = pd.read_csv(path, nrows=MAX_PROFILE_ROWS)
        rows = int(df.shape[0])
        cols = int(df.shape[1])
        missing_cells = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        return {
            "file": os.path.basename(path),
            "rows": rows,
            "cols": cols,
            "columns": [str(c) for c in df.columns],
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
        }
    except Exception as e:
        return {"file": os.path.basename(path), "error": f"Failed to profile: {e}"}


def _find_raw_dataset(output_dir: str, uploaded_files: Optional[List[str]]) -> Optional[str]:
    """
    Locates the raw uploaded dataset file.

    Only the run's actual upload paths are trusted. The output directory is
    deliberately NOT scanned: every CSV in there is a pipeline-produced artifact
    (cleaned/FE datasets, correlation matrices, comparison tables...), and
    guessing from it has already mis-profiled `corr_matrix.csv` as the raw
    dataset, which cascaded into false shape-mismatch findings.
    """
    pipeline_names = set(DATASET_LIFECYCLE)
    if uploaded_files:
        for f in uploaded_files:
            if not f or os.path.basename(f).lower() in pipeline_names:
                continue
            if os.path.exists(f):
                return f
            # Accept bare filenames that resolve in the workspace (uploads/, outputs/, root).
            try:
                from sandbox_execution.utils import resolve_file_path

                resolved = resolve_file_path(f)
                if resolved and os.path.exists(resolved):
                    return resolved
            except Exception:
                pass
    return None


def build_dataset_profile(
    output_dir: str,
    uploaded_files: Optional[List[str]] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Builds the authoritative dataset profile for the current run state.

    Returns a dict with one entry per dataset artifact present on disk plus the
    derived engineered-feature list and the current-dataset pointer. Safe to call
    repeatedly: it is cheap (header + bounded read) and always reflects disk.
    """
    profile: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": {},
        "engineered_features": [],
        "current_dataset": None,
        "notes": [],
    }

    if logfire:
        logfire.info("RunContext: building dataset profile", output_dir=output_dir, persist=persist)
    raw_path = _find_raw_dataset(output_dir, uploaded_files)
    if raw_path:
        ext = os.path.splitext(raw_path)[1].lower()
        if ext == ".csv":
            raw_profile = _profile_one_csv(raw_path)
            # Sanity guard: a genuine raw upload can never have fewer rows than
            # the cleaned dataset (cleaning only removes rows), so a "raw"
            # profile with fewer rows is a misidentified pipeline artifact
            # (e.g. a correlation matrix). Profile it as a note instead of as
            # ground truth for shape claims.
            _cleaned_probe = None
            for _name in DATASET_LIFECYCLE:
                _p = os.path.join(output_dir, _name)
                if os.path.exists(_p):
                    _cleaned_probe = _profile_one_csv(_p)
                    break
            if (
                _cleaned_probe
                and not raw_profile.get("error")
                and not _cleaned_probe.get("error")
                and raw_profile.get("rows") is not None
                and _cleaned_probe.get("rows") is not None
                and raw_profile["rows"] < _cleaned_probe["rows"]
            ):
                profile["notes"].append(
                    f"Refused to profile '{raw_profile.get('file')}' as the raw dataset: it has "
                    f"fewer rows ({raw_profile.get('rows')}) than '{_cleaned_probe.get('file')}' "
                    f"({_cleaned_probe.get('rows')}), so it looks like a derived pipeline "
                    f"artifact. Raw upload not profiled."
                )
            else:
                raw_profile["path"] = os.path.abspath(raw_path)
                profile["datasets"]["raw"] = raw_profile
        else:
            profile["datasets"]["raw"] = {"file": os.path.basename(raw_path), "note": "Non-CSV raw upload; profiled by agents during inspection."}

    for name in DATASET_LIFECYCLE:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            key = "feature_engineered" if name == "feature_engineered_dataset.csv" else "cleaned"
            profile["datasets"][key] = _profile_one_csv(path)

    # Engineered features = columns present in FE dataset but not in cleaned one.
    fe = profile["datasets"].get("feature_engineered")
    cleaned = profile["datasets"].get("cleaned")
    if fe and cleaned and fe.get("columns") and cleaned.get("columns"):
        cleaned_set = set(cleaned["columns"])
        profile["engineered_features"] = [c for c in fe["columns"] if c not in cleaned_set]

    # Current dataset pointer follows the lifecycle order.
    for key in ["feature_engineered", "cleaned"]:
        ds = profile["datasets"].get(key)
        if ds and ds.get("file"):
            profile["current_dataset"] = ds["file"]
            break
    if profile["current_dataset"] is None and profile["datasets"].get("raw"):
        profile["current_dataset"] = profile["datasets"]["raw"].get("file")

    if cleaned and fe:
        if fe.get("rows") != cleaned.get("rows"):
            profile["notes"].append(
                f"Row count changed between cleaned ({cleaned.get('rows')}) and "
                f"feature-engineered ({fe.get('rows')}) datasets."
            )

    if persist:
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, _PROFILE_FILENAME), "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, default=str)
        except Exception as e:
            if logfire:
                logfire.warn("RunContext: could not persist dataset profile", output_dir=output_dir, error=str(e))
            print(f"[RunContext] Notice: could not persist dataset profile: {e}")

    if logfire:
        ds_summaries = {
            key: {"file": ds.get("file"), "rows": ds.get("rows"), "cols": ds.get("cols"), "error": ds.get("error")}
            for key, ds in profile["datasets"].items() if ds
        }
        logfire.info(
            "RunContext: dataset profile built",
            datasets=ds_summaries,
            num_engineered_features=len(profile["engineered_features"]),
            current_dataset=profile["current_dataset"],
            notes=profile["notes"],
        )

    return profile


def _lf_span(name: str, **attrs):
    """Returns a logfire span when logfire is available, else a no-op context."""
    import contextlib
    if logfire:
        return logfire.span(name, **attrs)
    return contextlib.nullcontext()


def render_dataset_profile_block(profile: Optional[Dict[str, Any]]) -> str:
    """
    Renders the profile as a compact prompt block. This block is the ONLY
    admissible source for dataset shape/column claims in agent outputs.
    """
    if not profile:
        return ""
    try:
        profile_json = json.dumps(profile, indent=2, default=str)
    except Exception:
        return ""

    return (
        "\n=== AUTHORITATIVE DATASET PROFILE (computed directly from files on disk) ===\n"
        f"{profile_json}\n"
        "GROUND RULES for this profile:\n"
        "- These row counts, column counts, and column names are GROUND TRUTH for this run.\n"
        "- Use them VERBATIM in summaries and the report. NEVER re-derive, estimate, or state "
        "dataset shapes/columns that contradict this profile.\n"
        "- 'engineered_features' lists exactly which columns were added by feature engineering. "
        "Never describe a raw/cleaned column as engineered, and never name an engineered column "
        "absent from that list.\n"
        "- 'current_dataset' is the file all downstream tasks must train/analyze on.\n"
        "=====================================================================================\n"
    )


def get_column_set(profile: Optional[Dict[str, Any]], key: str) -> Tuple[str, set]:
    """Returns (filename, set_of_columns) for a profile dataset entry ('raw'/'cleaned'/'feature_engineered')."""
    if not profile:
        return "", set()
    ds = profile.get("datasets", {}).get(key) or {}
    return ds.get("file", ""), set(ds.get("columns") or [])


def compare_profile_claims(
    profile: Optional[Dict[str, Any]],
    claimed_rows: Optional[int] = None,
    claimed_cols: Optional[int] = None,
    dataset_key: Optional[str] = None,
) -> List[str]:
    """
    Cross-checks claimed shape values against the authoritative profile.
    Returns a list of human-readable discrepancy strings (empty if consistent).
    Used by the consistency checker and the numeric verifier gate.
    """
    issues: List[str] = []
    if not profile:
        return issues
    datasets = profile.get("datasets", {})
    candidates = [dataset_key] if dataset_key else list(datasets.keys())
    for key in candidates:
        ds = datasets.get(key)
        if not ds or ds.get("error"):
            continue
        if claimed_rows is not None and ds.get("rows") is not None and int(claimed_rows) != int(ds["rows"]):
            issues.append(
                f"Claimed {claimed_rows} rows for '{ds.get('file')}' ({key}); authoritative profile says {ds['rows']}."
            )
        if claimed_cols is not None and ds.get("cols") is not None and int(claimed_cols) != int(ds["cols"]):
            issues.append(
                f"Claimed {claimed_cols} columns for '{ds.get('file')}' ({key}); authoritative profile says {ds['cols']}."
            )
    return issues


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    p = build_dataset_profile(target, persist=False)
    print(render_dataset_profile_block(p))
