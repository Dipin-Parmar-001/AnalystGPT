"""
Verification Script Module for AnalystGPT Sandbox Execution.

Builds ground-truth verified metrics directly from physical artifacts on disk or in the Modal Cloud Sandbox
(e.g., cleaned_dataset.csv, model_comparison.csv, feature_importance.csv, confusion_matrix.json, baseline_score.json, predictions.csv,
tuned_model_comparison.csv, best_params.json, hyperparameter_tuning_summary.json, model_registry.json, champion_model.json,
and versioned model artifacts like logistic_regression_p0.pkl, random_forest_p0.pkl, xgboost_p0.pkl, logistic_regression_p1_tuned.pkl).

Can be executed as a standalone script inside the sandbox or imported as a Python module.
"""

import os
import sys
import json
import difflib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


def safe_float(val: Any) -> Optional[float]:
    """Safely converts a value to float if possible."""
    try:
        if val is None or (isinstance(val, str) and val.strip().lower() in ["", "nan", "none", "null"]):
            return None
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Generic segment-statistics grounding (dataset-agnostic).
#
# The goal: for ANY uploaded dataset, deterministically compute the per-category
# target rates and per-class numeric summaries that reports typically claim, so
# the verifier can prove or refute claims like "category X churns at Y%" without
# knowing anything about the domain in advance.
#
# Design rules (no domain keyword lists anywhere):
# - The target column is taken from what the pipeline itself declared
#   (`target_column` in the run's summary/registry JSONs); structural fallback:
#   if the frame has exactly ONE binary column, use it, otherwise refuse to
#   guess (no grounding beats wrong grounding).
# - Segments are structural: non-numeric columns with few unique values; 0/1
#   dummy columns grouped by their shared prefix (standard one-hot naming);
#   all-zero rows of a dummy group form the reference category.
# - The positive class is the minority class (generic for churn / fraud /
#   default / attrition style targets).
# - High-cardinality columns (IDs, free text, geo) are excluded structurally by
#   unique-count thresholds, never by name.
# ---------------------------------------------------------------------------

SEGMENT_MAX_CATEGORIES = 12          # skip categoricals wider than this (IDs, free text)
SEGMENT_MIN_CATEGORY_ROWS = 5        # skip tiny segments (noise / near-unique values)
SEGMENT_MAX_NUMERIC_BY_CLASS = 15    # cap per-class numeric stats to top-K by |corr|
SEGMENT_MAX_ENTRIES = 200            # hard cap on rate_by_category size


def declared_target_column(run_path: Path) -> Optional[str]:
    """
    Resolves the target column the pipeline itself declared in this run's
    summary/registry JSONs (analyst summaries carry dataset_overview.target_column;
    ML summaries and model_registry carry target_column). Returns the most common
    declared value, or None if nothing declared it.
    """
    import json as _json
    from collections import Counter

    votes: Counter = Counter()
    candidates = list(run_path.glob("*summary*.json")) + list(run_path.glob("model_registry.json"))
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = _json.load(f)
        except Exception:
            continue

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                v = node.get("target_column")
                if isinstance(v, str) and v.strip():
                    votes[v.strip()] += 1
                for vv in node.values():
                    _walk(vv)
            elif isinstance(node, list):
                for vv in node:
                    _walk(vv)

        _walk(data)
    return votes.most_common(1)[0][0] if votes else None


def detect_target_column(df: "pd.DataFrame", declared_target: Optional[str] = None) -> Optional[str]:
    """
    Picks the binary target column. Prefers the pipeline-declared target; the only
    structural fallback is a frame with exactly one binary column. Never guesses
    among multiple binary columns.
    """
    if declared_target is not None:
        for c in df.columns:
            if str(c) == str(declared_target) and df[c].nunique(dropna=True) == 2:
                return c
    binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if len(binary_cols) == 1:
        return binary_cols[0]
    return None


def compute_segment_stats(df: "pd.DataFrame", declared_target: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Computes ground-truth segment statistics from a dataset artifact (works for
    any dataset; see the module block above for the generic design rules).

    Returns None when no binary target can be identified (e.g. regression runs,
    or frames with several equally-plausible binary columns).
    """
    import pandas as pd
    import numpy as np

    if df is None or len(df) == 0:
        return None
    target = detect_target_column(df, declared_target)
    if target is None:
        return None

    y_raw = df[target]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
        pos_label = 1 if 1 in set(y.dropna().unique()) else sorted(y.dropna().unique())[-1]
    else:
        y = y_raw.astype(str)
        vc = y.value_counts()
        pos_label = vc.index[-1] if len(vc) else None  # minority class
    if pos_label is None or y.notna().sum() == 0:
        return None

    valid = y.notna()
    yv = y[valid]
    stats: Dict[str, Any] = {
        "target_column": str(target),
        "positive_class": int(pos_label) if isinstance(pos_label, (int, np.integer)) else str(pos_label),
        "overall": {
            "count": int(valid.sum()),
            "positives": int((yv == pos_label).sum()),
            "rate": round(float((yv == pos_label).mean()), 4),
        },
        "rate_by_category": [],
        "numeric_by_class": [],
        "notes": [
            "rate_by_category covers plain categorical columns, one-hot dummy groups "
            "(including the all-zeros '<reference>' category), and standalone 0/1 columns.",
            "Positive class is the minority class of the target.",
        ],
    }

    def _rate_for(mask: "pd.Series", feature: str, category: str) -> None:
        if len(stats["rate_by_category"]) >= SEGMENT_MAX_ENTRIES:
            return
        m = (mask & valid) if mask.dtype == bool else None
        if m is None:
            return
        n = int(m.sum())
        if n < SEGMENT_MIN_CATEGORY_ROWS:
            return
        pos = int((y[m] == pos_label).sum())
        stats["rate_by_category"].append({
            "feature": feature,
            "category": category,
            "count": n,
            "positives": pos,
            "rate": round(float(pos / n), 4),
        })

    # 1. Plain non-numeric categorical columns (structural cardinality limits only)
    for col in df.columns:
        if col == target or pd.api.types.is_numeric_dtype(df[col]):
            continue
        vc = df[col].value_counts(dropna=False)
        if 2 <= len(vc) <= SEGMENT_MAX_CATEGORIES:
            for cat in vc.index:
                _rate_for(df[col].astype(str) == str(cat), str(col), str(cat))

    # 2. One-hot dummy groups + standalone 0/1 columns
    binary_cols = [
        c for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
        and set(pd.unique(df[c].dropna())).issubset({0, 1, 0.0, 1.0, False, True})
    ]
    groups: Dict[str, List[Any]] = {}
    for col in binary_cols:
        name = str(col)
        prefix = name.rsplit("_", 1)[0] if "_" in name else name
        groups.setdefault(prefix, []).append(col)

    for prefix, cols in groups.items():
        if len(cols) >= 2:
            for col in cols:
                _rate_for(df[col] == 1, prefix, str(col)[len(prefix) + 1:])
            ref_mask = pd.Series(True, index=df.index)
            for col in cols:
                ref_mask &= df[col] == 0
            _rate_for(ref_mask, prefix, "<reference>")
        else:
            col = cols[0]
            _rate_for(df[col] == 1, str(col), "1")
            _rate_for(df[col] == 0, str(col), "0")

    # 3. Per-target-class median/mean for numeric columns (top-K by |corr| with
    #    the positive-class indicator; skips the binary/dummy columns themselves).
    #    No name filters — ranking is purely statistical.
    y_ind = (y == pos_label).astype(float)
    scored: List[Tuple[Any, float]] = []
    for col in df.columns:
        if col == target or not pd.api.types.is_numeric_dtype(df[col]) or col in binary_cols:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.nunique(dropna=True) <= 2:
            continue
        corr = series.corr(y_ind)
        scored.append((col, abs(float(corr)) if corr is not None and not pd.isna(corr) else 0.0))
    top_numeric = [c for c, _ in sorted(scored, key=lambda t: t[1], reverse=True)[:SEGMENT_MAX_NUMERIC_BY_CLASS]]

    for col in top_numeric:
        series = pd.to_numeric(df[col], errors="coerce")
        for cls in sorted(yv.unique(), key=lambda v: str(v)):
            vals = series[valid & (y == cls)].dropna()
            if len(vals) == 0:
                continue
            stats["numeric_by_class"].append({
                "column": str(col),
                "class": int(cls) if isinstance(cls, (int, np.integer)) else str(cls),
                "median": round(float(vals.median()), 4),
                "mean": round(float(vals.mean()), 4),
            })

    return stats


def build_verified_metrics(run_dir: str = ".", uploaded_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Scans run_dir for generated data science artifacts and deterministically computes
    ground-truth verified metrics.

    Args:
        run_dir: Path to directory containing artifact files (defaults to current working directory).
        uploaded_files: Optional paths of the user's raw uploads. When provided, their
            original (un-encoded) categorical columns are merged into segment_stats so
            reference categories dropped by one-hot encoding keep their real names.

    Returns:
        Dictionary containing all verified metrics.
    """
    import pandas as pd
    import numpy as np

    run_path = Path(run_dir).resolve()
    verified: Dict[str, Any] = {
        "dataset": {},
        "data_quality": {},
        "models": [],
        "best_model": None,
        "model_metrics": {},
        "top_features": [],
        "confusion_matrix": {},
        "baseline_score": {},
        "predictions_summary": {},
        "artifacts_manifest": []
    }

    # 1. Manifest of artifacts
    if run_path.exists():
        for item in sorted(run_path.iterdir()):
            if item.is_file():
                verified["artifacts_manifest"].append({
                    "filename": item.name,
                    "size_bytes": item.stat().st_size,
                    "extension": item.suffix.lower()
                })

    # Preserve the explicitly computed unsupervised metrics for the reporter.
    # These remain artifact-sourced values; unlike dataset shape and supervised
    # confusion-matrix metrics, they are not independently recomputed here.
    unsupervised_path = run_path / "unsupervised_metrics.json"
    if unsupervised_path.is_file():
        try:
            with unsupervised_path.open(encoding="utf-8") as f:
                unsupervised_metrics = json.load(f)
            if isinstance(unsupervised_metrics, dict):
                verified["unsupervised_metrics_artifact"] = {
                    "source_file": unsupervised_path.name,
                    "values_independently_recomputed": False,
                    "content": unsupervised_metrics,
                }
        except (OSError, json.JSONDecodeError) as exc:
            verified["unsupervised_metrics_artifact"] = {
                "source_file": unsupervised_path.name,
                "error": f"Could not read unsupervised metrics artifact: {exc}",
            }

    # 1b. Versioned Model Artifacts (e.g., logistic_regression_p0.pkl, random_forest_p1_tuned.pkl)
    versioned_model_artifacts = []
    for item in sorted(run_path.glob("*_p*.pkl")):
        if item.is_file():
            # Parse version info from filename
            # Patterns: logistic_regression_p0.pkl, random_forest_p1_tuned.pkl, xgboost_p2_tuned.pkl
            versioned_model_artifacts.append({
                "filename": item.name,
                "size_bytes": item.stat().st_size,
                "model_family": item.name.split("_p")[0] if "_p" in item.name else "unknown",
                "version": item.name.split("_p")[-1].replace(".pkl", "") if "_p" in item.name else "unknown"
            })
    if versioned_model_artifacts:
        verified["versioned_model_artifacts"] = versioned_model_artifacts

    # 2. Dataset Verification (cleaned_dataset.csv or fallback dataset CSVs)
    dataset_candidates = [
        "cleaned_dataset.csv",
        "feature_engineered_dataset.csv",
        "cleaned_data.csv",
        "dataset.csv",
        "data.csv"
    ]
    target_csv: Optional[Path] = None
    for cand in dataset_candidates:
        cand_path = run_path / cand
        if cand_path.exists() and cand_path.is_file():
            target_csv = cand_path
            break

    # If not found in candidates, look for any .csv that isn't model/prediction/feature
    if not target_csv:
        for item in sorted(run_path.glob("*.csv")):
            if item.name not in [
                "model_comparison.csv", "feature_importance.csv", "predictions.csv",
                "cluster_assignments.csv", "anomaly_scores.csv",
            ]:
                target_csv = item
                break

    if target_csv and target_csv.exists():
        try:
            df = pd.read_csv(target_csv)
            num_rows = int(df.shape[0])
            num_cols = int(df.shape[1])
            missing_cells = int(df.isna().sum().sum())
            total_cells = int(num_rows * num_cols) if num_rows * num_cols > 0 else 1
            missing_pct = round(float(missing_cells / total_cells * 100), 2)

            num_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols_list = df.select_dtypes(exclude=[np.number]).columns.tolist()
            # Numeric storage does not make an identifier a quantitative
            # feature. Keep identifier columns visible in schema metadata, but
            # do not publish misleading means/ranges for IDs as metric stats.
            import re
            identifier_cols = [
                str(col) for col in df.columns
                if re.search(r"(?:^|[_\s-])id(?:$|[_\s-])|identifier|id$|^id", str(col), re.IGNORECASE)
            ]
            numeric_measure_cols = [col for col in num_cols_list if col not in identifier_cols]

            numeric_stats: Dict[str, Dict[str, Any]] = {}
            for col in numeric_measure_cols:
                series = df[col].dropna()
                if len(series) > 0:
                    skew_val = safe_float(series.skew())
                    numeric_stats[col] = {
                        "min": round(float(series.min()), 4),
                        "max": round(float(series.max()), 4),
                        "mean": round(float(series.mean()), 4),
                        "median": round(float(series.median()), 4),
                        "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                        "skewness": round(skew_val, 4) if skew_val is not None else None,
                        "missing_count": int(df[col].isna().sum()),
                        "missing_pct": round(float(df[col].isna().mean() * 100), 2)
                    }

            categorical_stats: Dict[str, Dict[str, Any]] = {}
            for col in cat_cols_list:
                series = df[col].dropna()
                vc = series.value_counts()
                top_counts = {str(k): int(v) for k, v in vc.head(5).items()}
                categorical_stats[col] = {
                    "unique_count": int(series.nunique()),
                    "missing_count": int(df[col].isna().sum()),
                    "top_categories": top_counts
                }

            verified["dataset"] = {
                "source_file": target_csv.name,
                "rows": num_rows,
                "cols": num_cols,
                "missing_cells": missing_cells,
                "missing_pct": missing_pct,
                "numeric_columns_count": len(num_cols_list),
                "categorical_columns_count": len(cat_cols_list),
                "numeric_columns": num_cols_list,
                "numeric_measure_columns": numeric_measure_cols,
                "identifier_columns": identifier_cols,
                "categorical_columns": cat_cols_list,
                "numeric_stats": numeric_stats,
                "categorical_stats": categorical_stats
            }

            # 3. Data Quality & Bounds Check
            bound_violations = []
            distribution_anomalies = []
            near_duplicates = []

            for col, stats in numeric_stats.items():
                col_lower = col.lower()
                c_min = stats["min"]
                c_max = stats["max"]

                # Percentage / rate bounds
                if ("pct" in col_lower or "percent" in col_lower or "rate" in col_lower) and not ("growth" in col_lower or "diff" in col_lower):
                    if c_max > 100.0 or c_min < 0.0:
                        bound_violations.append(f"{col}: observed range [{c_min}, {c_max}] violates expected percentage [0, 100]")
                elif "age" in col_lower:
                    if c_min < 0 or c_max > 150:
                        bound_violations.append(f"{col}: observed range [{c_min}, {c_max}] violates age bounds [0, 150]")
                elif "day" in col_lower and "365" in col_lower:
                    if c_max > 365 or c_min < 0:
                        bound_violations.append(f"{col}: observed range [{c_min}, {c_max}] violates [0, 365] days")

                # Skewness / uniformity check
                skew = stats.get("skewness")
                if skew is not None and abs(skew) < 0.05 and ("price" in col_lower or "charge" in col_lower or "income" in col_lower):
                    distribution_anomalies.append(f"{col}: skewness is {skew} (suspiciously uniform for a financial/price metric)")

            # Categorical near duplicates check
            for col in cat_cols_list[:10]:
                uniques = [str(x) for x in df[col].dropna().unique() if str(x).strip()]
                if len(uniques) > 1 and len(uniques) < 100:
                    for i in range(len(uniques)):
                        for j in range(i + 1, len(uniques)):
                            s1, s2 = uniques[i], uniques[j]
                            if s1.lower() == s2.lower() and s1 != s2:
                                near_duplicates.append(f"{col}: case mismatch '{s1}' vs '{s2}'")
                            elif len(s1) > 4 and len(s2) > 4:
                                ratio = difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
                                if 0.85 <= ratio < 1.0:
                                    c1 = int((df[col] == s1).sum())
                                    c2 = int((df[col] == s2).sum())
                                    near_duplicates.append(f"{col}: potential typo '{s1}' ({c1} rows) vs '{s2}' ({c2} rows)")

            verified["data_quality"] = {
                "bound_violations": bound_violations,
                "distribution_anomalies": distribution_anomalies,
                "near_duplicates": near_duplicates[:10]
            }

        except Exception as e:
            verified["dataset"] = {"error": f"Failed to parse dataset {target_csv.name}: {str(e)}"}

    # 4d. Model Registry & Champion Pointer (load FIRST to get primary_metric from Planner)
    reg_path = run_path / "model_registry.json"
    if reg_path.exists() and reg_path.is_file():
        try:
            with open(reg_path, "r", encoding="utf-8", errors="replace") as f:
                verified["model_registry"] = json.load(f)
        except Exception as e:
            verified["model_registry_error"] = f"Failed to parse model_registry.json: {str(e)}"

    champ_path = run_path / "champion_model.json"
    primary_metric = None
    if champ_path.exists() and champ_path.is_file():
        try:
            with open(champ_path, "r", encoding="utf-8", errors="replace") as f:
                verified["champion_model"] = json.load(f)
            # Extract the primary_metric that was set by the Planner agent
            if isinstance(verified["champion_model"], dict):
                primary_metric = verified["champion_model"].get("primary_metric")
        except Exception as e:
            verified["champion_model_error"] = f"Failed to parse champion_model.json: {str(e)}"

    # 4. Model Comparison (model_comparison.csv)
    mc_path = run_path / "model_comparison.csv"
    if mc_path.exists() and mc_path.is_file():
        try:
            mc = pd.read_csv(mc_path)
            # Standardize numeric columns
            for col in mc.columns:
                if col.lower() not in ["model", "algorithm", "name", "notes", "model_artifact", "artifact", "configuration"]:
                    mc[col] = pd.to_numeric(mc[col], errors="coerce")

            # Round numeric columns to 4 decimals
            numeric_cols = mc.select_dtypes(include=[np.number]).columns
            mc[numeric_cols] = mc[numeric_cols].round(4)
            verified["models"] = mc.to_dict(orient="records")

            # Build quick model metrics map
            model_col = "model" if "model" in mc.columns else (mc.columns[0] if len(mc.columns) > 0 else None)
            if model_col:
                metrics_map = {}
                for _, row in mc.iterrows():
                    m_name = str(row[model_col])
                    m_dict = {}
                    for col in mc.columns:
                        if col != model_col:
                            val = row[col]
                            m_dict[col] = round(float(val), 4) if isinstance(val, (int, float, np.number)) and not np.isnan(val) else val
                    metrics_map[m_name] = m_dict
                verified["model_metrics"] = metrics_map

                # Determine best model using the primary_metric from champion_model.json (set by Planner)
                # Fall back to goal-aware priority if primary_metric not available
                target_metric = None
                sort_ascending = False

                # Normalize metric name for matching
                def _norm_metric_key(col: str) -> str:
                    return col.strip().lower().replace(" ", "").replace("-", "_")

                if primary_metric:
                    # Use the primary_metric specified by the Planner
                    pm = _norm_metric_key(primary_metric)
                    for col in mc.columns:
                        if _norm_metric_key(col) == pm:
                            target_metric = col
                            # Check if it's a metric to minimize (RMSE, MAE, MSE, log_loss)
                            if pm in {"rmse", "mae", "mse", "log_loss"}:
                                sort_ascending = True
                            break

                # Fallback: goal-aware priority (same logic as champion_selector)
                if not target_metric:
                    recall_priority = ["recall", "f1", "roc_auc", "roc-auc", "pr_auc", "pr-auc", "accuracy"]
                    default_priority = ["roc_auc", "roc-auc", "pr_auc", "pr-auc", "f1", "recall", "accuracy"]
                    # We don't have goal_text here, so use default priority
                    for cand in default_priority:
                        found_col = next((c for c in mc.columns if _norm_metric_key(c) == cand), None)
                        if found_col:
                            target_metric = found_col
                            sort_ascending = False
                            break

                if not target_metric:
                    for m_candidate in ["RMSE", "rmse", "MAE", "mae", "MSE", "mse"]:
                        found_col = next((c for c in mc.columns if c.strip().lower() == m_candidate.lower()), None)
                        if found_col:
                            target_metric = found_col
                            sort_ascending = True
                            break

                if target_metric and target_metric in mc.columns:
                    sorted_mc = mc.dropna(subset=[target_metric]).sort_values(by=target_metric, ascending=sort_ascending)
                    if len(sorted_mc) > 0:
                        verified["best_model"] = {
                            "model": str(sorted_mc.iloc[0][model_col]),
                            "selection_metric": target_metric,
                            "score": round(float(sorted_mc.iloc[0][target_metric]), 4)
                        }
        except Exception as e:
            verified["models_error"] = f"Failed to parse model_comparison.csv: {str(e)}"

    # 4b. Tuned Model Comparison (tuned_model_comparison.csv)
    tuned_mc_path = run_path / "tuned_model_comparison.csv"
    if tuned_mc_path.exists() and tuned_mc_path.is_file():
        try:
            tmc = pd.read_csv(tuned_mc_path)
            for col in tmc.columns:
                if col.lower() not in ["model", "algorithm", "name", "notes", "model_artifact", "artifact", "configuration"]:
                    tmc[col] = pd.to_numeric(tmc[col], errors="coerce")
            numeric_cols = tmc.select_dtypes(include=[np.number]).columns
            tmc[numeric_cols] = tmc[numeric_cols].round(4)
            verified["tuned_models"] = tmc.to_dict(orient="records")

            model_col = "model" if "model" in tmc.columns else (tmc.columns[0] if len(tmc.columns) > 0 else None)
            if model_col:
                tuned_metrics_map = {}
                for _, row in tmc.iterrows():
                    m_name = str(row[model_col])
                    m_dict = {}
                    for col in tmc.columns:
                        if col != model_col:
                            val = row[col]
                            m_dict[col] = round(float(val), 4) if isinstance(val, (int, float, np.number)) and not np.isnan(val) else val
                    tuned_metrics_map[m_name] = m_dict
                verified["tuned_model_metrics"] = tuned_metrics_map
        except Exception as e:
            verified["tuned_models_error"] = f"Failed to parse tuned_model_comparison.csv: {str(e)}"

    # 4c. Hyperparameter Tuning Summary & Best Params
    hpt_summary_path = run_path / "hyperparameter_tuning_summary.json"
    if hpt_summary_path.exists() and hpt_summary_path.is_file():
        try:
            with open(hpt_summary_path, "r", encoding="utf-8", errors="replace") as f:
                verified["hyperparameter_tuning_summary"] = json.load(f)
        except Exception as e:
            verified["hyperparameter_tuning_summary_error"] = f"Failed to parse hyperparameter_tuning_summary.json: {str(e)}"

    best_params_path = run_path / "best_params.json"
    if best_params_path.exists() and best_params_path.is_file():
        try:
            with open(best_params_path, "r", encoding="utf-8", errors="replace") as f:
                verified["best_params"] = json.load(f)
        except Exception as e:
            verified["best_params_error"] = f"Failed to parse best_params.json: {str(e)}"

    # 5. Feature Importance (feature_importance.csv)
    fi_path = run_path / "feature_importance.csv"
    if fi_path.exists() and fi_path.is_file():
        try:
            fi = pd.read_csv(fi_path)
            # Find feature name column and importance column
            feat_col = next((c for c in fi.columns if "feat" in c.lower() or "var" in c.lower() or "name" in c.lower()), fi.columns[0])
            imp_col = next((c for c in fi.columns if "imp" in c.lower() or "score" in c.lower() or "coef" in c.lower() or "weight" in c.lower()), fi.columns[1] if len(fi.columns) > 1 else None)

            if imp_col:
                fi[imp_col] = pd.to_numeric(fi[imp_col], errors="coerce")
                fi["abs_importance"] = fi[imp_col].abs()
                sorted_fi = fi.sort_values(by="abs_importance", ascending=False)
                top_items = []
                for _, row in sorted_fi.head(10).iterrows():
                    val = row[imp_col]
                    top_items.append({
                        "feature": str(row[feat_col]),
                        "importance": round(float(val), 4) if pd.notna(val) else None,
                        "abs_importance": round(float(row["abs_importance"]), 4) if pd.notna(row["abs_importance"]) else None
                    })
                verified["top_features"] = top_items
        except Exception as e:
            verified["top_features_error"] = f"Failed to parse feature_importance.csv: {str(e)}"

    # 6. Confusion Matrix (confusion_matrix.json)
    cm_path = run_path / "confusion_matrix.json"
    if cm_path.exists() and cm_path.is_file():
        try:
            with open(cm_path, "r", encoding="utf-8", errors="replace") as f:
                cm_data = json.load(f)
            verified["confusion_matrix"] = cm_data
        except Exception as e:
            verified["confusion_matrix_error"] = f"Failed to parse confusion_matrix.json: {str(e)}"

    # 7. Baseline Score (baseline_score.json)
    base_path = run_path / "baseline_score.json"
    if base_path.exists() and base_path.is_file():
        try:
            with open(base_path, "r", encoding="utf-8", errors="replace") as f:
                base_data = json.load(f)
            verified["baseline_score"] = base_data
        except Exception as e:
            verified["baseline_score_error"] = f"Failed to parse baseline_score.json: {str(e)}"

    # 8. Predictions Summary (predictions.csv)
    preds_df: Optional["pd.DataFrame"] = None
    pred_path = run_path / "predictions.csv"
    if pred_path.exists() and pred_path.is_file():
        try:
            preds_df = pd.read_csv(pred_path)
            total_preds = int(len(preds_df))
            pred_summary: Dict[str, Any] = {"total_predictions": total_preds}
            pred_col = next((c for c in preds_df.columns if "pred" in c.lower() or "target" in c.lower() or "label" in c.lower()), None)
            if pred_col:
                vc = preds_df[pred_col].value_counts()
                pred_summary["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
                if len(vc) == 2:
                    # binary classification positive rate
                    pos_class = vc.index[1] if 1 in vc.index else vc.index[0]
                    pred_summary["positive_class_count"] = int(vc[pos_class])
                    pred_summary["positive_class_pct"] = round(float(vc[pos_class] / total_preds * 100), 2)
            verified["predictions_summary"] = pred_summary
        except Exception as e:
            verified["predictions_summary_error"] = f"Failed to parse predictions.csv: {str(e)}"

    # 8b. Threshold analysis grounding (threshold_analysis.json) + flagged volumes
    # recomputed from stored prediction probabilities, so report claims like
    # "threshold 0.1885 flags 911/1409 (64.7%)" are verifiable.
    ta_path = run_path / "threshold_analysis.json"
    if ta_path.exists() and ta_path.is_file():
        try:
            with open(ta_path, "r", encoding="utf-8", errors="replace") as f:
                ta_data = json.load(f)
            if isinstance(ta_data, dict):
                flagged_volumes: Dict[str, Any] = {}
                if preds_df is not None:
                    prob_col = next((c for c in preds_df.columns if "prob" in str(c).lower()), None)
                    truth_col = next((c for c in preds_df.columns
                                      if any(k in str(c).lower() for k in ("true", "actual"))), None)
                    if prob_col is not None:
                        probs = pd.to_numeric(preds_df[prob_col], errors="coerce")
                        truth = (pd.to_numeric(preds_df[truth_col], errors="coerce")
                                 if truth_col is not None else None)
                        mask = probs.notna() & (truth.notna() if truth is not None else True)
                        probs = probs[mask]
                        truth = truth[mask] if truth is not None else None
                        for vol_key, thr in (("default", ta_data.get("default_threshold")),
                                             ("recommended", ta_data.get("recommended_threshold"))):
                            thr_f = safe_float(thr)
                            if thr_f is None or len(probs) == 0:
                                continue
                            flagged_mask = probs >= thr_f
                            flagged = int(flagged_mask.sum())
                            vol = {
                                "threshold": thr_f,
                                "flagged_count": flagged,
                                "total": int(len(probs)),
                                "flagged_pct": round(float(flagged / len(probs) * 100), 2),
                            }
                            if truth is not None:
                                tp = int(((probs >= thr_f) & (truth == 1)).sum())
                                vol["true_positives"] = tp
                                if tp > 0:
                                    vol["contacts_per_true_positive"] = round(flagged / tp, 2)
                            flagged_volumes[vol_key] = vol
                if flagged_volumes:
                    ta_data["flagged_volume"] = flagged_volumes
                verified["threshold_analysis"] = ta_data
        except Exception as e:
            verified["threshold_analysis_error"] = f"Failed to parse threshold_analysis.json: {str(e)}"

    # 10. Segment statistics grounding (dataset-agnostic; see compute_segment_stats).
    # Preferred frame: feature_engineered_dataset.csv (may hold plain categoricals
    # such as binned groups), else the cleaned dataset, else the profiled dataset.
    try:
        seg_df: Optional["pd.DataFrame"] = None
        seg_source: Optional[str] = None
        for cand in ("feature_engineered_dataset.csv", "cleaned_dataset.csv", "cleaned_data.csv"):
            cand_path = run_path / cand
            if cand_path.exists() and cand_path.is_file():
                seg_df = pd.read_csv(cand_path)
                seg_source = cand
                break
        if seg_df is None and target_csv is not None and target_csv.exists():
            seg_df = pd.read_csv(target_csv)
            seg_source = target_csv.name
        if seg_df is not None:
            declared = declared_target_column(run_path)
            seg_stats = compute_segment_stats(seg_df, declared_target=declared)
            if seg_stats is not None:
                seg_stats["source_file"] = seg_source
                # Merge named segments from the raw uploads: the run-dir datasets are
                # typically one-hot encoded, so reference categories only exist there
                # under their real names (raw entries win on label collisions).
                for up in uploaded_files or []:
                    try:
                        if not up or not str(up).lower().endswith(".csv") or not os.path.exists(up):
                            continue
                        raw_df = pd.read_csv(up, nrows=500_000)
                        if declared not in raw_df.columns:
                            continue
                        raw_stats = compute_segment_stats(raw_df, declared_target=declared)
                        if not raw_stats:
                            continue
                        by_key = {
                            (str(s["feature"]).lower(), str(s["category"]).lower()): s
                            for s in seg_stats["rate_by_category"]
                        }
                        for s in raw_stats.get("rate_by_category", []):
                            by_key[(str(s["feature"]).lower(), str(s["category"]).lower())] = s
                        seg_stats["rate_by_category"] = sorted(
                            by_key.values(), key=lambda s: abs(float(s["rate"]) - seg_stats["overall"]["rate"]),
                            reverse=True,
                        )
                        seg_stats["raw_source_file"] = os.path.basename(up)
                    except Exception:
                        continue
                verified["segment_stats"] = seg_stats
    except Exception as e:
        verified["segment_stats_error"] = f"Failed to compute segment stats: {str(e)}"

    # 9. Save verified_metrics.json locally to run_dir
    try:
        out_file = run_path / "verified_metrics.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(verified, f, indent=2)
    except Exception as e:
        pass

    return verified


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    result = build_verified_metrics(target_dir)
    print("VERIFIED_METRICS_JSON_START")
    print(json.dumps(result, indent=2, default=str))
    print("VERIFIED_METRICS_JSON_END")
