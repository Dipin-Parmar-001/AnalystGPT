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


def build_verified_metrics(run_dir: str = ".") -> Dict[str, Any]:
    """
    Scans run_dir for generated data science artifacts and deterministically computes
    ground-truth verified metrics.
    
    Args:
        run_dir: Path to directory containing artifact files (defaults to current working directory).
        
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
            if item.name not in ["model_comparison.csv", "feature_importance.csv", "predictions.csv"]:
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

            numeric_stats: Dict[str, Dict[str, Any]] = {}
            for col in num_cols_list:
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

    # 4. Model Comparison (model_comparison.csv & tuned_model_comparison.csv)
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

                # Determine best model
                # Priority metrics: ROC-AUC, ROC_AUC, roc_auc, PR-AUC, F1, F1-Score, f1_score, f1, AUC, Accuracy, accuracy, R2, RMSE
                target_metric = None
                sort_ascending = False
                for m_candidate in ["ROC-AUC", "ROC_AUC", "roc_auc", "PR-AUC", "pr_auc", "F1", "F1-Score", "f1_score", "f1", "AUC", "Accuracy", "accuracy", "R2", "r2_score"]:
                    found_col = next((c for c in mc.columns if c.strip().lower() == m_candidate.lower()), None)
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

    # 4d. Model Registry & Champion Pointer
    reg_path = run_path / "model_registry.json"
    if reg_path.exists() and reg_path.is_file():
        try:
            with open(reg_path, "r", encoding="utf-8", errors="replace") as f:
                verified["model_registry"] = json.load(f)
        except Exception as e:
            verified["model_registry_error"] = f"Failed to parse model_registry.json: {str(e)}"

    champ_path = run_path / "champion_model.json"
    if champ_path.exists() and champ_path.is_file():
        try:
            with open(champ_path, "r", encoding="utf-8", errors="replace") as f:
                verified["champion_model"] = json.load(f)
        except Exception as e:
            verified["champion_model_error"] = f"Failed to parse champion_model.json: {str(e)}"

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
