"""
Fact Sheet - AnalystGPT.

`build_facts()` runs after the preflight gate passes (or produced warnings) and
produces a SHORT, flat key/value fact sheet whose values are already formatted
for the report ("42.7%", "1,234 rows"). The reporter writes the report with
{{placeholder}} blanks referencing these keys only, and host-side code fills
the blanks — the reporter never types or converts a number itself.

The fact sheet is ADDITIVE context: the reporter still receives verified
metrics, summaries and the artifact list; the fact sheet is the single source
its blanks may reference.
"""

import os
import json
from typing import Dict, Any, List, Optional

# Columns whose "importance" is an artifact of data leakage or identifiers, not
# signal; plus constant columns are dropped by variance, not by name.
LEAKY_PATTERNS = ("churn", "target", "label", "customerid", "id", "count")


def _fmt_pct(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value * 100:.1f}%"


def _fmt_num(value: Optional[float], digits: int = 4) -> Optional[str]:
    if value is None:
        return None
    return f"{value:.{digits}f}"


def _pretty_name(raw: str) -> str:
    """'cat__Contract_Two year' -> 'Contract: Two year'; 'num__Tenure Months' -> 'Tenure Months'."""
    name = str(raw)
    if "__" in name:
        prefix, rest = name.split("__", 1)
        if prefix in ("cat", " categorical"):
            parts = rest.rsplit("_", 1)
            if len(parts) == 2 and parts[0]:
                return f"{parts[0].replace('_', ' ')}: {parts[1].replace('_', ' ')}"
        return rest.replace("_", " ")
    return name.replace("_", " ")


def _is_leaky(feature: str) -> bool:
    f = str(feature).lower()
    return any(p in f for p in LEAKY_PATTERNS)


def build_facts(
    output_dir: str,
    verified_metrics: Optional[Dict[str, Any]] = None,
    preflight: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build the flat fact sheet for the reporter.

    All values are strings, pre-formatted. Keys use snake_case and become the
    {{blanks}} the reporter may reference.
    """
    vm = verified_metrics or {}
    pre = preflight or {}
    champ = pre.get("champion") or {}
    facts: Dict[str, str] = {}

    # --- Best model (exactly one; no alternatives) -----------------------------
    model_display = champ.get("model_name")
    if not model_display:
        bm = vm.get("best_model")
        model_display = bm.get("name") if isinstance(bm, dict) else bm
    artifact = champ.get("artifact_filename") or champ.get("model")
    facts["best_model"] = str(model_display or "not determined")
    facts["best_model_artifact"] = str(artifact or "not available")
    metric_name = champ.get("primary_metric_name") or champ.get("primary_metric")
    if metric_name:
        facts["best_model_selection_metric"] = f"{metric_name.upper()} = {_fmt_num(champ.get('score'), 4)}"

    # --- Champion metrics (from the winner's full row) --------------------------
    metrics = champ.get("_winner_metrics") or {}
    for src, dst in (
        ("recall", "best_model_recall"),
        ("precision", "best_model_precision"),
        ("accuracy", "best_model_accuracy"),
        ("f1", "best_model_f1"),
    ):
        v = metrics.get(src)
        if v is not None:
            facts[dst] = _fmt_pct(v)

    # --- Confusion matrix (recomputed ground truth preferred) --------------------
    cm = vm.get("confusion_matrix") or {}
    tp, fp, fn, tn = (cm.get(k) for k in ("true_positives", "false_positives", "false_negatives", "true_negatives"))
    if all(isinstance(v, (int, float)) for v in (tp, fp, fn, tn)):
        facts["confusion_matrix"] = f"TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}"
        total = tp + fp + fn + tn
        facts["test_set_size"] = f"{int(total):,} samples"
        # Business framing: contacts needed per churner caught, at this threshold.
        if tp:
            facts["contacts_per_true_positive"] = f"{(tp + fp) / tp:.1f} contacts per true positive ({int(tp + fp)} flagged to catch {int(tp)})"

    # --- Dataset shape -----------------------------------------------------------
    ds = vm.get("dataset") or {}
    if ds.get("rows") is not None:
        facts["dataset_rows"] = f"{int(ds['rows']):,} rows"
        facts["dataset_columns"] = f"{int(ds['cols'])} columns"
        facts["dataset_source_file"] = str(ds.get("source_file", ""))
        if ds.get("missing_pct") is not None:
            facts["dataset_missing_pct"] = _fmt_pct(ds.get("missing_pct") / 100)

    # --- Drivers: two ranked lists with directions -------------------------------
    drivers_up: List[str] = []
    drivers_down: List[str] = []
    fi_path = os.path.join(output_dir, "feature_importance.csv")
    if os.path.isfile(fi_path):
        try:
            import pandas as pd
            fi = pd.read_csv(fi_path)
            imp_col = next((c for c in ("absolute_importance", "importance", "coefficient_abs", "|coefficient|") if c in fi.columns), None)
            feat_col = next((c for c in ("feature", "feature_name") if c in fi.columns), None)
            if imp_col and feat_col:
                fi = fi[~fi[feat_col].astype(str).str.lower().str.contains("|".join(LEAKY_PATTERNS), na=False)]
                if "direction" in fi.columns:
                    up = fi[fi["direction"].astype(str).str.lower().str.startswith("increase")]
                    down = fi[fi["direction"].astype(str).str.lower().str.startswith("reduce")]
                elif "coefficient" in fi.columns:
                    up = fi[fi["coefficient"] > 0]
                    down = fi[fi["coefficient"] < 0]
                else:
                    up, down = fi, fi.iloc[0:0]
                for i, (_, r) in enumerate(up.head(5).iterrows(), 1):
                    drivers_up.append(f"{i}. {_pretty_name(r[feat_col])}")
                for i, (_, r) in enumerate(down.head(5).iterrows(), 1):
                    drivers_down.append(f"{i}. {_pretty_name(r[feat_col])}")
        except Exception as e:
            print(f"[FactSheet] feature importance parse error: {e}")
    if drivers_up:
        facts["drivers_increasing_outcome"] = "\n".join(drivers_up)
    if drivers_down:
        facts["drivers_reducing_outcome"] = "\n".join(drivers_down)

    # --- File list (from the folder, once) ----------------------------------------
    if os.path.isdir(output_dir):
        files = sorted(f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
        facts["artifact_count"] = f"{len(files)} files"
        facts["artifact_list"] = "\n".join(files)

    # --- Machine-computed warnings (must appear in the report) --------------------
    warnings: List[str] = pre.get("warnings") or []
    if warnings:
        facts["mandatory_warnings"] = "\n".join(f"- {w}" for w in warnings)

    return facts


# ----------------------------------------------------------------------------
# Blank templating: the reporter writes {{key}} tokens, code fills them.
# ----------------------------------------------------------------------------

import re as _re

_BLANK_RE = _re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def extract_blanks(text: str) -> List[str]:
    """All {{blank}} names used in the report, in order of first appearance."""
    seen: List[str] = []
    for name in _BLANK_RE.findall(text or ""):
        if name not in seen:
            seen.append(name)
    return seen


def fill_blanks(text: str, facts: Dict[str, str]) -> Dict[str, Any]:
    """Substitute {{blank}} tokens with fact-sheet values.

    Returns {"report": filled_text, "unknown_blanks": [names not in facts]}.
    Unknown blanks are left in place so the caller can decide (repair turn or
    "(not verified in current run)" substitution).
    """
    unknown: List[str] = []

    def _sub(m: "_re.Match") -> str:
        key = m.group(1)
        if key in facts:
            return str(facts[key])
        if key not in unknown:
            unknown.append(key)
        return m.group(0)

    filled = _BLANK_RE.sub(_sub, text or "")
    return {"report": filled, "unknown_blanks": unknown}


def replace_unresolved_blanks(text: str) -> str:
    """Last-resort: any blank still unfilled becomes '(not verified in current run)'."""
    return _BLANK_RE.sub("(not verified in current run)", text or "")
