"""
Deterministic Numeric Verifier Module for AnalystGPT.

Extracts numeric claims from draft markdown reports via regex and cross-references each claim
against authoritative ground-truth metrics (VERIFIED_METRICS).

Operates deterministically via pure Python code (NO LLM calls).

VERIFIED_METRICS now includes:
- Dataset metrics (rows, cols, missing values, numeric/categorical stats)
- Data quality findings (bound violations, distribution anomalies, near duplicates)
- Model comparison metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, CV_Mean, CV_Std)
- Tuned model comparison metrics
- Hyperparameter tuning summary and best params
- Model registry entries and champion model pointer
- Versioned model artifacts (logistic_regression_p0.pkl, random_forest_p1_tuned.pkl, etc.)
- Feature importance, confusion matrix, baseline score, predictions summary
"""

import re
import math
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NumericClaim:
    raw_str: str
    num_val: float
    is_percentage: bool
    context: str
    line_number: int


@dataclass
class VerificationResult:
    is_valid: bool
    total_claims_count: int
    verified_count: int
    unverified_count: int
    verified_claims: List[NumericClaim] = field(default_factory=list)
    unverified_claims: List[NumericClaim] = field(default_factory=list)
    correction_prompt: str = ""


# Ignore patterns for headers, list bullets, years, standard phrases
RE_MARKDOWN_HEADER = re.compile(r"^\s*#{1,6}\s*(?:\d+[\.\)]\s*)?.*$")
RE_NUMBERED_LIST_PREFIX = re.compile(r"^\s*(?:\d+[\.\)]|\(\d+\))\s+")
RE_STEP_OR_TASK = re.compile(r"\b(?:step|task|phase|section|figure|table|chart|part|item)\s*#?\s*\d+\b", re.IGNORECASE)
RE_TOP_N = re.compile(r"\b(?:top|bottom|first|last|next|prior)\s+\d+\b", re.IGNORECASE)
RE_YEAR = re.compile(r"\b(?:19\d\d|20[0-3]\d)\b")
RE_FILE_EXT_DIGIT = re.compile(r"\b[\w\-]+_task_\d+\b|\b[\w\-]+\.\w+\b", re.IGNORECASE)


def _safe_float_parse(s: str) -> Optional[float]:
    """Converts a numeric string to float, removing commas and %."""
    try:
        clean = s.replace(",", "").replace("%", "").strip()
        if clean:
            return float(clean)
    except (ValueError, TypeError):
        pass
    return None


def extract_numeric_claims(markdown_text: str) -> List[NumericClaim]:
    """
    Extracts substantive numeric claims from markdown text while filtering out:
    - Markdown headings and numbered list bullets (e.g. '1. ', '2. ')
    - Reference indicators (e.g. 'Task #1', 'Step 2', 'Top 5')
    - Standard calendar years (e.g. 2020-2029)
    - Code blocks and file references
    """
    claims: List[NumericClaim] = []
    if not markdown_text:
        return claims

    in_code_block = False
    lines = markdown_text.splitlines()

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped:
            continue

        # Skip pure header lines where numbers might just be section numbers
        if stripped.startswith("#"):
            # Check if header contains metric claims like "Accuracy: 0.85" vs "## 1. Executive Summary"
            if not any(kw in stripped.lower() for kw in ["accuracy", "f1", "auc", "precision", "recall", "score", "rows", "records", "%"]):
                continue

        # Clean line of non-claim numeric prefixes
        working_line = line

        # Strip numbered list marker at start of line: e.g. "1. " or "  2) "
        working_line = RE_NUMBERED_LIST_PREFIX.sub(" ", working_line)

        # Mask out common false-positive patterns
        # 1. Step/Task references: "Task #1", "Step 2"
        working_line = RE_STEP_OR_TASK.sub(" [PHRASE] ", working_line)
        # 2. Top N phrases: "top 5", "top 10"
        working_line = RE_TOP_N.sub(" [TOP_N] ", working_line)
        # 3. File names: "analyst_summary_task_1.json", "chart1.png"
        working_line = re.sub(r"\b[\w\-]+_task_\d+\b|\b[A-Za-z0-9_\-]+\.[A-Za-z]{2,5}\b", " [FILE] ", working_line)

        # Regex to capture numeric tokens: percentages, floats, comma-ints, standalone ints
        # Matches patterns like 0.851, 1,397.47, 26.54%, 7043, 199, -0.45, etc.
        pattern = r"(?<![A-Za-z0-9_])([+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?)(?![A-Za-z0-9_])"

        for match in re.finditer(pattern, working_line):
            token = match.group(1).strip()
            if not token:
                continue

            is_pct = token.endswith("%")
            val = _safe_float_parse(token)
            if val is None:
                continue

            # Filter standard calendar years (e.g. 2020..2028) unless followed by % or decimal
            if not is_pct and "." not in token and 2000 <= val <= 2035:
                # Check if it was explicitly a year in context
                surrounding = working_line[max(0, match.start() - 15):min(len(working_line), match.end() + 15)]
                if any(w in surrounding.lower() for w in ["year", "since", "date", "202", "annual", "quarter"]):
                    continue

            # Get surrounding sentence context
            start_ctx = max(0, match.start() - 40)
            end_ctx = min(len(working_line), match.end() + 40)
            context_snippet = working_line[start_ctx:end_ctx].strip()

            claims.append(NumericClaim(
                raw_str=token,
                num_val=val,
                is_percentage=is_pct,
                context=context_snippet,
                line_number=line_idx
            ))

    return claims


def _collect_numbers_from_structure(data: Any, num_pool: Set[float], str_pool: Set[str]):
    """Recursively traverses arbitrary data structures and collects all numeric values and string representations."""
    if data is None:
        return
    if isinstance(data, (int, float)):
        if not math.isnan(data) and not math.isinf(data):
            f_val = float(data)
            num_pool.add(f_val)
            num_pool.add(round(f_val, 4))
            num_pool.add(round(f_val, 2))
            str_pool.add(str(data))
            str_pool.add(f"{data:.4f}".rstrip("0").rstrip("."))
            str_pool.add(f"{data:.2f}")
            if isinstance(data, int) or f_val.is_integer():
                str_pool.add(f"{int(f_val):,}")
    elif isinstance(data, str):
        s_clean = data.strip()
        str_pool.add(s_clean)
        parsed = _safe_float_parse(s_clean)
        if parsed is not None:
            num_pool.add(parsed)
            num_pool.add(round(parsed, 4))
            num_pool.add(round(parsed, 2))
    elif isinstance(data, dict):
        for k, v in data.items():
            # Also check numeric keys
            if isinstance(k, (int, float)):
                _collect_numbers_from_structure(k, num_pool, str_pool)
            elif isinstance(k, str) and _safe_float_parse(k) is not None:
                _collect_numbers_from_structure(k, num_pool, str_pool)
            _collect_numbers_from_structure(v, num_pool, str_pool)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            _collect_numbers_from_structure(item, num_pool, str_pool)


class GroundTruthMatcher:
    """Matches candidate numbers against indexed verified ground truth."""

    def __init__(self, verified_metrics: Dict[str, Any]):
        self.verified_metrics = verified_metrics or {}
        self.num_pool: Set[float] = set()
        self.str_pool: Set[str] = set()
        self._build_pool()

    def _build_pool(self):
        _collect_numbers_from_structure(self.verified_metrics, self.num_pool, self.str_pool)

    def is_match(self, claim: NumericClaim, tolerance: float = 1e-3) -> bool:
        """
        Determines if a numeric claim matches any ground-truth metric.
        Checks:
        1. Exact string match (e.g. '102,058', '0.8543', '85.4%')
        2. Float value match within tolerance
        3. Percentage conversions: e.g. claim '85.43%' (val 85.43) vs pool float 0.8543 (or 85.43)
        4. Integer rounding
        """
        val = claim.num_val
        raw = claim.raw_str

        # 1. Exact string match in pool
        if raw in self.str_pool or raw.replace(",", "") in self.str_pool:
            return True

        # 2. Direct float match
        for target in self.num_pool:
            if math.isclose(val, target, abs_tol=tolerance, rel_tol=0.01):
                return True
            # Check if claim was a percentage e.g. 85.4% (val=85.4) matching 0.854 in pool
            if claim.is_percentage:
                if math.isclose(val / 100.0, target, abs_tol=tolerance, rel_tol=0.01):
                    return True
            # Or pool had a percentage (85.4) and claim is decimal (0.854)
            if math.isclose(val * 100.0, target, abs_tol=tolerance, rel_tol=0.01):
                return True

        return False


def verify_report_numbers(report_text: str, verified_metrics: Dict[str, Any]) -> VerificationResult:
    """
    Deterministically extracts and verifies all numeric claims in the report against verified_metrics.
    
    Returns:
        VerificationResult with lists of verified and unverified claims, and formatted correction prompt if invalid.
    """
    claims = extract_numeric_claims(report_text)
    matcher = GroundTruthMatcher(verified_metrics)

    verified_list: List[NumericClaim] = []
    unverified_list: List[NumericClaim] = []

    for c in claims:
        if matcher.is_match(c):
            verified_list.append(c)
        else:
            unverified_list.append(c)

    is_valid = len(unverified_list) == 0

    correction_prompt = ""
    if not is_valid:
        mismatch_lines = []
        for u in unverified_list[:15]:  # limit to top 15 distinct issues
            mismatch_lines.append(f"- Line {u.line_number}: Claimed value `{u.raw_str}` in context: \"...{u.context}...\"")

        correction_prompt = (
            "AUTOMATED NUMERIC VERIFICATION FAILED:\n"
            "The following numeric values in your draft report DO NOT match the authoritative `VERIFIED_METRICS` ground truth:\n"
            + "\n".join(mismatch_lines) + "\n\n"
            "INSTRUCTION FOR REVISION:\n"
            "1. You MUST revise the draft report to fix these discrepancies.\n"
            "2. Replace each unverified number with the exact corresponding value from `VERIFIED_METRICS`.\n"
            "3. If a specific number or metric is NOT present anywhere in `VERIFIED_METRICS`, replace that claim with the phrase \"not verified in current run\".\n"
            "4. Do NOT compute, invent, or round new numbers."
        )

    return VerificationResult(
        is_valid=is_valid,
        total_claims_count=len(claims),
        verified_count=len(verified_list),
        unverified_count=len(unverified_list),
        verified_claims=verified_list,
        unverified_claims=unverified_list,
        correction_prompt=correction_prompt
    )


def attach_verification_warning(report_text: str, unverified_claims: List[NumericClaim]) -> str:
    """
    Attaches a prominent verification warning banner to the top of the report
    when unverified numeric claims remain after maximum correction turns.
    """
    if not unverified_claims:
        return report_text

    items_list = []
    for u in unverified_claims[:10]:
        items_list.append(f"- `{u.raw_str}` (Line {u.line_number}: \"{u.context}\")")

    warning_banner = (
        "> ⚠️ **Verification Notice: Unverified Numeric Claims Detected**\n"
        "> The following numeric values in this report could not be independently confirmed against the execution artifacts in `VERIFIED_METRICS`:\n"
        + "\n".join([f"> {item}" for item in items_list]) + "\n"
        "> *Readers should cross-check these specific figures against the generated data artifacts manifest.*\n\n"
    )

    return warning_banner + report_text
