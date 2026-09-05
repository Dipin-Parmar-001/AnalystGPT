"""
Artifact Validation Module - AnalystGPT

Validates that artifacts claimed by agents actually exist in the output directory.
Prevents hallucination where agents claim to have generated files that don't exist.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of artifact validation."""
    is_valid: bool
    missing_artifacts: List[str]
    existing_artifacts: List[str]
    fallback_detected: bool
    hallucination_warnings: List[str]
    report_structure_valid: bool
    report_structure_issues: List[str]


def extract_claimed_artifacts(agent_response: Dict[str, Any]) -> List[str]:
    """
    Extract claimed artifacts from agent response.

    Looks for:
    - generated_files field
    - generated_artifacts in summary_data
    - Any JSON field containing artifact filenames
    """
    claimed = []

    # Direct generated_files field
    if "generated_files" in agent_response:
        files = agent_response.get("generated_files", [])
        if isinstance(files, list):
            claimed.extend([f for f in files if isinstance(f, str)])

    # Check summary_data for generated_artifacts
    if "summary_data" in agent_response and agent_response["summary_data"]:
        summary = agent_response["summary_data"]
        if isinstance(summary, dict):
            # Check generated_artifacts field
            if "generated_artifacts" in summary:
                artifacts = summary.get("generated_artifacts", [])
                if isinstance(artifacts, list):
                    for art in artifacts:
                        if isinstance(art, str):
                            claimed.append(art)
                        elif isinstance(art, dict) and "file" in art:
                            claimed.append(art["file"])

            # Check saved_artifacts field (DataAnalystAgent)
            if "saved_artifacts" in summary:
                artifacts = summary.get("saved_artifacts", [])
                if isinstance(artifacts, list):
                    claimed.extend([f for f in artifacts if isinstance(f, str)])

            # Check generated_visualizations
            if "generated_visualizations" in summary:
                viz = summary.get("generated_visualizations", [])
                if isinstance(viz, list):
                    for v in viz:
                        if isinstance(v, dict) and "file" in v:
                            claimed.append(v["file"])
                        elif isinstance(v, str):
                            claimed.append(v)

            # Check model_artifacts (MLAgent)
            if "model_artifacts" in summary:
                artifacts = summary.get("model_artifacts", [])
                if isinstance(artifacts, list):
                    for art in artifacts:
                        if isinstance(art, dict) and "artifact" in art:
                            claimed.append(art["artifact"])

    # Check for generated_artifacts directly in response
    if "generated_artifacts" in agent_response:
        artifacts = agent_response.get("generated_artifacts", [])
        if isinstance(artifacts, list):
            for art in artifacts:
                if isinstance(art, str):
                    claimed.append(art)
                elif isinstance(art, dict) and "file" in art:
                    claimed.append(art["file"])

    # Deduplicate
    return list(set(claimed))


def detect_fallback_generation(agent_response: Dict[str, Any], output_dir: str) -> Tuple[bool, List[str]]:
    """
    Detect if fallback generation was used instead of actual artifact creation.

    Returns (fallback_detected, fallback_indicators)
    """
    fallback_indicators = []

    # No response-text keyword scanning: analysis prose routinely contains words
    # like "missing" ("no missing values"), causing false fallback flags.
    # Fallback is judged only by structural signals below.

    # Check if summary_data has fallback indicators
    summary = agent_response.get("summary_data", {})
    if isinstance(summary, dict):
        issues = summary.get("issues_and_caveats", [])
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, str):
                    if any(kw in issue.lower() for kw in ["fallback", "not generated", "not produced", "could not"]):
                        fallback_indicators.append(f"Summary issue: {issue[:100]}")

        # Check status
        if summary.get("status") == "tool_limit_reached":
            fallback_indicators.append("Agent hit tool limit before completing artifacts")

    # Check if expected artifacts are missing from output_dir (semantic match aware)
    claimed = extract_claimed_artifacts(agent_response)
    for artifact in claimed:
        artifact_path = os.path.join(output_dir, artifact)
        if not os.path.exists(artifact_path) and not find_artifact_semantic(output_dir, artifact):
            fallback_indicators.append(f"Claimed artifact missing: {artifact}")

    return len(fallback_indicators) > 0, fallback_indicators


def validate_report_structure(report: str) -> Tuple[bool, List[str]]:
    """
    Validate that the final report has the expected structure.

    Returns (is_valid, issues)
    """
    issues = []

    if not report or len(report.strip()) < 100:
        issues.append("Report is too short or empty")
        return False, issues

    # Check for required sections
    required_sections = [
        "Executive Summary",
        "Data Overview",
        "Machine Learning"  # Optional but expected if ML was run
    ]

    # More flexible check - look for section headers
    section_patterns = [
        r"##\s*Executive Summary",
        r"##\s*Data Overview",
        r"##\s*Data Quality",
        r"##\s*Machine Learning",
        r"##\s*Operational Caveats",
        r"##\s*Direct Recommendations",
        r"##\s*Generated Artifacts",
        r"# Analysis Report",
    ]

    found_sections = 0
    for pattern in section_patterns:
        if re.search(pattern, report, re.IGNORECASE):
            found_sections += 1

    if found_sections < 3:
        issues.append(f"Report missing expected sections (found {found_sections}/8)")

    # No placeholder-text searching: existence + minimum size, section and
    # content checks above are sufficient; prose legitimately contains words
    # like "TODO"/"N/A" in context (e.g. quoting data), causing false failures.

    # Check for actual numbers/metrics (not just text)
    # Look for patterns like "X%", "0.XX", "X.X%", etc.
    metric_pattern = r'\d+\.?\d*%|\d+\.\d+'
    metrics_found = re.findall(metric_pattern, report)
    if len(metrics_found) < 3:
        issues.append("Report lacks sufficient numeric metrics")

    # Check for artifact references
    artifact_refs = re.findall(r'[\w\-_]+\.(?:png|csv|json|pkl|md)', report)
    if len(artifact_refs) == 0:
        issues.append("Report doesn't reference any generated artifacts")

    is_valid = len(issues) == 0
    return is_valid, issues


# Known semantic equivalents: required artifact stem -> acceptable alternative stems.
# Agents sometimes save tuning results as 'tuning_results.json' instead of
# 'hyperparameter_tuning_summary.json', etc. Both carry the same content.
_ARTIFACT_SYNONYMS: Dict[str, List[str]] = {
    "hyperparameter_tuning_summary": ["tuning_results", "hpt_summary", "hpt_results"],
    "tuning_results": ["hyperparameter_tuning_summary", "hpt_summary", "hpt_results"],
    "cleaned_dataset": ["cleaned_data"],
    "cleaned_data": ["cleaned_dataset"],
    "analyst_summary": ["analysis_summary", "eda_summary"],
    "ml_summary": ["model_summary", "ml_results"],
}


def _normalize_artifact_stem(name: str) -> str:
    """Normalizes an artifact filename stem for semantic comparison:
    lowercases, strips extension, removes per-task suffixes and separators."""
    stem = os.path.splitext(name.lower().strip())[0]
    stem = re.sub(r"[_\-]?task[_\-]?\w*$", "", stem)  # analyst_summary_task_1 -> analyst_summary
    stem = re.sub(r"[_\-]+", "", stem)                # ignore separator variants
    return stem


def find_artifact_semantic(output_dir: str, artifact_name: str) -> Optional[str]:
    """
    Semantically locates an artifact in output_dir when the exact filename is absent.

    Agents legitimately save per-task variants (e.g. 'analyst_summary_task_1.json'
    instead of 'analyst_summary.json'). Rather than failing validation on a naming
    technicality, fall back to semantic matching:
      1. Exact filename match.
      2. Same extension + normalized stem equal to the required stem
         (so 'analyst_summary_task_1.json' satisfies 'analyst_summary.json').
      3. Same extension + stem matching a known synonym of the required stem
         (so 'tuning_results.json' satisfies 'hyperparameter_tuning_summary.json').

    Returns the matched filename, or None if nothing satisfies the requirement.
    """
    if not artifact_name or not os.path.isdir(output_dir):
        return None

    exact_path = os.path.join(output_dir, artifact_name)
    if os.path.exists(exact_path):
        return artifact_name

    want_ext = os.path.splitext(artifact_name.lower())[1]
    want_stem = _normalize_artifact_stem(artifact_name)
    if not want_stem:
        return None

    acceptable_stems = {want_stem}
    for base, synonyms in _ARTIFACT_SYNONYMS.items():
        if want_stem == _normalize_artifact_stem(base) or want_stem in [_normalize_artifact_stem(s) for s in synonyms]:
            acceptable_stems.update(_normalize_artifact_stem(x) for x in [base] + synonyms)

    try:
        candidates = os.listdir(output_dir)
    except OSError:
        return None

    for fname in sorted(candidates):
        f_ext = os.path.splitext(fname.lower())[1]
        if f_ext != want_ext:
            continue
        if _normalize_artifact_stem(fname) in acceptable_stems:
            return fname
    return None


def validate_artifacts(
    agent_response: Dict[str, Any],
    output_dir: str,
    agent_name: str,
    task_type: Optional[str] = None
) -> ValidationResult:
    """
    Comprehensive artifact validation for an agent's task completion.

    Args:
        agent_response: The response dict from the agent's run_task
        output_dir: Directory where artifacts should exist
        agent_name: Name of the agent (DataAnalystAgent, MLAgent, ReporterAgent)
        task_type: Optional task type for specific validation rules

    Returns:
        ValidationResult with validation details
    """
    claimed_artifacts = extract_claimed_artifacts(agent_response)
    existing_artifacts = []
    missing_artifacts = []

    # Check each claimed artifact (exact match first, then semantic per-task variant match)
    for artifact in claimed_artifacts:
        artifact_path = os.path.join(output_dir, artifact)
        if os.path.exists(artifact_path):
            existing_artifacts.append(artifact)
        else:
            semantic_match = find_artifact_semantic(output_dir, artifact)
            if semantic_match:
                existing_artifacts.append(artifact)
            else:
                missing_artifacts.append(artifact)

    # Check for fallback generation
    fallback_detected, fallback_indicators = detect_fallback_generation(agent_response, output_dir)

    # Agent-specific required artifacts
    required_artifacts = get_required_artifacts(agent_name, task_type)
    for required in required_artifacts:
        required_path = os.path.join(output_dir, required)
        if not os.path.exists(required_path) and required not in missing_artifacts:
            # Semantic fallback: a per-task variant (analyst_summary_task_1.json)
            # or known synonym (tuning_results.json) satisfies the requirement.
            if not find_artifact_semantic(output_dir, required):
                missing_artifacts.append(required)

    # Validate report structure if this is a reporter agent
    report_structure_valid = True
    report_structure_issues = []
    if agent_name == "ReporterAgent":
        report = agent_response.get("final_report", "") or agent_response.get("response", "")
        report_structure_valid, report_structure_issues = validate_report_structure(report)

    # Build hallucination warnings
    hallucination_warnings = []
    if missing_artifacts:
        hallucination_warnings.append(
            f"HALLUCINATION DETECTED: {agent_name} claimed to generate {len(missing_artifacts)} "
            f"artifact(s) that don't exist: {', '.join(missing_artifacts)}"
        )

    if fallback_detected:
        hallucination_warnings.append(
            f"FALLBACK DETECTED: {agent_name} appears to have used fallback generation: "
            f"{'; '.join(fallback_indicators[:3])}"
        )

    if not report_structure_valid:
        hallucination_warnings.append(
            f"INVALID REPORT STRUCTURE: {agent_name} generated report with issues: "
            f"{'; '.join(report_structure_issues[:3])}"
        )

    is_valid = len(missing_artifacts) == 0 and not fallback_detected and report_structure_valid

    return ValidationResult(
        is_valid=is_valid,
        missing_artifacts=missing_artifacts,
        existing_artifacts=existing_artifacts,
        fallback_detected=fallback_detected,
        hallucination_warnings=hallucination_warnings,
        report_structure_valid=report_structure_valid,
        report_structure_issues=report_structure_issues
    )


def get_required_artifacts(agent_name: str, task_type: Optional[str] = None) -> List[str]:
    """Get required artifacts for each agent type."""
    if agent_name == "DataAnalystAgent":
        if task_type == "cleaning":
            return ["cleaned_dataset.csv", "analyst_summary.json"]
        elif task_type == "feature_engineering":
            return ["feature_engineered_dataset.csv", "feature_engineering_summary.json"]
        elif task_type == "stats_and_viz":
            return ["analyst_summary.json"]
        else:
            return ["cleaned_dataset.csv", "analyst_summary.json"]

    elif agent_name == "MLAgent":
        if task_type == "model_comparison":
            return [
                "model_comparison.csv",
                "baseline_score.json",
                "feature_importance.csv",
                "confusion_matrix.json",
                "predictions.csv",
                "ml_summary.json",
                "model_registry.json",
                "champion_model.json"
            ]
        elif task_type == "hyperparameter_tuning":
            return [
                "tuned_model_comparison.csv",
                "best_params.json",
                "hyperparameter_tuning_summary.json"
            ]
        else:
            return [
                "model_comparison.csv",
                "ml_summary.json"
            ]

    elif agent_name == "ReporterAgent":
        return ["final_report.md"]

    return []


def build_hallucination_warning_prompt(
    validation_result: ValidationResult,
    agent_name: str,
    original_prompt: str,
    task_description: str
) -> str:
    """Build a warning prompt to re-prompt the agent for missing artifacts."""

    warning_parts = [
        f"⚠️ HALLUCINATION WARNING FOR {agent_name} ⚠️",
        "",
        "Your previous response claimed to generate artifacts that DO NOT EXIST in the output directory.",
        "This is a hallucination. You must actually create these files using the execute_python tool.",
        "",
        "VALIDATION RESULTS:",
    ]

    if validation_result.missing_artifacts:
        warning_parts.append(f"  ❌ MISSING ARTIFACTS (claimed but not found):")
        for art in validation_result.missing_artifacts:
            warning_parts.append(f"     - {art}")

    if validation_result.existing_artifacts:
        warning_parts.append(f"  ✅ EXISTING ARTIFACTS (verified):")
        for art in validation_result.existing_artifacts:
            warning_parts.append(f"     - {art}")

    if validation_result.fallback_detected:
        warning_parts.append(f"  🔄 FALLBACK DETECTED: The system detected fallback generation was used.")
        warning_parts.append(f"     You must actually execute code to create the real artifacts.")

    if not validation_result.report_structure_valid:
        warning_parts.append(f"  📝 REPORT STRUCTURE ISSUES:")
        for issue in validation_result.report_structure_issues:
            warning_parts.append(f"     - {issue}")

    warning_parts.extend([
        "",
        "REQUIRED ACTION:",
        "1. Use execute_python tool to ACTUALLY CREATE the missing artifacts",
        "2. Save files to the current working directory (they will be auto-captured)",
        "3. Verify each artifact exists by reading it back after saving",
        "4. Generate the required summary JSON (analyst_summary.json, ml_summary.json, etc.)",
        "5. Do NOT claim artifacts exist unless you have verified them with code execution",
        "",
        f"Original Task: {task_description}",
        "",
        "Please re-execute your task and ensure ALL claimed artifacts are actually generated."
    ])

    return "\n".join(warning_parts)


def validate_and_get_warning(
    agent_response: Dict[str, Any],
    output_dir: str,
    agent_name: str,
    task_description: str,
    task_type: Optional[str] = None
) -> Tuple[ValidationResult, Optional[str]]:
    """
    Validate artifacts and return warning prompt if validation fails.

    Returns:
        (validation_result, warning_prompt_or_none)
    """
    result = validate_artifacts(agent_response, output_dir, agent_name, task_type)

    if not result.is_valid:
        warning = build_hallucination_warning_prompt(result, agent_name, "", task_description)
        return result, warning

    return result, None


if __name__ == "__main__":
    # Quick test
    print("Artifact Validator Module")
    print("Available functions:")
    print("  - validate_artifacts()")
    print("  - extract_claimed_artifacts()")
    print("  - detect_fallback_generation()")
    print("  - validate_report_structure()")
    print("  - build_hallucination_warning_prompt()")
    print("  - validate_and_get_warning()")