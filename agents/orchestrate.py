"""
Orchestrator Module - AnalystGPT

Coordinates the multi-agent workflow:
1. PlannerAgent decomposes user queries and dataset context into a structured ExecutionPlan.
2. Dynamically routes each task in the plan to its designated specialized agent:
   - DataAnalystAgent: Data cleaning, EDA, statistical analysis, and visualizations.
   - MLAgent: Feature preprocessing, model training, metric evaluation, model_comparison.csv, predictions.csv.
   - ReporterAgent: Synthesizes findings, figures, and metrics into a stakeholder-ready executive markdown report.
3. Maintains shared sandbox environment, artifact aggregation, and comprehensive telemetry for UI and Logfire.
"""

import os
import sys
import json
import time
import copy
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import asdict

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from sandbox_execution import ToolManager, build_verified_metrics
from agents.tools import reset_global_sandbox_manager
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.planner_agent import PlannerAgent, ExecutionPlan, TaskItem, PlanStatus
from agents.data_analyst_agent import DataAnalystAgent
from agents.ml_agent import MLAgent
from agents.reporter_agent import ReporterAgent
from agents.cancellation import CancellationToken, WorkflowCancelledException, WorkflowPausedException
from agents.artifact_validator import (
    validate_artifacts,
    validate_and_get_warning,
    build_hallucination_warning_prompt,
    build_repair_prompt,
    build_champion_repair_prompt,
    is_champion_disagreement_error,
    ValidationResult
)
from agents.run_context import build_dataset_profile, render_dataset_profile_block
from agents.consistency_checker import run_consistency_checks, ConsistencyReport, count_by_severity
from agents.preflight_gate import run_preflight, verify_champion_consistency, _load_json_safe
from agents.champion_selector import select_initial_champion, evaluate_challenger, ChampionSelectionResult
from agents.fact_sheet import build_facts
from agents.observability import (
    get_recorder,
    set_task_context,
    reset_task_context,
    task_context,
)


try:
    import logfire
except ImportError:
    logfire = None


def _load_json_safe(path: str) -> Optional[Any]:
    """Safely load JSON file, returning None on missing file or parse error."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def build_post_verification_retry_prompt(unverified_errors: List[str]) -> str:
    """
    Retry prompt for post-verification failures on ReporterAgent tasks.

    The failure is about unverified numeric claims in final_report.md, not missing
    artifacts, so the generic hallucination-warning template would send the agent
    chasing nonexistent artifact problems. This prompt carries the specific
    unverified claims and the revision rules from the numeric verifier instead.
    """
    return (
        "⚠️ NUMERIC VERIFICATION WARNING FOR ReporterAgent ⚠️\n"
        "\n"
        "The previous draft report FAILED deterministic numeric post-verification.\n"
        "The following numeric claims DO NOT match the authoritative `VERIFIED_METRICS` ground truth:\n"
        + "\n".join(f"  - {err}" for err in unverified_errors) + "\n"
        "\n"
        "REQUIRED ACTION:\n"
        "1. Regenerate the final report with these discrepancies fixed.\n"
        "2. Replace each unverified number with the exact corresponding value from `VERIFIED_METRICS`.\n"
        "3. If a specific number or metric is NOT present anywhere in `VERIFIED_METRICS`, replace that claim with the phrase \"not verified in current run\".\n"
        "4. Do NOT compute, invent, or round new numbers.\n"
        "5. This is NOT a missing-artifact problem: all artifacts already exist. Do NOT re-create or re-save artifacts."
    )


def _is_tuning_task(task: Any) -> bool:
    """True when the task is a hyperparameter-tuning task (by declared type or title)."""
    return (
        getattr(task, "task_type", None) == "hyperparameter_tuning"
        or "hyperparameter" in str(getattr(task, "title", "")).lower()
        or "tuning" in str(getattr(task, "title", "")).lower()
        or "hpt" in str(getattr(task, "title", "")).lower()
    )


def _normalize_champion_pointer(output_dir: str) -> Optional[str]:
    """Rewrites champion_model.json's artifact reference to the exact on-disk
    filename when the recorded name is an unambiguous case/format variant of a
    real artifact (spaces vs underscores, missing or doubled .pkl suffix).

    Returns the corrected filename, or None when the pointer already matches a
    real artifact or no unambiguous match exists (in which case the stale-pointer
    consistency check fires as usual).
    """
    import re as _re

    champ_path = os.path.join(output_dir, "champion_model.json")
    if not os.path.exists(champ_path):
        return None
    try:
        with open(champ_path, encoding="utf-8") as f:
            champ = json.load(f)
    except Exception:
        return None
    if not isinstance(champ, dict):
        return None

    champ_artifact = champ.get("model") or champ.get("artifact_filename")
    if not champ_artifact:
        return None
    pkl_files = [f for f in os.listdir(output_dir) if f.lower().endswith(".pkl")]
    if champ_artifact in pkl_files:
        return None  # pointer is fine

    def _canon(name: str) -> str:
        name = name.strip().lower()
        if not name.endswith(".pkl"):
            name += ".pkl"
        name = _re.sub(r"[\s\-]+", "_", name)
        name = _re.sub(r"_*\.pkl$", ".pkl", name)
        return name

    target = _canon(champ_artifact)
    matches = [f for f in pkl_files if _canon(f) == target]
    if len(matches) != 1:
        # Also match by stripping underscores/non-alphanumerics completely
        clean_target = _re.sub(r"[^a-z0-9]", "", target[:-4] if target.endswith(".pkl") else target)
        matches = [f for f in pkl_files if _re.sub(r"[^a-z0-9]", "", f[:-4] if f.endswith(".pkl") else f) == clean_target]
    if len(matches) != 1:
        return None
    champ["model"] = matches[0]
    champ["artifact_filename"] = matches[0]
    champ["pointer_normalized_from"] = champ_artifact
    with open(champ_path, "w", encoding="utf-8") as f:
        json.dump(champ, f, indent=2)
    print(f"[Orchestrator] Normalized champion pointer '{champ_artifact}' -> '{matches[0]}'", flush=True)
    return matches[0]


class WorkflowOrchestrator:
    """
    Multi-Agent Workflow Orchestrator.
    Manages the lifecycle of planning, dynamic task delegation to specialized agents,
    shared sandbox state, artifact collection, and executive report synthesis.
    """

    def __init__(
        self,
        tool_manager: Optional[ToolManager] = None,
        planner_model: str = "nemotron-3-ultra-550b-a55b",
        analyst_model: str = "nemotron-3-ultra-550b-a55b",
        ml_model: str = "nemotron-3-ultra-550b-a55b",
        reporter_model: str = "nemotron-3-ultra-550b-a55b",
    ):
        setup_logfire()

        # Shared ToolManager across all sandbox-interacting agents
        self.tool_manager = tool_manager or ToolManager()

        # Initialize specialized agents with increased tool call budget (10 instead of 7)
        self.planner_agent = PlannerAgent(
            available_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
            model_name=planner_model
        )
        self.data_analyst_agent = DataAnalystAgent(
            model_name=analyst_model,
            tool_manager=self.tool_manager,
            max_tool_calls=10
        )
        self.ml_agent = MLAgent(
            model_name=ml_model,
            tool_manager=self.tool_manager,
            max_tool_calls=10
        )
        self.reporter_agent = ReporterAgent(
            model_name=reporter_model,
            tool_manager=self.tool_manager,
            max_tool_calls=10,
            enable_numeric_verifier=False,
        )

        self.current_plan: Optional[ExecutionPlan] = None
        self.completed_task_results: List[Dict[str, Any]] = []
        self.generated_artifacts: List[str] = []
        self.final_report: str = ""
        self.current_output_dir: str = os.path.join(WORKSPACE_ROOT, "outputs")
        self.cancellation_token: Optional[CancellationToken] = None
        self.last_pause_checkpoint: Optional[Dict[str, Any]] = None
        self.current_active_task: Optional[Any] = None

        # Persist model names for later use (e.g. the consistency judge defaults
        # to reporter_model when judge_model is not set).
        self.planner_model = planner_model
        self.analyst_model = analyst_model
        self.ml_model = ml_model
        self.reporter_model = reporter_model

        # Artifact validation settings
        self.max_validation_retries: int = 2  # Max times to re-prompt agent for missing artifacts

        # Post-report repair loop: max correction passes after the final report is
        # written. Each pass verifies with ALL methods (numeric verifier, consistency
        # checker, LLM judge), combines findings into ONE prompt, and repairs via a
        # single tool-free LLM call (no tool budget consumed).
        self.max_repair_attempts: int = 2

        # Hybrid verification & severity policy settings
        self.enable_consistency_judge: bool = True   # LLM prose-consistency judge (hybrid stack)
        self.enable_numeric_verifier: bool = False   # Deterministic numeric verifier (DISABLED per user request)
        self.judge_model: Optional[str] = None       # Defaults to reporter_model
        self.severity_thresholds: Dict[str, int] = {
            "max_critical_findings": 0,  # >0 critical findings => failed-consistency
        }
        self._consistency_judge_instance: Optional[Any] = None

        # Populated at the end of every execute_plan run (also persisted to disk)
        self.last_run_findings: List[Dict[str, Any]] = []
        self.last_run_manifest: Dict[str, Any] = {}
        self.consistency_report: Optional[Any] = None
        self.repair_attempts_used: int = 0
        # Basenames of the user's uploaded files, set at execute_plan start so the
        # final verification pass can resolve report references to raw uploads
        # (they live outside the run output dir).
        self.uploaded_files: List[str] = []

    def reset(self, keep_sandbox: bool = False):
        """Resets orchestrator state, memory, and optionally global sandbox."""
        self.current_plan = None
        self.completed_task_results = []
        self.generated_artifacts = []
        self.final_report = ""
        self.current_output_dir = os.path.join(WORKSPACE_ROOT, "outputs")
        self.cancellation_token = None
        self.last_pause_checkpoint = None
        self.current_active_task = None
        self.last_run_findings = []
        self.last_run_manifest = {}
        self.consistency_report = None
        self.repair_attempts_used = 0

        if not keep_sandbox and reset_global_sandbox_manager is not None:
            try:
                reset_global_sandbox_manager()
            except Exception as e:
                print(f"[Orchestrator Notice] Sandbox reset: {e}")

    def cancel_execution(self, reason: str = "User requested workflow termination") -> Dict[str, Any]:
        """
        Immediately signals cancellation to the running workflow.
        Halts agent tasks, tool calls, and LLM queries.
        """
        if self.cancellation_token:
            self.cancellation_token.cancel(reason=reason)
            return {
                "status": "cancelled",
                "message": f"Cancellation signal dispatched: {reason}",
                "plan_id": self.current_plan.plan_id if self.current_plan else None
            }
        return {
            "status": "idle",
            "message": "No active execution was running to cancel."
        }

    def pause_execution(self, reason: str = "User paused the workflow") -> Dict[str, Any]:
        """
        Immediately signals pause to the running workflow.
        Halts in-progress agent task, tool execution, or LLM wait immediately.
        The interrupted task is NOT marked as complete and will be re-executed upon resume.
        """
        active_tid = self.current_active_task.task_id if getattr(self, "current_active_task", None) else None
        if self.cancellation_token:
            self.cancellation_token.pause(
                reason=reason,
                step=f"Task #{active_tid}" if active_tid else "active step",
                task_id=active_tid
            )
            # Create a provisional checkpoint immediately so last_pause_checkpoint is non-null
            try:
                if self.current_plan and not self.last_pause_checkpoint:
                    paused_task_id = active_tid or 1
                    safe_plan_dict = None
                    try:
                        safe_plan_dict = self.current_plan.to_dict()
                    except Exception:
                        safe_plan_dict = {"plan_id": getattr(self.current_plan, "plan_id", "wf_plan"), "tasks": []}

                    safe_results = []
                    try:
                        for tr in list(self.completed_task_results or []):
                            if isinstance(tr, dict):
                                safe_results.append(dict(tr))
                            else:
                                safe_results.append(tr)
                    except Exception:
                        safe_results = []

                    self.last_pause_checkpoint = {
                        "workflow_id": getattr(self.current_plan, "plan_id", None) or f"wf_{int(time.time())}",
                        "status": "paused",
                        "paused_at": datetime.now().isoformat(),
                        "paused_task_id": paused_task_id,
                        "paused_step": f"Task #{paused_task_id}",
                        "reason": reason,
                        "user_query": getattr(self.current_plan, "query", ""),
                        "input_files": list(getattr(self, "uploaded_files", []) or []),
                        "output_dir": self.current_output_dir,
                        "plan": safe_plan_dict,
                        "completed_task_results": safe_results,
                        "generated_artifacts": list(self.generated_artifacts or []),
                        "final_report": self.final_report or "",
                    }
            except Exception as snap_err:
                print(f"[Orchestrator Warning] Provisional pause checkpoint creation skipped: {snap_err}")

            return {
                "status": "paused",
                "message": f"Pause signal dispatched: {reason}",
                "plan_id": self.current_plan.plan_id if self.current_plan else None,
                "task_id": active_tid
            }
        return {
            "status": "idle",
            "message": "No active execution was running to pause."
        }

    def save_checkpoint_to_disk(self, filepath: str, checkpoint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Saves workflow state checkpoint to a JSON file on disk (in workflows/).
        Allows user to shut down sessions and resume the paused workflow later.
        """
        data = checkpoint or self.last_pause_checkpoint
        if not data:
            if not self.current_plan:
                raise ValueError("No active plan or paused workflow to save.")
            data = {
                "workflow_id": self.current_plan.plan_id or f"wf_{int(time.time())}",
                "status": "paused",
                "paused_at": datetime.now().isoformat(),
                "paused_task_id": 1,
                "paused_step": "paused_state",
                "reason": "Manual save to disk",
                "user_query": self.current_plan.query,
                "input_files": getattr(self, "uploaded_files", []),
                "output_dir": self.current_output_dir,
                "plan": self.current_plan.to_dict(),
                "completed_task_results": copy.deepcopy(self.completed_task_results),
                "generated_artifacts": list(self.generated_artifacts),
                "final_report": self.final_report,
            }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[Orchestrator] Saved workflow checkpoint to {filepath}")
        return {
            "status": "saved",
            "filepath": filepath,
            "workflow_id": data.get("workflow_id"),
            "paused_task_id": data.get("paused_task_id")
        }

    def load_checkpoint_from_disk(self, filepath: str) -> Dict[str, Any]:
        """
        Loads saved workflow checkpoint from disk and restores state in orchestrator.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Workflow checkpoint file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.last_pause_checkpoint = data
        if "plan" in data and isinstance(data["plan"], dict):
            self.current_plan = ExecutionPlan.from_dict(data["plan"])
            self.current_plan.status = PlanStatus.PAUSED
        if "completed_task_results" in data:
            self.completed_task_results = data["completed_task_results"]
        if "generated_artifacts" in data:
            self.generated_artifacts = data["generated_artifacts"]
        if "final_report" in data:
            self.final_report = data["final_report"]
        if "output_dir" in data:
            self.current_output_dir = data["output_dir"]
            if hasattr(self.tool_manager, "set_output_dir"):
                self.tool_manager.set_output_dir(self.current_output_dir)
        if "input_files" in data:
            self.uploaded_files = data["input_files"]

        print(f"[Orchestrator] Loaded workflow checkpoint from {filepath} (paused at Task #{data.get('paused_task_id')})")
        return data

    def _get_consistency_judge(self):
        """Lazily instantiates the LLM consistency judge (hybrid verification layer)."""
        if self._consistency_judge_instance is not None:
            return self._consistency_judge_instance
        try:
            from agents.consistency_judge import ConsistencyJudge
            self._consistency_judge_instance = ConsistencyJudge(
                model_name=self.judge_model or self.reporter_model
            )
            return self._consistency_judge_instance
        except Exception as judge_err:
            logfire.warn("ConsistencyJudge unavailable; judge layer disabled", error=str(judge_err))
            return None

    def _run_final_verification(self, query_text: str, active_plan) -> Dict[str, Any]:
        """
        Full verification pass over the current final report using ALL methods:
        1. Deterministic numeric verification (report claims vs VERIFIED_METRICS) - DISABLED by default
        2. Deterministic cross-artifact consistency checks
        3. LLM prose-consistency judge (skipped if unavailable)

        Returns {"findings": [..], "verified_metrics": {..}, "numeric": {..}} where
        findings is the merged list consumed by the repair loop and severity policy.
        """
        with logfire.span("Orchestrator: Final Verification Pass (numeric:DISABLED + consistency + judge)",
                          report_length=len(self.final_report or "")):
            findings: List[Dict[str, Any]] = []

            # Ground-truth metrics for numeric verification
            verified_metrics: Dict[str, Any] = {}
            vm_path = os.path.join(self.current_output_dir, "verified_metrics.json")
            if os.path.exists(vm_path):
                try:
                    with open(vm_path, "r", encoding="utf-8", errors="replace") as f:
                        verified_metrics = json.load(f)
                except Exception as vm_err:
                    logfire.warn("Orchestrator: could not load verified_metrics.json; numeric gate will be skipped",
                                 path=vm_path, error=str(vm_err))
                    verified_metrics = {}
            else:
                logfire.warn("Orchestrator: verified_metrics.json not found; numeric verification skipped",
                             output_dir=self.current_output_dir)

            # 1. Numeric verification of the report itself (DISABLED - controlled by enable_numeric_verifier flag)
            numeric_summary: Dict[str, Any] = {"enabled": False}
            if False and self.enable_numeric_verifier and self.final_report and verified_metrics:
                try:
                    from agents.numeric_verifier import verify_report_numbers
                    numeric_res = verify_report_numbers(self.final_report, verified_metrics)
                    numeric_summary = {
                        "enabled": True,
                        "is_valid": numeric_res.is_valid,
                        "total_claims": numeric_res.total_claims_count,
                        "verified_count": numeric_res.verified_count,
                        "unverified_count": numeric_res.unverified_count,
                        "unverified_samples": [c.raw_str for c in numeric_res.unverified_claims[:10]],
                    }
                    logfire.info(
                        "Orchestrator: numeric verification finished",
                        is_valid=numeric_res.is_valid,
                        total_claims=numeric_res.total_claims_count,
                        verified_count=numeric_res.verified_count,
                        unverified_count=numeric_res.unverified_count,
                    )
                    if not numeric_res.is_valid:
                        for c in numeric_res.unverified_claims[:10]:
                            findings.append({
                                "severity": "minor",
                                "code": "unverified_numeric_claim",
                                "message": f"Numeric claim '{c.raw_str}' in section '{c.section or '?'}' is not grounded in VERIFIED_METRICS (context: \"{c.context[:80]}\").",
                                "artifacts": ["final_report.md"],
                                "source": "numeric_verifier",
                            })
                except Exception as num_err:
                    logfire.error("Orchestrator: numeric verification failed", error=str(num_err))
            else:
                logfire.info("Orchestrator: numeric verification SKIPPED (enable_numeric_verifier=False)")

            # 1b. Normalize the champion pointer against the artifacts actually on
            # disk. Agents occasionally record a display name (e.g.
            # "Gradient Boosting_p0.pkl") instead of the exact saved filename; if a
            # deterministic, unambiguous match exists, correct the pointer so the
            # stale-pointer check judges real staleness, not formatting drift.
            try:
                _normalize_champion_pointer(self.current_output_dir)
            except Exception as champ_err:
                logfire.warn("Orchestrator: champion pointer normalization failed", error=str(champ_err))

            # 2. Deterministic cross-artifact consistency checks
            try:
                consistency_report = run_consistency_checks(
                    self.current_output_dir,
                    report_text=self.final_report,
                    # Raw uploads live outside the run dir; the checker must know
                    # them so referencing the source dataset isn't flagged as a
                    # missing artifact.
                    uploaded_files=self.uploaded_files or None,
                    persist=True,
                    dataset_type=getattr(self.current_plan, "dataset_type", None),
                )
                self.consistency_report = consistency_report
                findings.extend([asdict(f) for f in consistency_report.findings])
                logfire.info(
                    "Orchestrator: consistency checks finished",
                    is_valid=consistency_report.is_valid,
                    **consistency_report.counts(),
                )
            except Exception as check_err:
                logfire.error("Orchestrator: consistency checks failed", error=str(check_err))

            # 3. LLM prose-consistency judge
            if self.enable_consistency_judge and self.final_report:
                judge = self._get_consistency_judge()
                if judge is not None:
                    try:
                        from agents.consistency_judge import judge_result_to_findings
                        judge_res = judge.judge_report(
                            report_text=self.final_report,
                            verified_metrics=verified_metrics,
                            consistency_findings=findings,
                        )

                        # Persist judge output for auditability
                        try:
                            with open(os.path.join(self.current_output_dir, "judge_result.json"), "w", encoding="utf-8") as jf:
                                json.dump(judge_res.to_dict(), jf, indent=2, default=str)
                        except Exception:
                            pass

                        findings.extend(judge_result_to_findings(judge_res))
                        if not judge_res.parse_ok:
                            # A judge that produced no verdict must be visible in
                            # the run findings — silently skipping it makes the
                            # hybrid stack look like it verified prose when it didn't.
                            findings.append({
                                "severity": "minor",
                                "code": "judge_failed",
                                "message": f"LLM consistency judge produced no parsable verdict ({judge_res.error}); "
                                           f"prose-level report claims were NOT judged this run. Only the deterministic "
                                           f"numeric and consistency gates ran.",
                                "artifacts": ["final_report.md", "judge_result.json"],
                                "source": "consistency_judge",
                            })
                        logfire.info(
                            "Orchestrator: consistency judge finished",
                            parse_ok=judge_res.parse_ok,
                            claims_judged=len(judge_res.claims),
                            contradicted=len(judge_res.contradicted),
                            unverifiable=len(judge_res.unverifiable),
                            discarded_by_guardrails=judge_res.discarded_claims,
                        )
                    except Exception as judge_err:
                        logfire.error("Orchestrator: consistency judge failed", error=str(judge_err))
                else:
                    logfire.warn("Orchestrator: consistency judge skipped (judge unavailable)")

            logfire.info(
                "Orchestrator: final verification pass complete",
                total_findings=len(findings),
                critical_findings=sum(1 for f in findings if f.get("severity") == "critical"),
                minor_findings=sum(1 for f in findings if f.get("severity") == "minor"),
            )

            self.last_run_findings = findings
            return {
                "findings": findings,
                "verified_metrics": verified_metrics,
                "numeric": numeric_summary,
            }

    def _repair_final_report(self, verification: Dict[str, Any], files_list: List[str],
                             step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Post-report repair loop. While critical findings remain and attempts are
        left: combine ALL findings (numeric + consistency + judge) into ONE
        correction prompt and repair via a compact agent turn with a dedicated
        5-call sandbox budget, then re-verify with all methods. Returns the
        updated verification dictionary.
        """
        def _cb(event: str, data: Dict[str, Any]):
            if step_callback:
                try:
                    step_callback(event, data)
                except BaseException:
                    pass

        repair_attempts_used = 0
        while repair_attempts_used < self.max_repair_attempts:
            critical = [f for f in verification.get("findings", []) if f.get("severity") == "critical"]
            if not critical:
                break

            repair_attempt_no = repair_attempts_used + 1
            get_recorder().increment("repair.attempts")
            get_recorder().log_event(
                "report_repair_attempt",
                attempt=repair_attempt_no,
                max_attempts=self.max_repair_attempts,
                critical_count=len(critical),
            )
            _cb("report_repair_started", {
                "attempt": repair_attempt_no,
                "max_attempts": self.max_repair_attempts,
                "critical_count": len(critical),
            })

            print(f"[Orchestrator] Report repair attempt {repair_attempt_no}/{self.max_repair_attempts} "
                  f"({len(critical)} critical finding(s))...")
            logfire.info(
                "Orchestrator: report repair attempt started",
                attempt=repair_attempt_no,
                max_attempts=self.max_repair_attempts,
                critical_count=len(critical),
                critical_codes=[f.get("code") for f in critical[:10]],
            )

            try:
                profile_block = render_dataset_profile_block(
                    build_dataset_profile(self.current_output_dir, files_list, persist=False)
                )
            except Exception as profile_err:
                logfire.warn("Orchestrator: could not build dataset profile for repair prompt", error=str(profile_err))
                profile_block = ""

            try:
                corrected = self.reporter_agent.repair_report(
                    current_report=self.final_report,
                    findings=verification.get("findings", []),
                    verified_metrics=verification.get("verified_metrics") or {},
                    profile_block=profile_block,
                    max_tool_calls=5,
                    cancellation_token=self.cancellation_token,
                    step_callback=step_callback,
                )
            except Exception as repair_err:
                logfire.error("Orchestrator: report repair encountered exception", error=str(repair_err))
                print(f"[Orchestrator Error] Report repair failed: {repair_err}")
                break

            if not corrected or corrected == self.final_report:
                logfire.warn("Orchestrator: repair produced no changed report; stopping repair loop",
                             attempt=repair_attempt_no)
                print("[Orchestrator] Repair produced no changed report; stopping repair loop.")
                break

            self.final_report = corrected
            try:
                with open(os.path.join(self.current_output_dir, "final_report.md"), "w", encoding="utf-8") as rf:
                    rf.write(self.final_report)
            except Exception as write_err:
                logfire.warn("Orchestrator: failed to save repaired report", error=str(write_err))

            repair_attempts_used += 1
            logfire.info(
                "Orchestrator: report repair attempt completed",
                attempt=repair_attempt_no,
                report_length=len(self.final_report),
            )
            _cb("report_repair_completed", {
                "attempt": repair_attempt_no,
                "report_length": len(self.final_report),
            })

            # Re-verify the repaired report with ALL methods and update verification in-place
            new_verification = self._run_final_verification(query_text="", active_plan=None)
            verification.clear()
            verification.update(new_verification)

        self.repair_attempts_used = repair_attempts_used
        return verification

    def _classify_run_status(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Severity-based failure policy (user-confirmed decision):
        - fatal task errors                -> "failed"
        - critical findings (post-retry)   -> "failed-consistency": findings surfaced
                                              instead of a polished report
        - minor findings only              -> "completed_with_warnings"
        - clean                            -> "completed"
        """
        failed_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "failed"]
        validation_failed_tasks = [tr for tr in self.completed_task_results if tr.get("validation_failed", False)]
        critical_findings = [f for f in findings if f.get("severity") == "critical"]
        minor_findings = [f for f in findings if f.get("severity") == "minor"]

        max_critical = int(self.severity_thresholds.get("max_critical_findings", 0))

        if failed_tasks:
            status = "failed"
            message = f"Workflow failed: {len(failed_tasks)} task(s) failed with errors."
        elif len(critical_findings) > max_critical:
            status = "failed-consistency"
            message = (
                f"Workflow produced {len(critical_findings)} critical consistency finding(s) that "
                f"survived retries. The report is NOT cleared for delivery; review the findings."
            )
        elif validation_failed_tasks:
            status = "failed-consistency"
            message = f"Workflow completed but {len(validation_failed_tasks)} task(s) failed artifact validation."
        elif minor_findings:
            status = "completed_with_warnings"
            message = f"Workflow completed with {len(minor_findings)} minor consistency warning(s). See findings."
        else:
            status = "completed"
            message = "Workflow execution completed successfully; all verification gates passed."

        result = {
            "status": status,
            "message": message,
            "counts": {
                "critical_findings": len(critical_findings),
                "minor_findings": len(minor_findings),
                "tasks_failed": len(failed_tasks),
                "tasks_validation_failed": len(validation_failed_tasks),
            },
        }

        logfire.info(
            "Orchestrator: run status classified",
            status=status,
            message=message,
            **result["counts"],
        )
        return result

    def _collect_agent_summaries(self, output_dir: str) -> Dict[str, Any]:
        """
        Collects structured JSON summaries generated across all Data Analyst and ML tasks.
        Returns a dictionary containing 'analyst_summaries' and 'ml_summaries'.
        """
        analyst_summaries: List[Dict[str, Any]] = []
        ml_summaries: List[Dict[str, Any]] = []
        seen_analyst_tasks = set()
        seen_ml_tasks = set()

        # 1. First, check completed task results in memory
        for tr in self.completed_task_results:
            agent_type = (tr.get("agent") or "").lower()
            sum_data = tr.get("summary_data")
            if sum_data and isinstance(sum_data, dict):
                tid = sum_data.get("task_id") or tr.get("task_id")
                if "analyst" in agent_type and tid not in seen_analyst_tasks:
                    analyst_summaries.append(sum_data)
                    seen_analyst_tasks.add(tid)
                elif "ml" in agent_type and tid not in seen_ml_tasks:
                    ml_summaries.append(sum_data)
                    seen_ml_tasks.add(tid)

        # 2. Scan output directory for saved summary JSON files
        if os.path.exists(output_dir):
            for root, _, files in os.walk(output_dir):
                for fname in sorted(files):
                    base_lower = fname.lower().strip()
                    if not base_lower.endswith(".json"):
                        continue

                    full_path = os.path.join(root, fname)

                    if "analyst_summary" in base_lower:
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                                data = json.load(f)
                            tid = data.get("task_id", fname)
                            if tid not in seen_analyst_tasks:
                                analyst_summaries.append(data)
                                seen_analyst_tasks.add(tid)
                        except Exception as err:
                            print(f"[Orchestrator Notice] Error reading {full_path}: {err}")

                    elif "ml_summary" in base_lower:
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                                data = json.load(f)
                            tid = data.get("task_id", fname)
                            if tid not in seen_ml_tasks:
                                ml_summaries.append(data)
                                seen_ml_tasks.add(tid)
                        except Exception as err:
                            print(f"[Orchestrator Notice] Error reading {full_path}: {err}")

        # 3. Fallback: If no analyst summary found, synthesize from task results
        if not analyst_summaries:
            for tr in self.completed_task_results:
                if "analyst" in (tr.get("agent") or "").lower():
                    analyst_summaries.append({
                        "task_id": tr.get("task_id"),
                        "title": tr.get("title"),
                        "status": tr.get("status"),
                        "summary": tr.get("output", "")[:400]
                    })

        # 4. Fallback: If no ML summary found, synthesize from task results
        if not ml_summaries:
            for tr in self.completed_task_results:
                if "ml" in (tr.get("agent") or "").lower():
                    ml_summaries.append({
                        "task_id": tr.get("task_id"),
                        "title": tr.get("title"),
                        "status": tr.get("status"),
                        "summary": tr.get("output", "")[:400]
                    })

        return {
            "analyst_summaries": analyst_summaries,
            "ml_summaries": ml_summaries
        }

    def create_plan(
        self,
        user_query: str,
        dataset_context: Optional[str] = None,
        files: Optional[List[str]] = None,
        max_tool_calls_per_task: Optional[int] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> ExecutionPlan:
        """
        Invokes PlannerAgent to generate a structured execution plan.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name="PlannerAgent initialization")

        if step_callback:
            step_callback("planning_started", {"user_query": user_query, "files": files})

        rec = get_recorder()
        rec.ensure_run(source="create_plan")

        # Global counter: total time to create the plan.
        plan_t0 = time.perf_counter()
        with task_context(agent="PlannerAgent", phase="planning"):
            with rec.step_timer("plan_creation"):
                plan = self.planner_agent.create_plan(
                    user_query=user_query,
                    dataset_context=dataset_context,
                    max_tool_calls_per_task=max_tool_calls_per_task,
                    step_callback=step_callback,
                    cancellation_token=cancellation_token
                )
        rec.add_duration("plan_creation_time_s", time.perf_counter() - plan_t0)
        self.current_plan = plan
        plan_dict = plan.to_dict()
        key_metrics = plan_dict.get("key_metrics", [])
        primary_metric = plan_dict.get("primary_metric")
        rec.log_event(
            "plan_created",
            plan_id=plan.plan_id,
            task_count=len(plan.tasks),
            selected_agents=plan.selected_agents,
            key_metrics=key_metrics,
            primary_metric=primary_metric,
            plan_duration_s=round(time.perf_counter() - plan_t0, 6),
        )

        if step_callback:
            step_callback("plan_created", {
                "plan": plan_dict,
                "plan_id": plan.plan_id,
                "selected_agents": plan.selected_agents,
                "tasks": [t.to_dict() for t in plan.tasks],
                "notes": plan.notes,
                "key_metrics": key_metrics,
                "primary_metric": primary_metric
            })

        return plan

    def modify_plan(
        self,
        modification_prompt: str,
        plan: Optional[ExecutionPlan] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> ExecutionPlan:
        """
        Modifies an existing execution plan based on user feedback.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name="PlannerAgent modifying plan")

        target_plan = plan or self.current_plan
        if not target_plan:
            raise ValueError("No active plan to modify.")

        get_recorder().ensure_run(source="modify_plan")
        with task_context(agent="PlannerAgent", phase="plan_modification"):
            with get_recorder().step_timer("plan_modification"):
                updated_plan = self.planner_agent.modify_plan(
                    target_plan,
                    modification_prompt,
                    step_callback=step_callback,
                    cancellation_token=cancellation_token
                )
        self.current_plan = updated_plan

        if step_callback:
            step_callback("plan_modified", {
                "plan": updated_plan.to_dict(),
                "revision_count": updated_plan.revision_count,
                "tasks": [t.to_dict() for t in updated_plan.tasks]
            })

        return updated_plan

    def confirm_plan(self, plan: Optional[ExecutionPlan] = None) -> ExecutionPlan:
        """Confirms the plan as ready for execution."""
        target_plan = plan or self.current_plan
        if not target_plan:
            raise ValueError("No active plan to confirm.")
        return self.planner_agent.confirm_plan(target_plan)

    def cancel_plan(self, plan: Optional[ExecutionPlan] = None) -> Dict[str, Any]:
        """Cancels the plan."""
        target_plan = plan or self.current_plan
        if not target_plan:
            return {"status": "cancelled", "message": "No active plan."}
        return self.planner_agent.cancel_plan(target_plan)

    def _resolve_agent(self, agent_name: str) -> Optional[Any]:
        """Maps an assigned agent string from the plan to its concrete agent instance."""
        norm = (agent_name or "").lower().replace(" ", "").replace("_", "")
        if "dataanalyst" in norm or "analyst" in norm or "cleaning" in norm or "visualization" in norm:
            return self.data_analyst_agent
        elif "ml" in norm or "machinelearning" in norm or "model" in norm:
            return self.ml_agent
        elif "reporter" in norm or "report" in norm or "synthesis" in norm:
            return self.reporter_agent
        return None

    def execute_plan(
        self,
        plan: Optional[ExecutionPlan] = None,
        user_query: Optional[str] = None,
        files: Optional[List[str]] = None,
        max_tool_calls_per_agent: int = 10,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        resume_from_task_id: Optional[int] = None,
        resume_checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes each task in the plan using its assigned specialized agent.
        Supports instant cancellation, pausing/resuming, explicit failure/completion telemetry,
        and artifact preservation.
        """
        from datetime import datetime

        if resume_checkpoint and not plan and "plan" in resume_checkpoint:
            active_plan = ExecutionPlan.from_dict(resume_checkpoint["plan"])
            self.current_plan = active_plan
        else:
            active_plan = plan or self.current_plan

        if not active_plan:
            raise ValueError("No active plan provided for execution.")

        query_text = user_query or (resume_checkpoint.get("user_query") if resume_checkpoint else None) or active_plan.query
        files_list = files or (resume_checkpoint.get("input_files") if resume_checkpoint else None) or []
        self.uploaded_files = list(files_list)

        # One run = one JSONL file in logs/. If create_plan already started the
        # run (normal flow), this reuses it.
        get_recorder().ensure_run(source="execute_plan")

        # Setup cancellation token
        self.cancellation_token = cancellation_token or CancellationToken()

        # Handle directory and state retention for resume vs fresh run
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if resume_checkpoint:
            self.current_output_dir = resume_checkpoint.get("output_dir", self.current_output_dir)
            if self.current_output_dir:
                dir_name = os.path.basename(self.current_output_dir)
                if dir_name.startswith("run_"):
                    run_timestamp = dir_name[4:]
            if resume_from_task_id is None:
                resume_from_task_id = resume_checkpoint.get("paused_task_id", 1)
            # Retain only results completed strictly before resume_from_task_id
            self.completed_task_results = [
                tr for tr in resume_checkpoint.get("completed_task_results", [])
                if tr.get("task_id", 0) < (resume_from_task_id or 1)
            ]
            self.generated_artifacts = list(resume_checkpoint.get("generated_artifacts", []))
            self.final_report = resume_checkpoint.get("final_report", "")
            os.makedirs(self.current_output_dir, exist_ok=True)
            print(f"[Orchestrator] Resuming workflow from checkpoint at Task #{resume_from_task_id} in {self.current_output_dir}")
        elif resume_from_task_id is not None:
            # Resuming in-memory: preserve output dir, keep prior completed tasks
            if self.current_output_dir:
                dir_name = os.path.basename(self.current_output_dir)
                if dir_name.startswith("run_"):
                    run_timestamp = dir_name[4:]
            self.completed_task_results = [
                tr for tr in self.completed_task_results
                if tr.get("task_id", 0) < resume_from_task_id
            ]
            os.makedirs(self.current_output_dir, exist_ok=True)
            print(f"[Orchestrator] Resuming in-memory workflow at Task #{resume_from_task_id} in {self.current_output_dir}")
        else:
            # Fresh execution: create new unique run output directory
            self.current_output_dir = os.path.join(WORKSPACE_ROOT, "outputs", f"run_{run_timestamp}")
            os.makedirs(self.current_output_dir, exist_ok=True)
            self.completed_task_results = []
            self.generated_artifacts = []
            self.final_report = ""

        if hasattr(self.tool_manager, "set_output_dir"):
            self.tool_manager.set_output_dir(self.current_output_dir)

        # Authoritative dataset profile: rebuilt from disk before every task
        # dispatch so agents never re-derive shapes/columns from memory.
        dataset_profile: Dict[str, Any] = build_dataset_profile(self.current_output_dir, files_list)

        def _refresh_dataset_profile() -> str:
            nonlocal dataset_profile
            try:
                dataset_profile = build_dataset_profile(self.current_output_dir, files_list)
            except Exception as profile_err:
                print(f"[Orchestrator Notice] dataset profile refresh failed: {profile_err}")
            return render_dataset_profile_block(dataset_profile)

        def _safe_callback(event: str, data: Dict[str, Any]):
            if step_callback:
                try:
                    step_callback(event, data)
                except BaseException:
                    pass

        def _execute_workflow():
            if HAS_LOGFIRE:
                logfire.info(
                    "Starting Execution of Multi-Agent Plan",
                    plan_id=active_plan.plan_id,
                    task_count=len(active_plan.tasks),
                    selected_agents=active_plan.selected_agents,
                    input_files=files_list,
                    output_dir=self.current_output_dir
                )

            _safe_callback("plan_execution_started", {
                "plan_id": active_plan.plan_id,
                "total_tasks": len(active_plan.tasks),
                "selected_agents": active_plan.selected_agents,
                "output_dir": self.current_output_dir,
                "resumed_from_task": resume_from_task_id
            })

            if resume_from_task_id is None:
                self.completed_task_results = []
            has_reporter_task = False
            current_active_task: Optional[TaskItem] = None

            try:
                # Check cancellation at start
                self.cancellation_token.check_cancelled(step_name="Workflow start")

                for task in active_plan.tasks:
                    # If resuming, skip tasks strictly before resume_from_task_id that were already completed
                    if resume_from_task_id is not None and task.task_id < resume_from_task_id:
                        if task.status == "completed" or any(tr.get("task_id") == task.task_id for tr in self.completed_task_results):
                            task.status = "completed"
                            print(f"[Orchestrator] Skipping already completed Task #{task.task_id}: {task.title}")
                            continue

                    # If resuming, ensure tasks after the resumed task remain pending
                    if resume_from_task_id is not None and task.task_id > resume_from_task_id:
                        if task.status in ["in_progress", "paused"]:
                            task.status = "pending"

                    self.current_active_task = task
                    current_active_task = task
                    # Check cancellation or pause before starting each task
                    self.cancellation_token.check_cancelled(
                        step_name=f"Before Task #{task.task_id}: {task.title}",
                        task_id=task.task_id
                    )

                    agent_instance = self._resolve_agent(task.assigned_agent)
                    agent_class_name = agent_instance.__class__.__name__ if agent_instance else task.assigned_agent

                    _safe_callback("task_execution_start", {
                        "task_id": task.task_id,
                        "title": task.title,
                        "assigned_agent": agent_class_name,
                        "description": task.description,
                        "primary_metric": task.primary_metric
                    })

                    task.status = "in_progress"

                    # Per-task latency + LLM/tool event attribution
                    task_t0 = time.perf_counter()
                    task_ctx_token = set_task_context(
                        task_id=task.task_id, agent=agent_class_name, phase="task", title=task.title
                    )

                    def _execute_task():
                        nonlocal has_reporter_task
                        nonlocal agent_class_name

                        # ================================================================
                        # TASK EXECUTION WITH COMBINED VALIDATION & SHARED RETRY LOGIC
                        # ================================================================
                        # We use a while loop with a shared max retry count (2) to prevent
                        # infinite loops. For ReporterAgent tasks, BOTH checks — artifact
                        # validation and the deterministic numeric post-verification — are
                        # evaluated on the same attempt, then ONE combined retry budget is
                        # spent (never 2 retries per check). On any failure, task status is
                        # set back to "pending" and re-executed with the relevant errors +
                        # extra tool budget. After max combined retries, we mark
                        # validation_failed and continue workflow.
                        #
                        # Errors from the immediately previous attempt are stored in
                        # last_validation_errors and passed to the next retry. This ensures the
                        # agent receives specific feedback about what failed, rather than relying
                        # on completed_task_results which is only updated after all retries.

                        max_retries = self.max_validation_retries  # 2
                        attempt = 0
                        validation_passed = False
                        task_validation_failed = False  # True if task exhausted retries without passing
                        last_validation_errors = []  # Errors from the immediately previous attempt
                        last_failure_kind = None  # "artifact_validation" or "post_verification"
                        last_missing_artifacts: list = []  # Missing artifacts from the previous attempt (drives targeted repair)
                        last_fallback_detected = False  # Fallback flag from the previous attempt

                        # Snapshot pre-task champion before ML task runs (e.g. initial champion before HPT)
                        pre_ml_champion = None
                        if isinstance(agent_instance, MLAgent) or getattr(task, "task_type", None) in ("model_comparison", "hyperparameter_tuning"):
                            champ_file_path = os.path.join(self.current_output_dir, "champion_model.json")
                            if os.path.isfile(champ_file_path):
                                pre_ml_champion = _load_json_safe(champ_file_path)

                        while attempt <= max_retries and not validation_passed and not task_validation_failed:
                            _attempt_t0 = time.perf_counter()
                            if attempt > 0:
                                # Re-prompt: set status back to pending and add hallucination warning
                                task.status = "pending"
                                print(f"[Orchestrator] Retry attempt {attempt}/{max_retries} for Task #{task.task_id} ({agent_class_name})")

                                get_recorder().log_event(
                                    "task_retry",
                                    task_id=task.task_id,
                                    title=task.title,
                                    agent=agent_class_name,
                                    attempt=attempt,
                                )
                                get_recorder().increment("retries.validation")

                                if HAS_LOGFIRE:
                                    logfire.info(
                                        "Retrying task after artifact validation failure",
                                        task_id=task.task_id,
                                        agent=agent_class_name,
                                        attempt=attempt,
                                        max_retries=max_retries
                                    )

                            # Tool call budget: fixed per task. Validation retries do
                            # NOT grant extra calls — the agent must fit its corrected
                            # work inside the same budget (user-confirmed policy).
                            # Hyperparameter tuning needs a larger budget: grid search,
                            # tuned-vs-baseline comparison, champion repoint, and artifact
                            # verification do not fit reliably in the standard allowance.
                            current_tool_budget = max_tool_calls_per_agent
                            if _is_tuning_task(task):
                                current_tool_budget = max(15, max_tool_calls_per_agent)

                            # Rebuild the authoritative dataset profile from disk for this
                            # attempt so prompts always carry the current ground truth.
                            profile_block = _refresh_dataset_profile()

                            # Use validation errors from the immediately previous attempt
                            validation_errors = last_validation_errors

                            # Execute the task based on agent type
                            agent_res = None
                            task_out = ""

                            if isinstance(agent_instance, ReporterAgent):
                                has_reporter_task = True
                                summaries = self._collect_agent_summaries(self.current_output_dir)
                                try:
                                    verified_metrics = build_verified_metrics(self.current_output_dir, uploaded_files=files_list)
                                    if "verified_metrics.json" not in self.generated_artifacts:
                                        self.generated_artifacts.append("verified_metrics.json")
                                except Exception as vm_err:
                                    print(f"[Orchestrator Notice] build_verified_metrics error: {vm_err}")
                                    verified_metrics = {}

                                # Preflight gate (checks + auto-repair; warn-not-stop) and the
                                # code-built fact sheet the reporter's {{blanks}} reference.
                                # Only on the first attempt: retries reuse the repaired state.
                                fact_sheet = None
                                if attempt == 0:
                                    try:
                                        preflight = run_preflight(self.current_output_dir, goal_text=query_text, repair=True)
                                        if preflight.get("repairs_applied"):
                                            print(f"[Orchestrator] Preflight repairs applied: {preflight['repairs_applied']}", flush=True)
                                        elif preflight.get("repairs"):  # backward compat
                                            print(f"[Orchestrator] Preflight repairs applied: {preflight['repairs']}", flush=True)
                                        # Refresh verified metrics after possible artifact repairs
                                        verified_metrics = build_verified_metrics(self.current_output_dir, uploaded_files=files_list)
                                        fact_sheet = build_facts(self.current_output_dir, verified_metrics, preflight)
                                    except Exception as pre_err:
                                        print(f"[Orchestrator Notice] preflight/fact sheet skipped: {pre_err}")

                                # Build prompt - add warning if this is a retry
                                reporter_prompt = query_text
                                if attempt > 0 and validation_errors:
                                    if last_failure_kind == "post_verification":
                                        # Unverified numeric claims, not missing artifacts — use the
                                        # numeric-verification retry prompt, not the hallucination template.
                                        reporter_prompt = f"{build_post_verification_retry_prompt(validation_errors)}\n\nOriginal Query: {query_text}"
                                    else:
                                        # Targeted repair: fix only the flagged report issues /
                                        # missing artifacts; do not rewrite the report from scratch.
                                        reporter_prompt = f"{build_repair_prompt(ValidationResult(
                                            is_valid=False,
                                            missing_artifacts=last_missing_artifacts,
                                            existing_artifacts=[],
                                            fallback_detected=last_fallback_detected,
                                            hallucination_warnings=validation_errors,
                                            report_structure_valid=False,
                                            report_structure_issues=validation_errors
                                        ), agent_class_name, task.title, task.description, self.current_output_dir, validation_errors)}\n\nOriginal Query: {query_text}"
                                reporter_prompt = f"{reporter_prompt}{profile_block}"

                                agent_res = agent_instance.run_task(
                                    prompt=reporter_prompt,
                                    analyst_summaries=summaries.get("analyst_summaries", []),
                                    ml_summaries=summaries.get("ml_summaries", []),
                                    artifacts=self.generated_artifacts,
                                    plan=active_plan,
                                    step_callback=step_callback,
                                    cancellation_token=self.cancellation_token,
                                    verified_metrics=verified_metrics,
                                    fact_sheet=fact_sheet,
                                    max_tool_calls=current_tool_budget,
                                    max_correction_turns=0,  # retries owned by the combined orchestrator loop below
                                )
                                self.final_report = agent_res.get("final_report", "")
                                task_out = self.final_report

                                # Write final_report.md IMMEDIATELY after ReporterAgent generates it,
                                # BEFORE validation, so the validator can verify the file exists.
                                if self.final_report:
                                    report_filename = "final_report.md"
                                    report_filepath = os.path.join(self.current_output_dir, report_filename)
                                    try:
                                        with open(report_filepath, "w", encoding="utf-8") as rf:
                                            rf.write(self.final_report)
                                        print(f"[Orchestrator] ReporterAgent task #{task.task_id}: wrote final_report.md for validation")
                                        if report_filename not in self.generated_artifacts:
                                            self.generated_artifacts.append(report_filename)
                                    except Exception as write_err:
                                        print(f"[Orchestrator Warning] Failed to save {report_filepath}: {write_err}")

                            elif isinstance(agent_instance, MLAgent):
                                # Resolve current dataset artifact
                                current_dataset_file = None
                                for cand in ["feature_engineered_dataset.csv", "cleaned_dataset.csv", "cleaned_data.csv"]:
                                    cand_path = os.path.join(self.current_output_dir, cand)
                                    if cand in self.generated_artifacts or os.path.exists(cand_path):
                                        current_dataset_file = cand
                                        break

                                ml_files = list(files_list)
                                if current_dataset_file:
                                    full_cand_path = os.path.join(self.current_output_dir, current_dataset_file)
                                    if full_cand_path not in ml_files and current_dataset_file not in ml_files:
                                        ml_files.insert(0, full_cand_path)

                                # Surface ground-truth verified metrics inside the sandbox as soon
                                # as ML artifacts exist, so agents can load them directly instead
                                # of burning tool calls searching for metric files.
                                vm_path = os.path.join(self.current_output_dir, "verified_metrics.json")
                                if not os.path.exists(vm_path) and os.path.exists(
                                    os.path.join(self.current_output_dir, "model_comparison.csv")
                                ):
                                    try:
                                        build_verified_metrics(self.current_output_dir, uploaded_files=files_list)
                                        if "verified_metrics.json" not in self.generated_artifacts:
                                            self.generated_artifacts.append("verified_metrics.json")
                                    except Exception as vm_err:
                                        print(f"[Orchestrator Notice] early build_verified_metrics error: {vm_err}")
                                if os.path.exists(vm_path) and vm_path not in ml_files and "verified_metrics.json" not in ml_files:
                                    ml_files.append(vm_path)

                                # Check if this is a Hyperparameter Tuning task
                                is_hpt_task = _is_tuning_task(task)

                                if is_hpt_task:
                                    # Validate presence of model comparison artifacts.
                                    # The champion pointer (not a mutable best_model.pkl)
                                    # is the canonical trained-model reference.
                                    model_artifacts = ["model_comparison.csv", "champion_model.json"]
                                    missing_model_artifacts = [
                                        art for art in model_artifacts
                                        if art not in self.generated_artifacts and not os.path.exists(os.path.join(self.current_output_dir, art))
                                    ]

                                    if missing_model_artifacts:
                                        warn_msg = f"[Orchestrator Warning] Skipping Task #{task.task_id} ({task.title}): Hyperparameter Tuning requires prior model comparison artifacts ({', '.join(model_artifacts)}), but {', '.join(missing_model_artifacts)} were not found."
                                        if HAS_LOGFIRE:
                                            logfire.warn(warn_msg, task_id=task.task_id, missing=missing_model_artifacts)
                                        print(warn_msg, file=sys.stderr)

                                        task.status = "skipped"
                                        task.result = f"Skipped: Required model comparison artifacts ({', '.join(missing_model_artifacts)}) not found."
                                        self.completed_task_results.append({
                                            "task_id": task.task_id,
                                            "title": task.title,
                                            "agent": agent_class_name,
                                            "status": "skipped",
                                            "output": task.result,
                                            "warning": warn_msg
                                        })
                                        return

                                    # Add comparison artifacts to ml_files
                                    for art in ["model_comparison.csv", "champion_model.json", "model_registry.json", "ml_summary.json"]:
                                        art_path = os.path.join(self.current_output_dir, art)
                                        if os.path.exists(art_path) and art_path not in ml_files and art not in ml_files:
                                            ml_files.append(art_path)

                                    data_context_note = f"\n[INPUT DATASET CONTEXT]: Load the champion model referenced by 'champion_model.json' (an immutable versioned artifact, e.g. logistic_regression_p0.pkl) and baseline scores from 'model_comparison.csv'. Dataset is '{current_dataset_file or 'dataset.csv'}'. NEVER load a mutable 'best_model.pkl'."
                                elif current_dataset_file:
                                    data_context_note = f"\n[INPUT DATASET CONTEXT]: Processed dataset is available as '{current_dataset_file}'. Load and train on this dataset directly (e.g. pd.read_csv('{current_dataset_file}')). Do NOT re-calculate raw duplicate/missing values."
                                else:
                                    main_name = os.path.basename(files_list[0]) if files_list else "dataset.csv"
                                    data_context_note = f"\n[INPUT DATASET CONTEXT]: Main dataset is available as '{main_name}'. Note: No pre-cleaned dataset artifact was produced; proceed directly with this main dataset."

                                if os.path.exists(vm_path):
                                    data_context_note += (
                                        "\n[VERIFIED METRICS]: 'verified_metrics.json' is in FILE_REGISTRY — the "
                                        "authoritative ground-truth metrics for this run. Load it directly via "
                                        "json.load('verified_metrics.json'); do not search for metric files."
                                    )

                                # Build prompt - add warning if this is a retry
                                ml_prompt = f"{query_text}\nTask: {task.title}\nDescription: {task.description}{data_context_note}"
                                if attempt > 0 and validation_errors:
                                    # Targeted repair: only re-do the failed parts (missing
                                    # artifacts / own summary), not the whole task.
                                    ml_prompt = f"{build_repair_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=last_missing_artifacts,
                                        existing_artifacts=[],
                                        fallback_detected=last_fallback_detected,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=True,
                                        report_structure_issues=[]
                                    ), agent_class_name, task.title, task.description, self.current_output_dir, validation_errors)}{data_context_note}"
                                ml_prompt = f"{ml_prompt}{profile_block}"

                                agent_res = agent_instance.run_task(
                                    prompt=ml_prompt,
                                    uploaded_files=ml_files,
                                    plan=active_plan,
                                    task_id=task.task_id,
                                    max_tool_calls=current_tool_budget,
                                    step_callback=step_callback,
                                    cancellation_token=self.cancellation_token
                                )

                                task_out = agent_res.get("response") or agent_res.get("final_response") or ""

                            elif isinstance(agent_instance, DataAnalystAgent):
                                analyst_files = list(files_list)
                                # Pass existing cleaned or feature engineered datasets if available
                                for cand in ["feature_engineered_dataset.csv", "cleaned_dataset.csv", "cleaned_data.csv"]:
                                    cand_path = os.path.join(self.current_output_dir, cand)
                                    if (cand in self.generated_artifacts or os.path.exists(cand_path)) and cand_path not in analyst_files and cand not in analyst_files:
                                        analyst_files.insert(0, cand_path)

                                # Expose verified metrics to the sandbox once they exist.
                                vm_path = os.path.join(self.current_output_dir, "verified_metrics.json")
                                if os.path.exists(vm_path) and vm_path not in analyst_files and "verified_metrics.json" not in analyst_files:
                                    analyst_files.append(vm_path)

                                # Build prompt - add warning if this is a retry
                                analyst_prompt = f"{query_text}\nTask: {task.title}\nDescription: {task.description}"
                                if attempt > 0 and validation_errors:
                                    # Targeted repair: only re-do the failed parts, not the whole task.
                                    analyst_prompt = f"{build_repair_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=last_missing_artifacts,
                                        existing_artifacts=[],
                                        fallback_detected=last_fallback_detected,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=True,
                                        report_structure_issues=[]
                                    ), agent_class_name, task.title, task.description, self.current_output_dir, validation_errors)}"
                                analyst_prompt = f"{analyst_prompt}{profile_block}"

                                agent_res = agent_instance.run_task(
                                    prompt=analyst_prompt,
                                    uploaded_files=analyst_files,
                                    plan=active_plan,
                                    task_id=task.task_id,
                                    max_tool_calls=current_tool_budget,
                                    step_callback=step_callback,
                                    cancellation_token=self.cancellation_token
                                )

                                task_out = agent_res.get("response") or agent_res.get("final_response") or ""

                            else:
                                # Fallback agent
                                fallback_prompt = f"{query_text}\nTask: {task.title}\nDescription: {task.description}"
                                if attempt > 0 and validation_errors:
                                    # Targeted repair: only re-do the failed parts, not the whole task.
                                    fallback_prompt = f"{build_repair_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=last_missing_artifacts,
                                        existing_artifacts=[],
                                        fallback_detected=last_fallback_detected,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=True,
                                        report_structure_issues=[]
                                    ), 'DataAnalystAgent', task.title, task.description, self.current_output_dir, validation_errors)}"

                                fallback_res = self.data_analyst_agent.run_task(
                                    prompt=f"{fallback_prompt}{profile_block}",
                                    uploaded_files=files_list,
                                    plan=active_plan,
                                    task_id=task.task_id,
                                    max_tool_calls=current_tool_budget,
                                    step_callback=step_callback,
                                    cancellation_token=self.cancellation_token
                                )
                                agent_res = fallback_res
                                agent_class_name = "DataAnalystAgent (Fallback)"
                                task_out = fallback_res.get("response", "")

                            # ================================================================
                            # COMBINED VALIDATION: ARTIFACT VALIDATION + REPORTER POST-VERIFICATION
                            # ================================================================
                            # Both checks are evaluated on the SAME attempt, then ONE shared
                            # retry budget (max_retries, from max_validation_retries) is spent:
                            #   - both fail   -> retry driven by artifact-validation errors
                            #   - only artifact validation fails -> retry per artifact validation
                            #   - only post-verification fails   -> retry per post-verification
                            # There is no separate retry budget for either check.
                            task_type = getattr(task, "task_type", None)
                            validation_result = validate_artifacts(
                                agent_response=agent_res,
                                output_dir=self.current_output_dir,
                                agent_name=agent_class_name,
                                task_type=task_type
                            )

                            # Global counters: per-attempt latency, retry latency
                            _attempt_dt = time.perf_counter() - _attempt_t0
                            if attempt > 0:
                                get_recorder().add_duration("retries.latency_s", _attempt_dt)
                            get_recorder().add_duration("task_attempt.latency_s", _attempt_dt)

                            post_verification_failed = False
                            post_verification_errors: list = []
                            if isinstance(agent_instance, ReporterAgent):
                                pv = getattr(agent_instance, "last_verification_result", None)
                                if pv is not None and not pv.get("is_valid", True):
                                    post_verification_failed = True
                                    unverified = pv.get("unverified_claims") or []
                                    post_verification_errors = (
                                        [f"Unverified numeric claim in report (must match VERIFIED_METRICS verbatim): '{c}'" for c in unverified[:10]]
                                        or [f"Post-verification failed: {pv.get('unverified_count', 'unknown')} unverified numeric claims in the report."]
                                    )
                            elif isinstance(agent_instance, MLAgent) or getattr(task, "task_type", None) in ("model_comparison", "hyperparameter_tuning"):
                                ml_verif = verify_champion_consistency(
                                    output_dir=self.current_output_dir,
                                    goal_text=query_text,
                                    repair=False,
                                    task_type=getattr(task, "task_type", None),
                                    pre_task_champion=pre_ml_champion,
                                    primary_metric=getattr(task, "primary_metric", None),
                                )
                                if not ml_verif.get("consistent", True):
                                    post_verification_failed = True
                                    post_verification_errors = [
                                        f.message if hasattr(f, "message") else (f.get("message") if isinstance(f, dict) else str(f))
                                        for f in ml_verif.get("findings", [])
                                        if (getattr(f, "severity", None) or (f.get("severity") if isinstance(f, dict) else "")) == "error"
                                    ] or ["Post-ML verification failed: champion and artifact consistency requirements were not met."]

                            if validation_result.is_valid and not post_verification_failed:
                                # Validation passed!
                                validation_passed = True
                                task.status = "completed"
                                task.result = task_out

                                # Collect generated files
                                for gf in agent_res.get("generated_files", []) or []:
                                    if gf not in self.generated_artifacts:
                                        self.generated_artifacts.append(gf)
                                        if HAS_LOGFIRE:
                                            logfire.info(f"New Artifact: {gf}", artifact_file=gf, producer_task=task.task_id, producer_agent=agent_class_name)

                                self.completed_task_results.append({
                                    "task_id": task.task_id,
                                    "title": task.title,
                                    "agent": agent_class_name,
                                    "status": "completed",
                                    "output": task_out,
                                    "summary_data": agent_res.get("summary_data"),
                                    "generated_files": agent_res.get("generated_files", []),
                                    "tool_calls_made": agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0),
                                    "validation_retries": attempt
                                })

                                if HAS_LOGFIRE:
                                    logfire.info(
                                        f"Task #{task.task_id} Completed ({agent_class_name})",
                                        task_id=task.task_id,
                                        agent=agent_class_name,
                                        status="completed",
                                        tool_calls_made=agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0),
                                        generated_files=agent_res.get("generated_files", []),
                                        validation_retries=attempt
                                    )

                                if attempt > 0:
                                    print(f"[Orchestrator] Task #{task.task_id} validation passed after {attempt} retry(s)")

                            else:
                                # At least one check failed on this attempt.
                                # Error-source priority: artifact validation wins when both fail.
                                if not validation_result.is_valid:
                                    failure_kind = "artifact_validation"
                                    last_validation_errors = validation_result.hallucination_warnings
                                    last_missing_artifacts = list(validation_result.missing_artifacts or [])
                                    last_fallback_detected = validation_result.fallback_detected
                                else:
                                    failure_kind = "post_ml_verification" if (isinstance(agent_instance, MLAgent) or getattr(task, "task_type", None) in ("model_comparison", "hyperparameter_tuning")) else "post_verification"
                                    last_validation_errors = post_verification_errors
                                    last_missing_artifacts = []
                                    last_fallback_detected = False

                                last_failure_kind = failure_kind

                                print(f"[Orchestrator Validation] Task #{task.task_id} ({agent_class_name}) failed {failure_kind} (attempt {attempt + 1}/{max_retries + 1}):")
                                for warn in last_validation_errors:
                                    print(f"  - {warn}")

                                if HAS_LOGFIRE:
                                    logfire.warn(
                                        "Combined validation failed",
                                        task_id=task.task_id,
                                        agent=agent_class_name,
                                        failure_kind=failure_kind,
                                        attempt=attempt + 1,
                                        max_retries=max_retries,
                                        artifact_validation_valid=validation_result.is_valid,
                                        post_verification_valid=not post_verification_failed,
                                        missing_artifacts=validation_result.missing_artifacts,
                                        fallback_detected=validation_result.fallback_detected,
                                        report_structure_valid=validation_result.report_structure_valid,
                                        report_structure_issues=validation_result.report_structure_issues,
                                        failure_reasons=last_validation_errors
                                    )

                                attempt += 1

                                if attempt <= max_retries:
                                    # Will loop again with updated prompt and extra tool budget
                                    # The errors chosen above will be passed to the next attempt
                                    continue
                                else:
                                    # Max combined retries exceeded - mark as validation_failed
                                    print(f"[Orchestrator ERROR] Task #{task.task_id} failed validation ({failure_kind}) after {max_retries} combined retries. Marking as validation_failed and continuing workflow...")
                                    print(f"  Last validation errors: {last_validation_errors}")

                                    # Downgrade: if every artifact exists on disk and the report is
                                    # structurally valid, the only remaining failure signal is the
                                    # fallback-summary marker (a degraded summary, not missing work).
                                    # Accept the task with a warning instead of failing it — retrying
                                    # cannot fix an agent that never writes its own summary, and the
                                    # artifacts the rest of the workflow consumes are all present.
                                    if (failure_kind == "artifact_validation"
                                            and not validation_result.missing_artifacts
                                            and validation_result.report_structure_valid):
                                        print(f"[Orchestrator] Task #{task.task_id} accepted with degraded summary (all artifacts present, fallback summary in use).")
                                        task.status = "completed"
                                        task.result = task_out
                                        for gf in agent_res.get("generated_files", []) or []:
                                            if gf not in self.generated_artifacts:
                                                self.generated_artifacts.append(gf)
                                                if HAS_LOGFIRE:
                                                    logfire.info(f"New Artifact: {gf}", artifact_file=gf, producer_task=task.task_id, producer_agent=agent_class_name)
                                        self.completed_task_results.append({
                                            "task_id": task.task_id,
                                            "title": task.title,
                                            "agent": agent_class_name,
                                            "status": "completed",
                                            "output": task_out,
                                            "summary_data": agent_res.get("summary_data"),
                                            "generated_files": agent_res.get("generated_files", []),
                                            "tool_calls_made": agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0),
                                            "validation_retries": attempt,
                                            "warning": "Accepted with fallback summary: agent did not produce its own summary JSON, but all required artifacts exist and were verified."
                                        })
                                        if HAS_LOGFIRE:
                                            logfire.warn(
                                                "Task accepted with degraded summary",
                                                task_id=task.task_id,
                                                agent=agent_class_name,
                                                missing_artifacts=[],
                                                fallback_detected=validation_result.fallback_detected
                                            )
                                        task_validation_failed = True
                                        validation_passed = True
                                        downgraded_to_warning = True
                                    else:
                                        task_validation_failed = True
                                        task.status = "validation_failed"
                                        task.result = task_out
                                        downgraded_to_warning = False

                                    # If the reporter's numeric post-verification is still failing,
                                    # append the warning banner to the final persisted report.
                                    if isinstance(agent_instance, ReporterAgent) and post_verification_failed:
                                        unverified = (getattr(agent_instance, "last_verification_result", None) or {}).get("unverified_claims") or []
                                        if unverified:
                                            banner = ("\n\n> **Verification Notice:** The following numeric claims could not be "
                                                      "verified against the ground-truth verified metrics: "
                                                      + "; ".join(f"'{c}'" for c in unverified[:10]) + "\n")
                                            self.final_report = (self.final_report or "") + banner
                                            task_out = self.final_report
                                            report_filepath = os.path.join(self.current_output_dir, "final_report.md")
                                            try:
                                                with open(report_filepath, "w", encoding="utf-8") as rf:
                                                    rf.write(self.final_report)
                                            except Exception as write_err:
                                                print(f"[Orchestrator Warning] Failed to update {report_filepath}: {write_err}")
                                    if not downgraded_to_warning:
                                        task_validation_failed = True
                                        task.status = "validation_failed"
                                        task.result = task_out

                                        # Collect whatever files were generated
                                        for gf in agent_res.get("generated_files", []) or []:
                                            if gf not in self.generated_artifacts:
                                                self.generated_artifacts.append(gf)

                                        self.completed_task_results.append({
                                            "task_id": task.task_id,
                                            "title": task.title,
                                            "agent": agent_class_name,
                                            "status": "validation_failed",
                                            "output": task_out,
                                            "summary_data": agent_res.get("summary_data"),
                                            "generated_files": agent_res.get("generated_files", []),
                                            "tool_calls_made": agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0),
                                            "validation_failed": True,
                                            "validation_errors": validation_result.hallucination_warnings,
                                            "validation_retries": max_retries
                                        })

                                        if HAS_LOGFIRE:
                                            logfire.error(
                                                "Artifact validation failed after max retries",
                                                task_id=task.task_id,
                                                agent=agent_class_name,
                                                missing_artifacts=validation_result.missing_artifacts,
                                                fallback_detected=validation_result.fallback_detected,
                                                report_structure_valid=validation_result.report_structure_valid,
                                                final_errors=last_validation_errors
                                            )

                                    # Exit the while loop - task has permanently failed validation
                                    # We do NOT set validation_passed = True here; task_validation_failed handles exit

                    # Execute the task (with artifact validation & retries),
                    # nested under a per-task parent span so all LLM/tool spans
                    # of this task appear as children in Logfire traces.
                    if HAS_LOGFIRE:
                        with logfire.span(
                            f"Task #{task.task_id}: {task.title}",
                            task_id=task.task_id,
                            assigned_agent=agent_class_name
                        ):
                            _execute_task()
                    else:
                        _execute_task()

                    # Post-task champion selection: after model_comparison task, select initial champion.
                    # After hyperparameter_tuning task, evaluate challenger against current champion.
                    if isinstance(agent_instance, MLAgent) and task.task_type in ("model_comparison", "hyperparameter_tuning"):
                        try:
                            primary_metric = task.primary_metric
                            if task.task_type == "model_comparison":
                                champ_result = select_initial_champion(
                                    self.current_output_dir,
                                    goal_text=query_text,
                                    primary_metric=primary_metric
                                )
                                # Handle new ChampionSelectionResult type
                                if isinstance(champ_result, ChampionSelectionResult):
                                    if champ_result.is_success() and champ_result.champion:
                                        champ = champ_result.champion
                                        print(f"[Orchestrator] Initial champion selected (Task #{task.task_id}): "
                                              f"{champ['model_name']} ({champ_result.primary_metric_name or primary_metric}={champ['score']:.4f})", flush=True)
                                    else:
                                        print(f"[Orchestrator] Champion selection issue (Task #{task.task_id}): {champ_result.error_message or champ_result.status.value}", flush=True)
                                elif champ_result and isinstance(champ_result, dict) and not champ_result.get("error"):
                                    # Backward compat
                                    print(f"[Orchestrator] Initial champion selected (Task #{task.task_id}): "
                                          f"{champ_result['model_name']} ({champ_result.get('primary_metric_name', primary_metric)}={champ_result['score']:.4f})", flush=True)
                                elif champ_result and isinstance(champ_result, dict) and champ_result.get("error"):
                                    print(f"[Orchestrator] Champion selection issue (Task #{task.task_id}): {champ_result['message']}", flush=True)
                            elif task.task_type == "hyperparameter_tuning":
                                champ_result = evaluate_challenger(
                                    self.current_output_dir,
                                    goal_text=query_text,
                                    primary_metric=primary_metric
                                )
                                if isinstance(champ_result, ChampionSelectionResult):
                                    if champ_result.is_success() and champ_result.champion:
                                        champ = champ_result.champion
                                        print(f"[Orchestrator] Challenger evaluated (Task #{task.task_id}): "
                                              f"champion={champ['model_name']} ({champ_result.primary_metric_name or primary_metric}={champ['score']:.4f})", flush=True)
                                    else:
                                        print(f"[Orchestrator] Challenger evaluation issue (Task #{task.task_id}): {champ_result.error_message or champ_result.status.value}", flush=True)
                                elif champ_result and isinstance(champ_result, dict):
                                    # Backward compat
                                    print(f"[Orchestrator] Challenger evaluated (Task #{task.task_id}): "
                                          f"champion={champ_result['model_name']} ({champ_result.get('primary_metric_name', primary_metric)}={champ_result['score']:.4f})", flush=True)
                        except Exception as champ_err:
                            print(f"[Orchestrator Notice] champion selection skipped: {champ_err}")

                    # Global counters: per-task latency, status, retries
                    task_duration = time.perf_counter() - task_t0
                    task_result_entry = next(
                        (r for r in reversed(self.completed_task_results)
                         if r.get("task_id") == task.task_id and "duration_s" not in r),
                        None,
                    )
                    if task_result_entry is not None:
                        task_result_entry["duration_s"] = round(task_duration, 3)
                    get_recorder().log_event(
                        "task_completed",
                        task_id=task.task_id,
                        title=task.title,
                        agent=agent_class_name,
                        status=task.status,
                        duration_s=round(task_duration, 3),
                        retries=(task_result_entry or {}).get("validation_retries", 0),
                        primary_metric=task.primary_metric,
                    )
                    reset_task_context(task_ctx_token)

                    # Check cancellation after task execution
                    self.cancellation_token.check_cancelled(
                        step_name=f"After Task #{task.task_id}: {task.title}",
                        task_id=task.task_id
                    )

                    _safe_callback("task_execution_completed", {
                        "task_id": task.task_id,
                        "title": task.title,
                        "status": task.status,
                        "assigned_agent": agent_class_name,
                        "primary_metric": task.primary_metric
                    })

                # If no explicit Reporter task was executed or report is empty, synthesize final report now
                if not has_reporter_task or not self.final_report:
                    self.cancellation_token.check_cancelled(step_name="Before final report synthesis")

                    _safe_callback("final_synthesis_started", {
                        "agent": "ReporterAgent",
                        "task_results_count": len(self.completed_task_results)
                    })

                    def _do_reporter_synthesis():
                        try:
                            summaries = self._collect_agent_summaries(self.current_output_dir)
                            try:
                                verified_metrics = build_verified_metrics(self.current_output_dir, uploaded_files=files_list)
                                if "verified_metrics.json" not in self.generated_artifacts:
                                    self.generated_artifacts.append("verified_metrics.json")
                            except Exception as vm_err:
                                print(f"[Orchestrator Notice] build_verified_metrics error: {vm_err}")
                                verified_metrics = {}

                            # Preflight gate (checks + auto-repair; warn-not-stop) then the
                            # code-built fact sheet for the reporter's {{blanks}}.
                            fact_sheet = None
                            try:
                                preflight = run_preflight(self.current_output_dir, goal_text=query_text)
                                if preflight.get("repairs_applied"):
                                    print(f"[Orchestrator] Preflight repairs applied: {preflight['repairs_applied']}", flush=True)
                                elif preflight.get("repairs"):  # backward compat
                                    print(f"[Orchestrator] Preflight repairs applied: {preflight['repairs']}", flush=True)
                                verified_metrics = build_verified_metrics(self.current_output_dir, uploaded_files=files_list)
                                fact_sheet = build_facts(self.current_output_dir, verified_metrics, preflight)
                            except Exception as pre_err:
                                print(f"[Orchestrator Notice] preflight/fact sheet skipped: {pre_err}")

                            self.final_report = self.reporter_agent.generate_report(
                                user_goal=query_text,
                                analyst_summaries=summaries.get("analyst_summaries", []),
                                ml_summaries=summaries.get("ml_summaries", []),
                                artifacts=self.generated_artifacts,
                                plan=active_plan,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token,
                                verified_metrics=verified_metrics,
                                fact_sheet=fact_sheet,
                                max_correction_turns=2,  # standalone synthesis path: no orchestrator retry loop here
                            )
                        except WorkflowCancelledException:
                            raise
                        except Exception as rep_err:
                            print(f"[Orchestrator Warning] ReporterAgent synthesis failed: {rep_err}", file=sys.stderr)
                            if HAS_LOGFIRE:
                                logfire.warn("ReporterAgent synthesis failed, concatenating task results", error=str(rep_err))
                            parts = ["# Analysis Execution Summary\n"]
                            for tr in self.completed_task_results:
                                parts.append(f"## Task #{tr.get('task_id')}: {tr.get('title')} ({tr.get('agent')})")
                                parts.append(tr.get('output') or tr.get('summary') or tr.get('error') or "")
                                parts.append("")
                            self.final_report = "\n".join(parts)

                    with task_context(agent="ReporterAgent", phase="report_synthesis"):
                        with get_recorder().step_timer("report_synthesis"):
                            if HAS_LOGFIRE:
                                with logfire.span("ReporterAgent: Synthesize Final Report", task_count=len(self.completed_task_results)):
                                    _do_reporter_synthesis()
                            else:
                                _do_reporter_synthesis()

                    # Validate the synthesized final report
                    if self.final_report:
                        report_filename = "final_report.md"
                        report_filepath = os.path.join(self.current_output_dir, report_filename)
                        if report_filename not in self.generated_artifacts:
                            self.generated_artifacts.append(report_filename)
                        try:
                            with open(report_filepath, "w", encoding="utf-8") as rf:
                                rf.write(self.final_report)
                        except Exception as write_err:
                            print(f"[Orchestrator Warning] Failed to save {report_filepath}: {write_err}")

                        # Validate the synthesized report
                        task_type = "report_generation"
                        synthesis_validation = validate_artifacts(
                            agent_response={"final_report": self.final_report, "generated_files": [report_filename]},
                            output_dir=self.current_output_dir,
                            agent_name="ReporterAgent (Synthesis)",
                            task_type=task_type
                        )
                        if not synthesis_validation.is_valid:
                            print(f"[Orchestrator Validation] Synthesized final report failed validation:")
                            for warn in synthesis_validation.hallucination_warnings:
                                print(f"  - {warn}")
                            # Add validation failure to task results for tracking
                            self.completed_task_results.append({
                                "task_id": 0,
                                "title": "Final Report Synthesis (Auto)",
                                "agent": "ReporterAgent (Synthesis)",
                                "status": "validation_failed",
                                "output": self.final_report,
                                "validation_failed": True,
                                "validation_errors": synthesis_validation.hallucination_warnings,
                                "validation_retries": 0
                            })

                # Save report as markdown file in the run output directory
                report_filename = "final_report.md"
                report_filepath = os.path.join(self.current_output_dir, report_filename)
                try:
                    with open(report_filepath, "w", encoding="utf-8") as rf:
                        rf.write(self.final_report)
                    if report_filename not in self.generated_artifacts:
                        self.generated_artifacts.append(report_filename)
                except Exception as write_err:
                    print(f"[Orchestrator Warning] Failed to save {report_filepath}: {write_err}")

                _safe_callback("report_completed", {
                    "final_report": self.final_report,
                    "report_file": report_filepath,
                    "output_dir": self.current_output_dir
                })

                _safe_callback("plan_execution_finished", {
                    "plan_id": active_plan.plan_id,
                    "final_report": self.final_report,
                    "artifacts": self.generated_artifacts,
                    "output_dir": self.current_output_dir
                })

                # ==================================================================
                # FINAL VERIFICATION PHASE (hybrid stack) + SEVERITY-BASED STATUS
                # ==================================================================
                # 1. Deterministic cross-artifact consistency checks
                # 2. LLM prose-consistency judge (if enabled/available)
                # 3. Severity policy classifies the run:
                #    failed / failed-consistency / completed_with_warnings / completed
                completed_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "completed"]
                skipped_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "skipped"]
                failed_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "failed"]
                validation_failed_tasks = [tr for tr in self.completed_task_results if tr.get("validation_failed", False)]
                has_validation_failures = len(validation_failed_tasks) > 0
                has_fatal_failures = len(failed_tasks) > 0

                _safe_callback("final_verification_started", {"output_dir": self.current_output_dir})
                with task_context(agent="VerificationStack", phase="verification"):
                    with get_recorder().step_timer("final_verification"):
                        verification = self._run_final_verification(query_text, active_plan)

                # Post-report repair loop: while critical findings remain, combine
                # ALL findings (numeric + consistency + judge) into one correction
                # prompt and repair (max max_repair_attempts, 5 dedicated sandbox
                # calls per attempt), re-verifying with all methods after each.
                with task_context(agent="ReporterAgent", phase="repair"):
                    with get_recorder().step_timer("report_repair"):
                        verification = self._repair_final_report(verification, files_list, step_callback=step_callback)
                findings = verification.get("findings", [])

                classification = self._classify_run_status(findings)
                workflow_status = classification["status"]
                workflow_message = classification["message"]

                # Persist the findings for the UI and post-run audits
                try:
                    with open(os.path.join(self.current_output_dir, "findings.json"), "w", encoding="utf-8") as ff:
                        json.dump({
                            "run_status": workflow_status,
                            "message": workflow_message,
                            "counts": classification["counts"],
                            "repair_attempts_used": getattr(self, "repair_attempts_used", 0),
                            "numeric_verification": verification.get("numeric"),
                            "findings": findings,
                        }, ff, indent=2, default=str)
                except Exception as ff_err:
                    print(f"[Orchestrator Notice] could not persist findings.json: {ff_err}")

                # Severity policy: a failed-consistency run must NOT look like a
                # polished deliverable. Prepend a prominent non-delivery banner.
                if workflow_status == "failed-consistency" and self.final_report:
                    critical_msgs = [f.get("message", "") for f in findings if f.get("severity") == "critical"][:8]
                    banner_lines = [
                        "> ⛔ **RUN NOT CLEARED FOR DELIVERY (`failed-consistency`)**",
                        "> This report failed the final consistency verification. The findings below were",
                        "> established after agent retries and remain unresolved:",
                    ]
                    for msg in critical_msgs:
                        banner_lines.append(f"> - {msg}")
                    banner_lines.append("> ")
                    banner_lines.append("> Treat every figure below as unverified until the findings are resolved.")
                    self.final_report = "\n".join(banner_lines) + "\n\n" + self.final_report
                    report_filepath = os.path.join(self.current_output_dir, "final_report.md")
                    try:
                        with open(report_filepath, "w", encoding="utf-8") as rf:
                            rf.write(self.final_report)
                    except Exception as write_err:
                        print(f"[Orchestrator Warning] Failed to update {report_filepath}: {write_err}")

                # Build the run manifest (machine-readable run record)
                run_manifest = {
                    "run_id": os.path.basename(self.current_output_dir),
                    "plan_id": active_plan.plan_id,
                    "dataset_type": getattr(active_plan, "dataset_type", "classification"),
                    "query": query_text,
                    "input_files": files_list,
                    "output_dir": self.current_output_dir,
                    "started_at": run_timestamp,
                    "finished_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "run_status": workflow_status,
                    "status_message": workflow_message,
                    "severity_counts": classification["counts"],
                    "repair_attempts_used": getattr(self, "repair_attempts_used", 0),
                    "numeric_verification": verification.get("numeric"),
                    "tasks": [
                        {
                            "task_id": tr.get("task_id"),
                            "title": tr.get("title"),
                            "agent": tr.get("agent"),
                            "status": tr.get("status"),
                            "duration_s": tr.get("duration_s"),
                            "validation_retries": tr.get("validation_retries", 0),
                        }
                        for tr in self.completed_task_results
                    ],
                    "artifacts": sorted(
                        set(self.generated_artifacts)
                        | {f for f in os.listdir(self.current_output_dir) if os.path.isfile(os.path.join(self.current_output_dir, f))}
                    ) if os.path.isdir(self.current_output_dir) else sorted(set(self.generated_artifacts)),
                    "findings": findings,
                }
                self.last_run_manifest = run_manifest
                try:
                    with open(os.path.join(self.current_output_dir, "run_manifest.json"), "w", encoding="utf-8") as mf:
                        json.dump(run_manifest, mf, indent=2, default=str)
                except Exception as mf_err:
                    print(f"[Orchestrator Notice] could not persist run_manifest.json: {mf_err}")

                # Log validation failures for debugging
                if has_validation_failures:
                    print(f"[Orchestrator] Workflow {workflow_status}: {len(validation_failed_tasks)} task(s) had validation failures:")
                    for tr in validation_failed_tasks:
                        print(f"  - Task #{tr.get('task_id')}: {tr.get('title')} ({tr.get('agent')}) - {tr.get('validation_errors', [])}")

                if HAS_LOGFIRE:
                    logfire.info(
                        f"Multi-Agent Plan Execution {workflow_status.replace('_', ' ').title()}",
                        plan_id=active_plan.plan_id,
                        tasks_completed=len(completed_tasks),
                        tasks_failed=len(failed_tasks),
                        tasks_skipped=len(skipped_tasks),
                        tasks_validation_failed=len(validation_failed_tasks),
                        artifacts_generated=self.generated_artifacts,
                        report_length=len(self.final_report),
                        output_dir=self.current_output_dir
                    )

                # Explicit workflow completed notification
                _safe_callback("workflow_completed", {
                    "status": workflow_status,
                    "plan_id": active_plan.plan_id,
                    "message": workflow_message,
                    "tasks_completed": len(completed_tasks),
                    "tasks_failed": len(failed_tasks),
                    "tasks_skipped": len(skipped_tasks),
                    "tasks_validation_failed": len(validation_failed_tasks),
                    "total_tasks": len(active_plan.tasks),
                    "artifacts": self.generated_artifacts,
                    "output_dir": self.current_output_dir,
                    "final_report": self.final_report,
                    "severity_counts": classification["counts"],
                    "repair_attempts_used": getattr(self, "repair_attempts_used", 0),
                    "findings": findings[:20]
                })

                return {
                    "status": workflow_status,
                    "plan": active_plan.to_dict(),
                    "task_results": self.completed_task_results,
                    "generated_artifacts": self.generated_artifacts,
                    "final_report": self.final_report,
                    "output_dir": self.current_output_dir,
                    "findings": findings,
                    "severity_counts": classification["counts"],
                    "repair_attempts_used": getattr(self, "repair_attempts_used", 0),
                    "status_message": workflow_message,
                    "run_manifest": run_manifest
                }

            except WorkflowPausedException as pause_err:
                active_plan.status = PlanStatus.PAUSED
                paused_task_id = pause_err.task_id or (current_active_task.task_id if current_active_task else 1)

                # Mark the currently running task as "paused", NOT completed!
                if current_active_task:
                    current_active_task.status = "paused"
                    current_active_task.result = f"Paused during execution ({pause_err.step or 'in-progress step'}). Will re-execute on resume."

                # Keep subsequent tasks in pending state
                for t in active_plan.tasks:
                    if t.task_id > paused_task_id and t.status in ["in_progress", "paused"]:
                        t.status = "pending"

                # Filter out any incomplete result for paused_task_id or later
                self.completed_task_results = [
                    tr for tr in self.completed_task_results
                    if tr.get("task_id", 0) < paused_task_id
                ]

                pause_msg = (
                    f"Workflow paused at {pause_err.step or (f'Task #{paused_task_id}' if paused_task_id else 'in-progress step')}. "
                    f"In-progress step halted immediately and will be re-executed upon resuming."
                )

                checkpoint = {
                    "workflow_id": active_plan.plan_id or f"wf_{int(time.time())}",
                    "status": "paused",
                    "paused_at": datetime.now().isoformat(),
                    "paused_task_id": paused_task_id,
                    "paused_step": pause_err.step or f"Task #{paused_task_id}",
                    "reason": pause_err.message,
                    "user_query": query_text,
                    "input_files": files_list,
                    "output_dir": self.current_output_dir,
                    "plan": active_plan.to_dict(),
                    "completed_task_results": copy.deepcopy(self.completed_task_results),
                    "generated_artifacts": list(self.generated_artifacts),
                    "final_report": self.final_report,
                }
                self.last_pause_checkpoint = checkpoint

                _safe_callback("workflow_paused", {
                    "status": "paused",
                    "plan_id": active_plan.plan_id,
                    "task_id": paused_task_id,
                    "step": pause_err.step,
                    "message": pause_msg,
                    "output_dir": self.current_output_dir,
                    "artifacts": self.generated_artifacts,
                    "completed_tasks": len(self.completed_task_results),
                    "checkpoint": checkpoint,
                })

                if HAS_LOGFIRE:
                    logfire.info("Workflow Paused by User", reason=pause_err.message, step=pause_err.step, task_id=paused_task_id)

                return {
                    "status": "paused",
                    "message": pause_msg,
                    "plan": active_plan.to_dict(),
                    "task_results": self.completed_task_results,
                    "generated_artifacts": self.generated_artifacts,
                    "output_dir": self.current_output_dir,
                    "paused_task_id": paused_task_id,
                    "checkpoint": checkpoint,
                }

            except WorkflowCancelledException as cancel_err:
                active_plan.status = PlanStatus.CANCELLED
                # Mark current task and remaining tasks as cancelled
                found_current = False
                for t in active_plan.tasks:
                    if current_active_task and t.task_id == current_active_task.task_id:
                        t.status = "cancelled"
                        t.result = f"Terminated by user: {cancel_err.message}"
                        found_current = True
                    elif found_current and t.status in ["pending", "in_progress"]:
                        t.status = "cancelled"
                        t.result = "Skipped due to workflow cancellation"

                cancel_msg = (
                    f"Workflow was terminated on user request at "
                    f"{cancel_err.step or (f'Task #{current_active_task.task_id}' if current_active_task else 'workflow initiation')}. "
                    f"All subsequent agent actions were aborted."
                )

                _safe_callback("workflow_cancelled", {
                    "status": "cancelled",
                    "plan_id": active_plan.plan_id,
                    "task_id": cancel_err.task_id or (current_active_task.task_id if current_active_task else None),
                    "step": cancel_err.step,
                    "message": cancel_msg,
                    "output_dir": self.current_output_dir,
                    "artifacts": self.generated_artifacts,
                    "completed_tasks": len(self.completed_task_results)
                })

                if HAS_LOGFIRE:
                    logfire.warn("Workflow Terminated by User", reason=cancel_err.message, step=cancel_err.step)

                return {
                    "status": "cancelled",
                    "message": cancel_msg,
                    "plan": active_plan.to_dict(),
                    "task_results": self.completed_task_results,
                    "generated_artifacts": self.generated_artifacts,
                    "output_dir": self.current_output_dir
                }

            except Exception as unhandled_err:
                err_str = str(unhandled_err)
                tb_str = traceback.format_exc()
                print(f"[Orchestrator Fatal Error] Workflow failed: {err_str}\n{tb_str}", file=sys.stderr)

                if current_active_task:
                    current_active_task.status = "failed"
                    current_active_task.result = f"Fatal error: {err_str}"

                # Mark remaining tasks as cancelled
                for t in active_plan.tasks:
                    if t.status == "pending":
                        t.status = "cancelled"
                        t.result = "Skipped due to prior failure"

                fail_msg = (
                    f"Workflow execution stopped due to an error at "
                    f"{f'Task #{current_active_task.task_id}' if current_active_task else 'workflow orchestration'}: {err_str}"
                )

                _safe_callback("workflow_failed", {
                    "status": "failed",
                    "plan_id": active_plan.plan_id,
                    "task_id": current_active_task.task_id if current_active_task else None,
                    "error": err_str,
                    "traceback": tb_str,
                    "message": fail_msg,
                    "output_dir": self.current_output_dir,
                    "artifacts": self.generated_artifacts
                })

                if HAS_LOGFIRE:
                    logfire.error("Workflow Execution Failed", error=err_str, traceback=tb_str)

                return {
                    "status": "failed",
                    "error": err_str,
                    "message": fail_msg,
                    "plan": active_plan.to_dict(),
                    "task_results": self.completed_task_results,
                    "generated_artifacts": self.generated_artifacts,
                    "output_dir": self.current_output_dir
                }

        rec = get_recorder()
        workflow_result: Optional[Dict[str, Any]] = None
        try:
            with rec.step_timer("total_workflow"):
                if HAS_LOGFIRE:
                    with logfire.span(f"Multi-Agent Execution: {query_text[:100]}", plan_id=active_plan.plan_id, total_tasks=len(active_plan.tasks)):
                        workflow_result = _execute_workflow()
                else:
                    workflow_result = _execute_workflow()
            return workflow_result
        finally:
            self.current_active_task = None
            # Close the run: writes the run_summary event + run_metrics.json.
            try:
                status = workflow_result.get("status") if isinstance(workflow_result, dict) else "failed"
                rec.finalize(status=status)
            except Exception:
                pass

    def run(
        self,
        user_query: str,
        files: Optional[List[str]] = None,
        dataset_context: Optional[str] = None,
        max_tool_calls_per_agent: int = 10,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end multi-agent execution:
        1. Creates execution plan via PlannerAgent
        2. Executes tasks across specialized agents
        3. Returns final synthesized report and artifacts
        """
        self.cancellation_token = cancellation_token or CancellationToken()

        # Give the planner the same authoritative profile so the plan itself
        # references real shapes/columns from the start.
        planner_context = dataset_context or ""
        try:
            profile = build_dataset_profile(
                self.current_output_dir if os.path.isdir(self.current_output_dir) else WORKSPACE_ROOT,
                files,
                persist=False,
            )
            planner_context = f"{planner_context}\n{render_dataset_profile_block(profile)}".strip()
        except Exception as profile_err:
            print(f"[Orchestrator Notice] planner profile build failed: {profile_err}")

        plan = self.create_plan(
            user_query=user_query,
            dataset_context=planner_context,
            files=files,
            max_tool_calls_per_task=max_tool_calls_per_agent,
            step_callback=step_callback,
            cancellation_token=self.cancellation_token
        )

        plan = self.confirm_plan(plan)

        return self.execute_plan(
            plan=plan,
            user_query=user_query,
            files=files,
            max_tool_calls_per_agent=max_tool_calls_per_agent,
            step_callback=step_callback,
            cancellation_token=self.cancellation_token
        )



# Backward-compatible alias
MultiAgentOrchestrator = WorkflowOrchestrator


if __name__ == "__main__":
    print("Testing WorkflowOrchestrator...")
    orchestrator = WorkflowOrchestrator()
    print("Initialized PlannerAgent, DataAnalystAgent, MLAgent, ReporterAgent successfully!")
