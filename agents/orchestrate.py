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
import traceback
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
from agents.cancellation import CancellationToken, WorkflowCancelledException
from agents.artifact_validator import (
    validate_artifacts,
    validate_and_get_warning,
    build_hallucination_warning_prompt,
    ValidationResult
)


try:
    import logfire
except ImportError:
    logfire = None


class WorkflowOrchestrator:
    """
    Multi-Agent Workflow Orchestrator.
    Manages the lifecycle of planning, dynamic task delegation to specialized agents,
    shared sandbox state, artifact collection, and executive report synthesis.
    """

    def __init__(
        self,
        tool_manager: Optional[ToolManager] = None,
        planner_model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        analyst_model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        ml_model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        reporter_model: str = "nvidia/nemotron-3-ultra-550b-a55b",
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
            max_tool_calls=10
        )

        self.current_plan: Optional[ExecutionPlan] = None
        self.completed_task_results: List[Dict[str, Any]] = []
        self.generated_artifacts: List[str] = []
        self.final_report: str = ""
        self.current_output_dir: str = os.path.join(WORKSPACE_ROOT, "outputs")
        self.cancellation_token: Optional[CancellationToken] = None

        # Artifact validation settings
        self.max_validation_retries: int = 2  # Max times to re-prompt agent for missing artifacts
        self.extra_tool_calls_on_retry: int = 3  # Extra tool calls when retrying after validation failure

    def reset(self, keep_sandbox: bool = False):
        """Resets orchestrator state, memory, and optionally global sandbox."""
        self.current_plan = None
        self.completed_task_results = []
        self.generated_artifacts = []
        self.final_report = ""
        self.current_output_dir = os.path.join(WORKSPACE_ROOT, "outputs")
        self.cancellation_token = None

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

        plan = self.planner_agent.create_plan(
            user_query=user_query,
            dataset_context=dataset_context,
            max_tool_calls_per_task=max_tool_calls_per_task,
            cancellation_token=cancellation_token
        )
        self.current_plan = plan

        if step_callback:
            step_callback("plan_created", {
                "plan": plan.to_dict(),
                "plan_id": plan.plan_id,
                "selected_agents": plan.selected_agents,
                "tasks": [t.to_dict() for t in plan.tasks],
                "notes": plan.notes
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

        updated_plan = self.planner_agent.modify_plan(
            target_plan,
            modification_prompt,
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
    ) -> Dict[str, Any]:
        """
        Executes each task in the plan using its assigned specialized agent.
        Supports instant cancellation, explicit failure/completion telemetry,
        and artifact preservation.
        """
        from datetime import datetime

        active_plan = plan or self.current_plan
        if not active_plan:
            raise ValueError("No active plan provided for execution.")

        query_text = user_query or active_plan.query
        files_list = files or []

        # Setup cancellation token
        self.cancellation_token = cancellation_token or CancellationToken()

        # Create new unique run output directory
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_output_dir = os.path.join(WORKSPACE_ROOT, "outputs", f"run_{run_timestamp}")
        os.makedirs(self.current_output_dir, exist_ok=True)

        if hasattr(self.tool_manager, "set_output_dir"):
            self.tool_manager.set_output_dir(self.current_output_dir)

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
                "output_dir": self.current_output_dir
            })

            self.completed_task_results = []
            has_reporter_task = False
            current_active_task: Optional[TaskItem] = None

            try:
                # Check cancellation at start
                self.cancellation_token.check_cancelled(step_name="Workflow start")

                for task in active_plan.tasks:
                    current_active_task = task
                    # Check cancellation before starting each task
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
                        "description": task.description
                    })

                    task.status = "in_progress"

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

                        while attempt <= max_retries and not validation_passed and not task_validation_failed:
                            if attempt > 0:
                                # Re-prompt: set status back to pending and add hallucination warning
                                task.status = "pending"
                                print(f"[Orchestrator] Retry attempt {attempt}/{max_retries} for Task #{task.task_id} ({agent_class_name})")

                                if HAS_LOGFIRE:
                                    logfire.info(
                                        "Retrying task after artifact validation failure",
                                        task_id=task.task_id,
                                        agent=agent_class_name,
                                        attempt=attempt,
                                        max_retries=max_retries
                                    )

                            # Determine tool call budget for this attempt
                            current_tool_budget = max_tool_calls_per_agent
                            if attempt > 0:
                                current_tool_budget += self.extra_tool_calls_on_retry  # +3 extra calls on retry

                            # Use validation errors from the immediately previous attempt
                            validation_errors = last_validation_errors

                            # Execute the task based on agent type
                            agent_res = None
                            task_out = ""

                            if isinstance(agent_instance, ReporterAgent):
                                has_reporter_task = True
                                summaries = self._collect_agent_summaries(self.current_output_dir)
                                try:
                                    verified_metrics = build_verified_metrics(self.current_output_dir)
                                    if "verified_metrics.json" not in self.generated_artifacts:
                                        self.generated_artifacts.append("verified_metrics.json")
                                except Exception as vm_err:
                                    print(f"[Orchestrator Notice] build_verified_metrics error: {vm_err}")
                                    verified_metrics = {}

                                # Build prompt - add warning if this is a retry
                                reporter_prompt = query_text
                                if attempt > 0 and validation_errors:
                                    reporter_prompt = f"{build_hallucination_warning_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=[],
                                        existing_artifacts=[],
                                        fallback_detected=False,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=False,
                                        report_structure_issues=[]
                                    ), agent_class_name, '', task.description)}\n\nOriginal Query: {query_text}"

                                agent_res = agent_instance.run_task(
                                    prompt=reporter_prompt,
                                    analyst_summaries=summaries.get("analyst_summaries", []),
                                    ml_summaries=summaries.get("ml_summaries", []),
                                    artifacts=self.generated_artifacts,
                                    plan=active_plan,
                                    step_callback=step_callback,
                                    cancellation_token=self.cancellation_token,
                                    verified_metrics=verified_metrics,
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

                                # Check if this is a Hyperparameter Tuning task
                                is_hpt_task = (
                                    getattr(task, "task_type", None) == "hyperparameter_tuning"
                                    or "hyperparameter" in task.title.lower()
                                    or "tuning" in task.title.lower()
                                    or "hpt" in task.title.lower()
                                )

                                if is_hpt_task:
                                    # Validate presence of model comparison artifacts
                                    model_artifacts = ["model_comparison.csv", "best_model.pkl"]
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
                                    for art in ["model_comparison.csv", "best_model.pkl", "ml_summary.json"]:
                                        art_path = os.path.join(self.current_output_dir, art)
                                        if os.path.exists(art_path) and art_path not in ml_files and art not in ml_files:
                                            ml_files.append(art_path)

                                    data_context_note = f"\n[INPUT DATASET CONTEXT]: Load trained model from 'best_model.pkl' and baseline scores from 'model_comparison.csv'. Dataset is '{current_dataset_file or 'dataset.csv'}'."
                                elif current_dataset_file:
                                    data_context_note = f"\n[INPUT DATASET CONTEXT]: Processed dataset is available as '{current_dataset_file}'. Load and train on this dataset directly (e.g. pd.read_csv('{current_dataset_file}')). Do NOT re-calculate raw duplicate/missing values."
                                else:
                                    main_name = os.path.basename(files_list[0]) if files_list else "dataset.csv"
                                    data_context_note = f"\n[INPUT DATASET CONTEXT]: Main dataset is available as '{main_name}'. Note: No pre-cleaned dataset artifact was produced; proceed directly with this main dataset."

                                # Build prompt - add warning if this is a retry
                                ml_prompt = f"{query_text}\nTask: {task.title}\nDescription: {task.description}{data_context_note}"
                                if attempt > 0 and validation_errors:
                                    ml_prompt = f"{build_hallucination_warning_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=[],
                                        existing_artifacts=[],
                                        fallback_detected=False,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=True,
                                        report_structure_issues=[]
                                    ), agent_class_name, '', task.description)}\n\n{ml_prompt}"

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

                                # Build prompt - add warning if this is a retry
                                analyst_prompt = f"{query_text}\nTask: {task.title}\nDescription: {task.description}"
                                if attempt > 0 and validation_errors:
                                    analyst_prompt = f"{build_hallucination_warning_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=[],
                                        existing_artifacts=[],
                                        fallback_detected=False,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=True,
                                        report_structure_issues=[]
                                    ), agent_class_name, '', task.description)}\n\n{analyst_prompt}"

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
                                    fallback_prompt = f"{build_hallucination_warning_prompt(ValidationResult(
                                        is_valid=False,
                                        missing_artifacts=[],
                                        existing_artifacts=[],
                                        fallback_detected=False,
                                        hallucination_warnings=validation_errors,
                                        report_structure_valid=True,
                                        report_structure_issues=[]
                                    ), 'DataAnalystAgent', '', task.description)}\n\n{fallback_prompt}"

                                fallback_res = self.data_analyst_agent.run_task(
                                    prompt=fallback_prompt,
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
                                else:
                                    failure_kind = "post_verification"
                                    last_validation_errors = post_verification_errors

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
                                        report_structure_issues=validation_result.report_structure_issues
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

                    # Check cancellation after task execution
                    self.cancellation_token.check_cancelled(
                        step_name=f"After Task #{task.task_id}: {task.title}",
                        task_id=task.task_id
                    )

                    _safe_callback("task_execution_completed", {
                        "task_id": task.task_id,
                        "title": task.title,
                        "status": task.status,
                        "assigned_agent": agent_class_name
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
                                verified_metrics = build_verified_metrics(self.current_output_dir)
                                if "verified_metrics.json" not in self.generated_artifacts:
                                    self.generated_artifacts.append("verified_metrics.json")
                            except Exception as vm_err:
                                print(f"[Orchestrator Notice] build_verified_metrics error: {vm_err}")
                                verified_metrics = {}

                            self.final_report = self.reporter_agent.generate_report(
                                user_goal=query_text,
                                analyst_summaries=summaries.get("analyst_summaries", []),
                                ml_summaries=summaries.get("ml_summaries", []),
                                artifacts=self.generated_artifacts,
                                plan=active_plan,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token,
                                verified_metrics=verified_metrics,
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

                # Determine overall workflow status based on task results
                validation_failed_tasks = [tr for tr in self.completed_task_results if tr.get("validation_failed", False)]
                completed_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "completed"]
                skipped_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "skipped"]
                failed_tasks = [tr for tr in self.completed_task_results if tr.get("status") == "failed"]

                has_validation_failures = len(validation_failed_tasks) > 0
                has_fatal_failures = len(failed_tasks) > 0

                if has_fatal_failures:
                    workflow_status = "failed"
                    workflow_message = f"Workflow failed: {len(failed_tasks)} task(s) failed with errors."
                elif has_validation_failures:
                    workflow_status = "completed_with_validation_failures"
                    workflow_message = f"Workflow completed but {len(validation_failed_tasks)} task(s) failed artifact validation."
                else:
                    workflow_status = "completed"
                    workflow_message = "Workflow execution completed successfully."

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
                    "final_report": self.final_report
                })

                return {
                    "status": workflow_status,
                    "plan": active_plan.to_dict(),
                    "task_results": self.completed_task_results,
                    "generated_artifacts": self.generated_artifacts,
                    "final_report": self.final_report,
                    "output_dir": self.current_output_dir
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

        if HAS_LOGFIRE:
            with logfire.span(f"Multi-Agent Execution: {query_text[:100]}", plan_id=active_plan.plan_id, total_tasks=len(active_plan.tasks)):
                return _execute_workflow()
        else:
            return _execute_workflow()

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

        plan = self.create_plan(
            user_query=user_query,
            dataset_context=dataset_context,
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
