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

from sandbox_execution import ToolManager
from agents.tools import reset_global_sandbox_manager
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.planner_agent import PlannerAgent, ExecutionPlan, TaskItem, PlanStatus
from agents.data_analyst_agent import DataAnalystAgent
from agents.ml_agent import MLAgent
from agents.reporter_agent import ReporterAgent
from agents.cancellation import CancellationToken, WorkflowCancelledException


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

        # Initialize specialized agents
        self.planner_agent = PlannerAgent(
            available_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
            model_name=planner_model
        )
        self.data_analyst_agent = DataAnalystAgent(
            model_name=analyst_model,
            tool_manager=self.tool_manager
        )
        self.ml_agent = MLAgent(
            model_name=ml_model,
            tool_manager=self.tool_manager
        )
        self.reporter_agent = ReporterAgent(
            model_name=reporter_model
        )

        self.current_plan: Optional[ExecutionPlan] = None
        self.completed_task_results: List[Dict[str, Any]] = []
        self.generated_artifacts: List[str] = []
        self.final_report: str = ""
        self.current_output_dir: str = os.path.join(WORKSPACE_ROOT, "outputs")
        self.cancellation_token: Optional[CancellationToken] = None

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
        max_tool_calls_per_agent: int = 7,
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
                        if isinstance(agent_instance, ReporterAgent):
                            has_reporter_task = True
                            summaries = self._collect_agent_summaries(self.current_output_dir)
                            reporter_res = agent_instance.run_task(
                                prompt=query_text,
                                analyst_summaries=summaries.get("analyst_summaries", []),
                                ml_summaries=summaries.get("ml_summaries", []),
                                artifacts=self.generated_artifacts,
                                plan=active_plan,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token
                            )
                            self.final_report = reporter_res.get("final_report", "")
                            task.status = "completed"
                            task.result = self.final_report
                            self.completed_task_results.append({
                                "task_id": task.task_id,
                                "title": task.title,
                                "agent": agent_class_name,
                                "status": "completed",
                                "summary": "Executive report generated.",
                                "report_snippet": self.final_report[:500]
                            })
                            if HAS_LOGFIRE:
                                logfire.info(
                                    f"Task #{task.task_id} Completed ({agent_class_name})",
                                    task_id=task.task_id,
                                    agent=agent_class_name,
                                    status="completed",
                                    report_length=len(self.final_report)
                                )

                        elif isinstance(agent_instance, MLAgent):
                            # Check if a cleaned dataset was generated by a previous task or exists in current_output_dir
                            cleaned_dataset_file = None
                            for cand in ["cleaned_dataset.csv", "cleaned_data.csv"]:
                                cand_path = os.path.join(self.current_output_dir, cand)
                                if cand in self.generated_artifacts or os.path.exists(cand_path):
                                    cleaned_dataset_file = cand
                                    break

                            ml_files = list(files_list)
                            if cleaned_dataset_file:
                                full_cand_path = os.path.join(self.current_output_dir, cleaned_dataset_file)
                                if full_cand_path not in ml_files and cleaned_dataset_file not in ml_files:
                                    ml_files.insert(0, full_cand_path)
                                data_context_note = f"\n[INPUT DATASET CONTEXT]: Pre-cleaned dataset produced by DataAnalystAgent is available as '{cleaned_dataset_file}'. Load and train on this cleaned dataset directly (e.g. pd.read_csv('{cleaned_dataset_file}')). Do NOT re-calculate raw duplicate/missing values."
                            else:
                                main_name = os.path.basename(files_list[0]) if files_list else "dataset.csv"
                                data_context_note = f"\n[INPUT DATASET CONTEXT]: Main dataset is available as '{main_name}'. Note: No pre-cleaned dataset artifact was produced; proceed directly with this main dataset."

                            agent_res = agent_instance.run_task(
                                prompt=f"{query_text}\nTask: {task.title}\nDescription: {task.description}{data_context_note}",
                                uploaded_files=ml_files,
                                plan=active_plan,
                                task_id=task.task_id,
                                max_tool_calls=max_tool_calls_per_agent,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token
                            )

                            task.status = "completed" if agent_res.get("status") != "failed" else "failed"
                            task_out = agent_res.get("response") or agent_res.get("final_response") or ""
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
                                "status": task.status,
                                "output": task_out,
                                "summary_data": agent_res.get("summary_data"),
                                "generated_files": agent_res.get("generated_files", []),
                                "tool_calls_made": agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0)
                            })
                            if HAS_LOGFIRE:
                                logfire.info(
                                    f"Task #{task.task_id} Completed ({agent_class_name})",
                                    task_id=task.task_id,
                                    agent=agent_class_name,
                                    status=task.status,
                                    tool_calls_made=agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0),
                                    generated_files=agent_res.get("generated_files", [])
                                )

                        elif isinstance(agent_instance, DataAnalystAgent):
                            agent_res = agent_instance.run_task(
                                prompt=f"{query_text}\nTask: {task.title}\nDescription: {task.description}",
                                uploaded_files=files_list,
                                plan=active_plan,
                                task_id=task.task_id,
                                max_tool_calls=max_tool_calls_per_agent,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token
                            )

                            task.status = "completed" if agent_res.get("status") != "failed" else "failed"
                            task_out = agent_res.get("response") or agent_res.get("final_response") or ""
                            task.result = task_out

                            # Collect generated files (including cleaned_dataset.csv)
                            for gf in agent_res.get("generated_files", []) or []:
                                if gf not in self.generated_artifacts:
                                    self.generated_artifacts.append(gf)
                                    if HAS_LOGFIRE:
                                        logfire.info(f"New Artifact: {gf}", artifact_file=gf, producer_task=task.task_id, producer_agent=agent_class_name)

                            self.completed_task_results.append({
                                "task_id": task.task_id,
                                "title": task.title,
                                "agent": agent_class_name,
                                "status": task.status,
                                "output": task_out,
                                "summary_data": agent_res.get("summary_data"),
                                "generated_files": agent_res.get("generated_files", []),
                                "tool_calls_made": agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0)
                            })
                            if HAS_LOGFIRE:
                                logfire.info(
                                    f"Task #{task.task_id} Completed ({agent_class_name})",
                                    task_id=task.task_id,
                                    agent=agent_class_name,
                                    status=task.status,
                                    tool_calls_made=agent_res.get("tool_calls_made") or agent_res.get("tool_calls_used", 0),
                                    generated_files=agent_res.get("generated_files", [])
                                )

                        else:
                            fallback_res = self.data_analyst_agent.run_task(
                                prompt=f"{query_text}\nTask: {task.title}\nDescription: {task.description}",
                                uploaded_files=files_list,
                                plan=active_plan,
                                task_id=task.task_id,
                                max_tool_calls=max_tool_calls_per_agent,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token
                            )
                            task.status = "completed"
                            task.result = fallback_res.get("response", "")
                            self.completed_task_results.append({
                                "task_id": task.task_id,
                                "title": task.title,
                                "agent": "DataAnalystAgent (Fallback)",
                                "status": "completed",
                                "summary_data": fallback_res.get("summary_data"),
                                "output": fallback_res.get("response", "")
                            })
                            if HAS_LOGFIRE:
                                logfire.info(
                                    f"Task #{task.task_id} Completed (Fallback)",
                                    task_id=task.task_id,
                                    status="completed"
                                )

                    if HAS_LOGFIRE:
                        with logfire.span(f"Task #{task.task_id}: {task.title}", task_id=task.task_id, assigned_agent=agent_class_name):
                            _execute_task()
                    else:
                        _execute_task()

                    # Check cancellation immediately after task execution
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
                            self.final_report = self.reporter_agent.generate_report(
                                user_goal=query_text,
                                analyst_summaries=summaries.get("analyst_summaries", []),
                                ml_summaries=summaries.get("ml_summaries", []),
                                artifacts=self.generated_artifacts,
                                plan=active_plan,
                                step_callback=step_callback,
                                cancellation_token=self.cancellation_token
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

                # Explicit workflow completed notification
                _safe_callback("workflow_completed", {
                    "status": "completed",
                    "plan_id": active_plan.plan_id,
                    "message": "Workflow execution completed successfully.",
                    "tasks_completed": len(self.completed_task_results),
                    "total_tasks": len(active_plan.tasks),
                    "artifacts": self.generated_artifacts,
                    "output_dir": self.current_output_dir,
                    "final_report": self.final_report
                })

                if HAS_LOGFIRE:
                    logfire.info(
                        "Multi-Agent Plan Execution Completed Successfully",
                        plan_id=active_plan.plan_id,
                        tasks_completed=len(self.completed_task_results),
                        artifacts_generated=self.generated_artifacts,
                        report_length=len(self.final_report),
                        output_dir=self.current_output_dir
                    )

                return {
                    "status": "completed",
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
            step_callback=step_callback,
            cancellation_token=self.cancellation_token
        )

        plan = self.confirm_plan(plan)

        return self.execute_plan(
            plan=plan,
            user_query=user_query,
            files=files,
            step_callback=step_callback,
            cancellation_token=self.cancellation_token
        )



# Backward-compatible alias
MultiAgentOrchestrator = WorkflowOrchestrator


if __name__ == "__main__":
    print("Testing WorkflowOrchestrator...")
    orchestrator = WorkflowOrchestrator()
    print("Initialized PlannerAgent, DataAnalystAgent, MLAgent, ReporterAgent successfully!")
