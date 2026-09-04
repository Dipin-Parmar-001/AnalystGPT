"""
Planner Agent Module - AnalystGPT

Analyzes user queries, selects appropriate agents, builds structured task checklists,
and manages user confirmation, modification, and cancellation workflows.
"""

import os
import sys
import json
import uuid
import re
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from openai import OpenAI

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.cancellation import CancellationToken, WorkflowCancelledException
from agents.memory_manager import sanitize_messages


try:
    import logfire
except ImportError:
    logfire = None

# Load environment variables
load_dotenv()

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

DEFAULT_PORTKEY_CONFIG = {
    "strategy": {
        "mode": "fallback",
        "on_status_codes": [429, 500, 502, 503, 504]
    },
    "targets": [
        {
            "virtual_key": "analystgpt-openrouter",
            "override_params": {
                "model": "nemotron-3-ultra-550b-a55b",
                "temperature": 0.2,
                "max_tokens": 1500
            }
        },
        {
            "virtual_key": "analystgpt-openrouter",
            "override_params": {
                "model": "poolside/laguna-xs-2.1:free",
                "temperature": 0.2,
                "max_tokens": 1500
            }
        },
        {
            "virtual_key": "analystgpt-nvidia",
            "override_params": {
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "temperature": 0.2,
                "max_tokens": 1500
            }
        },
        {
            "virtual_key": "analystgpt-gemini",
            "override_params": {
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "max_tokens": 1500
            }
        }
    ]
}

from agents.prompts import PLANNER_AGENT, PLANNER_SYSTEM_PROMPT, PLANNER_MODIFY_PROMPT

DEFAULT_AVAILABLE_AGENTS = [
    "DataAnalystAgent",
    "MLAgent",
    "ReporterAgent",
    "VisualizationAgent",
    "DataCleaningAgent"
]


class PlanStatus(str, Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    MODIFIED = "modified"
    CANCELLED = "cancelled"


def infer_task_type(title: str = "", description: str = "", assigned_agent: str = "") -> str:
    """Infers task_type from task title, description, and assigned agent."""
    combined = f"{title} {description} {assigned_agent}".lower()
    if "feature_engineering" in combined or "feature engineering" in combined or "engineer feature" in combined or "derived feature" in combined:
        return "feature_engineering"
    if "hyperparameter_tuning" in combined or "hyperparameter tuning" in combined or "hyperparameter" in combined or "tuning" in combined or "hpt" in combined:
        return "hyperparameter_tuning"
    if "clean" in combined or "inspection" in combined or "imputation" in combined or ("preprocessing" in combined and "ml" not in combined):
        return "cleaning"
    if "model" in combined or "train" in combined or "comparison" in combined or "evaluat" in combined or "predict" in combined or "classif" in combined or "regress" in combined:
        return "model_comparison"
    if "report" in combined or "synthesis" in combined or "executive" in combined:
        return "report"

    agent_lower = (assigned_agent or "").lower()
    if "ml" in agent_lower or "machinelearning" in agent_lower:
        return "model_comparison"
    elif "report" in agent_lower:
        return "report"
    return "stats_and_viz"


@dataclass
class TaskItem:
    task_id: Any = 1
    title: str = ""
    description: str = ""
    assigned_agent: str = "DataAnalystAgent"
    task_type: Optional[str] = None  # cleaning, feature_engineering, stats_and_viz, model_comparison, hyperparameter_tuning, report
    status: str = "pending"  # pending, in_progress, completed, failed, skipped, cancelled
    expected_output: str = ""
    depends_on: List[Any] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    est_tool_calls: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskItem":
        assigned = data.get("assigned_agent") or data.get("agent", "DataAnalystAgent")
        title = data.get("title", "")
        desc = data.get("description") or data.get("objective", "")
        ttype = data.get("task_type") or infer_task_type(title, desc, assigned)

        return cls(
            task_id=data.get("task_id", 1),
            title=title,
            description=desc,
            assigned_agent=assigned,
            task_type=ttype,
            status=data.get("status", "pending"),
            expected_output=data.get("expected_output", ""),
            depends_on=data.get("depends_on", []),
            inputs=data.get("inputs", []),
            result=data.get("result"),
            est_tool_calls=data.get("est_tool_calls")
        )


@dataclass
class ExecutionPlan:
    plan_id: str
    query: str
    selected_agents: List[str]
    tasks: List[TaskItem]
    status: PlanStatus = PlanStatus.DRAFT
    notes: str = ""
    revision_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "selected_agents": self.selected_agents,
            "tasks": [t.to_dict() for t in self.tasks],
            "status": self.status.value if isinstance(self.status, PlanStatus) else str(self.status),
            "notes": self.notes,
            "revision_count": self.revision_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        tasks = [TaskItem.from_dict(t) for t in data.get("tasks", [])]
        status_val = data.get("status", PlanStatus.DRAFT.value)
        try:
            status = PlanStatus(status_val)
        except ValueError:
            status = PlanStatus.DRAFT

        return cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())[:8]),
            query=data.get("query", data.get("user_goal", "")),
            selected_agents=data.get("selected_agents", []),
            tasks=tasks,
            status=status,
            notes=data.get("notes", ""),
            revision_count=data.get("revision_count", 0)
        )


def validate_plan_prerequisites(plan: ExecutionPlan) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Validates structural pipeline prerequisites for Feature Engineering and Hyperparameter Tuning.
    
    Returns:
        (is_valid, errors, details)
        - is_valid: True if plan satisfies all prerequisites (or only needs reordering).
        - errors: List of validation error strings if prerequisites are completely missing.
        - details: Dict with flags (needs_reorder, missing_cleaning, missing_model_comparison).
    """
    errors: list[str] = []
    details: dict[str, Any] = {
        "needs_reorder": False,
        "missing_cleaning": False,
        "missing_model_comparison": False,
        "fe_task_ids": [],
        "hpt_task_ids": [],
    }

    tasks = plan.tasks
    cleaning_indices = [i for i, t in enumerate(tasks) if t.task_type == "cleaning" or "clean" in t.title.lower()]
    model_indices = [i for i, t in enumerate(tasks) if t.task_type == "model_comparison" or ("model" in t.title.lower() and "tuning" not in t.title.lower())]

    # 1. Feature Engineering Check
    for idx, t in enumerate(tasks):
        if t.task_type == "feature_engineering" or ("feature" in t.title.lower() and "engineering" in t.title.lower()):
            details["fe_task_ids"].append(t.task_id)
            if not cleaning_indices:
                details["missing_cleaning"] = True
                errors.append(
                    f"Feature Engineering task #{t.task_id} ('{t.title}') requires a preceding Data Inspection & Cleaning task "
                    f"that produces 'cleaned_dataset.csv'. No cleaning task was found in the plan."
                )
            else:
                earliest_cleaning = min(cleaning_indices)
                if earliest_cleaning > idx:
                    details["needs_reorder"] = True

    # 2. Hyperparameter Tuning Check
    for idx, t in enumerate(tasks):
        if t.task_type == "hyperparameter_tuning" or ("tuning" in t.title.lower() or "hyperparameter" in t.title.lower() or "hpt" in t.title.lower()):
            details["hpt_task_ids"].append(t.task_id)
            if not model_indices:
                details["missing_model_comparison"] = True
                errors.append(
                    f"Hyperparameter Tuning task #{t.task_id} ('{t.title}') requires a preceding Model Training & Comparison task "
                    f"that produces 'model_comparison.csv' and 'best_model.pkl'. No model comparison task was found in the plan."
                )
            else:
                earliest_model = min(model_indices)
                if earliest_model > idx:
                    details["needs_reorder"] = True

    is_valid = len(errors) == 0
    return is_valid, errors, details


def reorder_plan_tasks(plan: ExecutionPlan) -> ExecutionPlan:
    """
    Reorders tasks so that prerequisites strictly precede dependent tasks:
    Cleaning -> Feature Engineering -> Stats/Viz -> Model Comparison -> Hyperparameter Tuning -> Report.
    Re-indexes task IDs (1, 2, ...) and reconciles depends_on / inputs.
    """
    if not plan.tasks:
        return plan

    type_order = {
        "cleaning": 10,
        "feature_engineering": 20,
        "stats_and_viz": 30,
        "model_comparison": 40,
        "hyperparameter_tuning": 50,
        "report": 60
    }

    indexed_tasks = list(enumerate(plan.tasks))
    # Sort by task type precedence, preserving relative order of same-type tasks
    sorted_pairs = sorted(
        indexed_tasks,
        key=lambda pair: (type_order.get(pair[1].task_type or "stats_and_viz", 35), pair[0])
    )
    sorted_tasks = [pair[1] for pair in sorted_pairs]

    # Build old_id -> new_id map
    id_map: dict[Any, int] = {}
    for new_idx, task in enumerate(sorted_tasks, start=1):
        id_map[task.task_id] = new_idx
        id_map[str(task.task_id)] = new_idx

    # Assign new IDs and reconcile dependencies
    cleaning_new_id = None
    fe_new_id = None
    model_new_id = None

    for new_idx, task in enumerate(sorted_tasks, start=1):
        task.task_id = new_idx
        if task.task_type == "cleaning":
            cleaning_new_id = new_idx
        elif task.task_type == "feature_engineering":
            fe_new_id = new_idx
        elif task.task_type == "model_comparison":
            model_new_id = new_idx

    # Reconcile depends_on and dataset inputs
    current_dataset = "cleaned_dataset.csv" if cleaning_new_id else "dataset.csv"
    if fe_new_id:
        current_dataset_after_fe = "feature_engineered_dataset.csv"
    else:
        current_dataset_after_fe = current_dataset

    for task in sorted_tasks:
        # Remap existing depends_on
        remapped_deps = []
        for dep in task.depends_on:
            if dep in id_map:
                remapped_deps.append(id_map[dep])
            elif str(dep) in id_map:
                remapped_deps.append(id_map[str(dep)])

        if task.task_type == "cleaning":
            task.depends_on = [d for d in remapped_deps if d < task.task_id]
            if not task.inputs:
                task.inputs = ["dataset.csv"]
            if not task.expected_output:
                task.expected_output = "cleaned_dataset.csv, cleaning_summary.json"

        elif task.task_type == "feature_engineering":
            if cleaning_new_id and cleaning_new_id not in remapped_deps:
                remapped_deps.append(cleaning_new_id)
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            task.inputs = ["cleaned_dataset.csv"]
            if not task.expected_output:
                task.expected_output = "feature_engineered_dataset.csv, feature_engineering_summary.json"

        elif task.task_type == "stats_and_viz":
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            task.inputs = [current_dataset_after_fe if fe_new_id else current_dataset]
            if not task.expected_output:
                task.expected_output = "analyst_summary.json"

        elif task.task_type == "model_comparison":
            if fe_new_id and fe_new_id not in remapped_deps:
                remapped_deps.append(fe_new_id)
            elif cleaning_new_id and cleaning_new_id not in remapped_deps:
                remapped_deps.append(cleaning_new_id)
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            task.inputs = [current_dataset_after_fe]
            if not task.expected_output:
                task.expected_output = "model_comparison.csv, best_model.pkl, ml_summary.json, predictions.csv, feature_importance.csv"

        elif task.task_type == "hyperparameter_tuning":
            if model_new_id and model_new_id not in remapped_deps:
                remapped_deps.append(model_new_id)
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            task.inputs = ["model_comparison.csv", "best_model.pkl"]
            if not task.expected_output:
                task.expected_output = "tuned_model_comparison.csv, best_params.json, hyperparameter_tuning_summary.json"

        elif task.task_type == "report":
            prior_task_ids = [t.task_id for t in sorted_tasks if t.task_id < task.task_id]
            task.depends_on = prior_task_ids
            if not task.expected_output:
                task.expected_output = "final_report.md"

    plan.tasks = sorted_tasks
    # Update selected_agents
    all_assigned = list(set([t.assigned_agent for t in plan.tasks if t.assigned_agent]))
    plan.selected_agents = sorted(all_assigned, key=lambda a: ["DataAnalystAgent", "MLAgent", "ReporterAgent"].index(a) if a in ["DataAnalystAgent", "MLAgent", "ReporterAgent"] else 99)

    return plan


def insert_missing_prerequisites_fallback(plan: ExecutionPlan, details: dict[str, Any]) -> ExecutionPlan:
    """
    Fallback insertion: Inserts missing prerequisite tasks (Cleaning for FE, Model Comparison for HPT)
    only after the Planner retry loop has exceeded its maximum attempts. Logs a warning.
    """
    inserted_descriptions = []

    if details.get("missing_cleaning"):
        cleaning_task = TaskItem(
            task_id=991,
            title="Data Inspection & Cleaning",
            description="Inspect raw data for duplicates/missing values and anomalies, clean and export cleaned_dataset.csv.",
            assigned_agent="DataAnalystAgent",
            task_type="cleaning",
            status="pending",
            expected_output="cleaned_dataset.csv, cleaning_summary.json",
            depends_on=[],
            inputs=["dataset.csv"],
            est_tool_calls=6
        )
        plan.tasks.append(cleaning_task)
        if "DataAnalystAgent" not in plan.selected_agents:
            plan.selected_agents.append("DataAnalystAgent")
        inserted_descriptions.append("Data Inspection & Cleaning (required by Feature Engineering)")

    if details.get("missing_model_comparison"):
        model_task = TaskItem(
            task_id=992,
            title="Model Training & Comparison",
            description="Train, evaluate, and compare baseline candidate models; export model_comparison.csv and best_model.pkl.",
            assigned_agent="MLAgent",
            task_type="model_comparison",
            status="pending",
            expected_output="model_comparison.csv, best_model.pkl, ml_summary.json",
            depends_on=[],
            inputs=["feature_engineered_dataset.csv" if details.get("missing_cleaning") or any(t.task_type == "feature_engineering" for t in plan.tasks) else "cleaned_dataset.csv"],
            est_tool_calls=6
        )
        plan.tasks.append(model_task)
        if "MLAgent" not in plan.selected_agents:
            plan.selected_agents.append("MLAgent")
        inserted_descriptions.append("Model Training & Comparison (required by Hyperparameter Tuning)")

    warn_msg = f"[Planner Warning] Fallback insertion applied after retry limit: Auto-inserted {', '.join(inserted_descriptions)}."
    if HAS_LOGFIRE:
        logfire.warn(warn_msg, plan_id=plan.plan_id, inserted=inserted_descriptions)
    print(warn_msg, file=sys.stderr)

    plan.notes = (plan.notes + f" | {warn_msg}").strip(" |")
    return reorder_plan_tasks(plan)



def _extract_json(text: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to extract and parse JSON object from LLM string output or envelopes."""
    if isinstance(text, dict):
        if "body" in text or "choices" in text:
            from agents.fallback_parser import unwrap_envelope
            unwrapped = unwrap_envelope(text)
            if isinstance(unwrapped, dict):
                return unwrapped
            text = str(unwrapped)
        else:
            return text

    text_str = str(text).strip()

    # Strip any 'Malformed function call:' prefix
    malformed_match = re.match(r"^Malformed function call:?\s*", text_str, re.IGNORECASE)
    if malformed_match:
        text_str = text_str[malformed_match.end():].strip()

    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_str, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: search for first { to last }
        start = text_str.find("{")
        end = text_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text_str[start : end + 1]
        else:
            json_str = text_str

    parsed = json.loads(json_str)
    if isinstance(parsed, dict) and ("body" in parsed or "choices" in parsed):
        from agents.fallback_parser import unwrap_envelope
        inner = unwrap_envelope(parsed)
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, str) and inner != json_str:
            return _extract_json(inner)
    return parsed



class PlannerAgent:
    def __init__(
        self,
        available_agents: Optional[List[str]] = None,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
        max_tool_calls_per_task: int = 7
    ):
        if not PORTKEY_API_KEY:
            raise ValueError("PORTKEY_API_KEY is not set in environment variables.")

        setup_logfire()

        headers = {
            "x-portkey-api-key": PORTKEY_API_KEY,
        }

        if PORTKEY_CONFIG_ID:
            if HAS_LOGFIRE:
                logfire.info("PlannerAgent setup using Portkey Config ID", config_id=PORTKEY_CONFIG_ID)
            else:
                print(f"[PlannerAgent Setup] Using Portkey Config ID: {PORTKEY_CONFIG_ID}")
            headers["x-portkey-config"] = PORTKEY_CONFIG_ID
        else:
            if HAS_LOGFIRE:
                logfire.warn("[PlannerAgent Setup] PORTKEY_CONFIG_ID not found. Using fallback config.")
            else:
                print("[PlannerAgent Setup] WARNING: PORTKEY_CONFIG_ID not found. Using fallback config.")
            headers["x-portkey-config"] = json.dumps(DEFAULT_PORTKEY_CONFIG)

        self.client = OpenAI(
            base_url="https://api.portkey.ai/v1",
            api_key=PORTKEY_API_KEY,
            default_headers=headers
        )

        self.available_agents = available_agents or DEFAULT_AVAILABLE_AGENTS
        self.model_name = model_name
        self.max_tool_calls_per_task = max_tool_calls_per_task

    def create_plan(
        self,
        user_query: str,
        dataset_context: Optional[str] = None,
        custom_agents: Optional[List[str]] = None,
        max_tool_calls_per_task: Optional[int] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> ExecutionPlan:
        """
        Analyzes user query and produces an ExecutionPlan with selected agents and task list.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name="PlannerAgent starting plan generation")

        agents_list = custom_agents or self.available_agents
        agents_str = ", ".join(agents_list)
        effective_max_tool_calls = (
            max_tool_calls_per_task
            if max_tool_calls_per_task is not None
            else getattr(self, "max_tool_calls_per_task", 7)
        )

        system_msg = PLANNER_SYSTEM_PROMPT.format(
            available_agents_list=agents_str,
            max_tool_calls_per_task=effective_max_tool_calls
        )
        user_msg = f"User Request: {user_query}"
        if dataset_context:
            user_msg += f"\nDataset Context / Files: {dataset_context}"

        messages = sanitize_messages([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ])

        def _do_create_plan():
            if cancellation_token:
                cancellation_token.check_cancelled(step_name="PlannerAgent before LLM plan generation")

            try:
                delays = [10, 20, 30]

                for attempt, delay in enumerate(delays + [0]):

                    try:

                        response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2
                )

                        break

                    except Exception as err:

                        if attempt < len(delays):

                            import time

                            print(f"[PlannerAgent Warning] LLM request failed (attempt {attempt + 1}), retrying in {delay}s... Error: {err}")

                            time.sleep(delay)

                            continue

                        raise err
                if cancellation_token:
                    cancellation_token.check_cancelled(step_name="PlannerAgent after LLM plan generation")

                # 1. Parse initial plan
                raw_content = response.choices[0].message.content
                parsed = _extract_json(raw_content)

                selected_agents = parsed.get("selected_agents", [self.available_agents[0]])
                notes = parsed.get("notes", "")

                tasks = []
                for idx, task_raw in enumerate(parsed.get("tasks", []), start=1):
                    task = TaskItem(
                        task_id=task_raw.get("task_id", idx),
                        title=task_raw.get("title", f"Task {idx}"),
                        description=task_raw.get("description", ""),
                        assigned_agent=task_raw.get("assigned_agent", selected_agents[0]),
                        task_type=task_raw.get("task_type") or infer_task_type(task_raw.get("title", ""), task_raw.get("description", ""), task_raw.get("assigned_agent", selected_agents[0])),
                        status="pending",
                        expected_output=task_raw.get("expected_output", ""),
                        depends_on=task_raw.get("depends_on", []),
                        inputs=task_raw.get("inputs", []),
                        est_tool_calls=task_raw.get("est_tool_calls")
                    )
                    tasks.append(task)

                plan = ExecutionPlan(
                    plan_id=str(uuid.uuid4())[:8],
                    query=user_query,
                    selected_agents=selected_agents,
                    tasks=tasks,
                    status=PlanStatus.DRAFT,
                    notes=notes,
                    revision_count=0
                )

                # 2. Validate prerequisites
                is_valid, errors, details = validate_plan_prerequisites(plan)

                # 3. Retry loop (max 2 iterations) if missing prerequisites
                current_messages = list(messages)
                current_messages.append({"role": "assistant", "content": raw_content})

                attempt = 0
                while not is_valid and errors and attempt < 2:
                    attempt += 1
                    if cancellation_token:
                        cancellation_token.check_cancelled(step_name=f"PlannerAgent retry loop attempt {attempt}")

                    retry_prompt = (
                        f"The execution plan has validation errors that violate mandatory pipeline prerequisites:\n"
                        + "\n".join(f"- {e}" for e in errors)
                        + "\n\nPlease correct the JSON plan:\n"
                        + "- If there is a Feature Engineering task, it MUST be preceded by a Data Inspection & Cleaning task producing 'cleaned_dataset.csv'.\n"
                        + "- If there is a Hyperparameter Tuning task, it MUST be preceded by a Model Training & Comparison task producing 'model_comparison.csv' and 'best_model.pkl'.\n"
                        + "- Ensure depends_on and inputs correctly reflect this sequence."
                    )
                    current_messages.append({"role": "user", "content": retry_prompt})

                    try:
                        delays = [10, 20, 30]

                        for llm_attempt, delay in enumerate(delays + [0]):
                            try:
                                retry_resp = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=sanitize_messages(current_messages),
                                    temperature=0.2
                                )
                                break
                            except Exception as err:
                                if llm_attempt < len(delays):
                                    import time
                                    print(f"[PlannerAgent Warning] LLM request failed (attempt {llm_attempt + 1}), retrying in {delay}s... Error: {err}")
                                    time.sleep(delay)
                                    continue
                                raise err
                            
                        retry_raw = retry_resp.choices[0].message.content
                        current_messages.append({"role": "assistant", "content": retry_raw})
                        retry_parsed = _extract_json(retry_raw)

                        retry_tasks = []
                        retry_agents = retry_parsed.get("selected_agents", plan.selected_agents)
                        for idx, task_raw in enumerate(retry_parsed.get("tasks", []), start=1):
                            rt = TaskItem(
                                task_id=task_raw.get("task_id", idx),
                                title=task_raw.get("title", f"Task {idx}"),
                                description=task_raw.get("description", ""),
                                assigned_agent=task_raw.get("assigned_agent", retry_agents[0]),
                                task_type=task_raw.get("task_type") or infer_task_type(task_raw.get("title", ""), task_raw.get("description", ""), task_raw.get("assigned_agent", retry_agents[0])),
                                status="pending",
                                expected_output=task_raw.get("expected_output", ""),
                                depends_on=task_raw.get("depends_on", []),
                                inputs=task_raw.get("inputs", []),
                                est_tool_calls=task_raw.get("est_tool_calls")
                            )
                            retry_tasks.append(rt)

                        plan = ExecutionPlan(
                            plan_id=plan.plan_id,
                            query=user_query,
                            selected_agents=retry_agents,
                            tasks=retry_tasks,
                            status=PlanStatus.DRAFT,
                            notes=retry_parsed.get("notes", plan.notes),
                            revision_count=attempt
                        )
                        is_valid, errors, details = validate_plan_prerequisites(plan)
                    except Exception as retry_err:
                        print(f"[PlannerAgent Notice] Retry attempt {attempt} failed: {retry_err}", file=sys.stderr)
                        break

                # 4. Fallback insertion if still invalid after max attempts
                if not is_valid and errors:
                    plan = insert_missing_prerequisites_fallback(plan, details)

                # 5. Final reordering to guarantee sequential order and correct dependencies
                plan = reorder_plan_tasks(plan)

                if HAS_LOGFIRE:
                    logfire.info(
                        "PlannerAgent Created Plan",
                        plan_id=plan.plan_id,
                        selected_agents=plan.selected_agents,
                        task_count=len(plan.tasks),
                        task_summary=[{"id": t.task_id, "title": t.title, "agent": t.assigned_agent, "type": t.task_type} for t in plan.tasks],
                        notes=plan.notes
                    )

                return plan

            except WorkflowCancelledException:
                raise
            except Exception as e:
                print(f"[PlannerAgent Error] Failed to generate plan: {e}", file=sys.stderr)
                if HAS_LOGFIRE:
                    logfire.error("PlannerAgent Failed to Generate Plan", error=str(e))
                fallback_task = TaskItem(
                    task_id=1,
                    title="Execute Data Analysis",
                    description=user_query,
                    assigned_agent="DataAnalystAgent",
                    task_type="stats_and_viz",
                    status="pending",
                    expected_output="Analysis output report",
                    est_tool_calls=4
                )
                return ExecutionPlan(
                    plan_id=str(uuid.uuid4())[:8],
                    query=user_query,
                    selected_agents=["DataAnalystAgent"],
                    tasks=[fallback_task],
                    status=PlanStatus.DRAFT,
                    notes=f"Fallback plan generated due to error: {e}",
                    revision_count=0
                )

        if HAS_LOGFIRE:
            with logfire.span("PlannerAgent: Create Plan", user_query=user_query):
                return _do_create_plan()
        else:
            return _do_create_plan()

    def modify_plan(
        self,
        current_plan: ExecutionPlan,
        modification_prompt: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> ExecutionPlan:
        """
        Modifies an existing ExecutionPlan according to user feedback.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name="PlannerAgent before plan modification")

        agents_str = ", ".join(self.available_agents)

        current_tasks_str = json.dumps([t.to_dict() for t in current_plan.tasks], indent=2)

        prompt = PLANNER_MODIFY_PROMPT.format(
            query=current_plan.query,
            current_tasks=current_tasks_str,
            modification_prompt=modification_prompt,
            available_agents_list=agents_str
        )

        messages = sanitize_messages([
            {"role": "user", "content": prompt}
        ])

        try:
            delays = [10, 20, 30]

            for attempt, delay in enumerate(delays + [0]):

                try:

                    response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )

                    break

                except Exception as err:

                    if attempt < len(delays):

                        import time

                        print(f"[PlannerAgent Warning] LLM request failed (attempt {attempt + 1}), retrying in {delay}s... Error: {err}")

                        time.sleep(delay)

                        continue

                    raise err
            if cancellation_token:
                cancellation_token.check_cancelled(step_name="PlannerAgent after plan modification")
            raw_content = response.choices[0].message.content
            parsed = _extract_json(raw_content)

            selected_agents = parsed.get("selected_agents", current_plan.selected_agents)
            notes = parsed.get("notes", current_plan.notes)

            tasks = []
            for idx, task_raw in enumerate(parsed.get("tasks", []), start=1):
                task = TaskItem(
                    task_id=task_raw.get("task_id", idx),
                    title=task_raw.get("title", f"Task {idx}"),
                    description=task_raw.get("description", ""),
                    assigned_agent=task_raw.get("assigned_agent", selected_agents[0]),
                    task_type=task_raw.get("task_type") or infer_task_type(task_raw.get("title", ""), task_raw.get("description", ""), task_raw.get("assigned_agent", selected_agents[0])),
                    status="pending",
                    expected_output=task_raw.get("expected_output", ""),
                    depends_on=task_raw.get("depends_on", []),
                    inputs=task_raw.get("inputs", []),
                    est_tool_calls=task_raw.get("est_tool_calls")
                )
                tasks.append(task)

            updated_plan = ExecutionPlan(
                plan_id=current_plan.plan_id,
                query=current_plan.query,
                selected_agents=selected_agents,
                tasks=tasks,
                status=PlanStatus.MODIFIED,
                notes=notes,
                revision_count=current_plan.revision_count + 1
            )

            # Validate and self-correct if modified plan has prerequisite errors
            is_valid, errors, details = validate_plan_prerequisites(updated_plan)
            if not is_valid and errors:
                updated_plan = insert_missing_prerequisites_fallback(updated_plan, details)

            updated_plan = reorder_plan_tasks(updated_plan)

            if HAS_LOGFIRE:
                logfire.info("Execution plan modified", plan_id=updated_plan.plan_id, revision=updated_plan.revision_count)

            return updated_plan

        except Exception as e:
            print(f"[PlannerAgent Error] Failed to modify plan: {e}", file=sys.stderr)
            current_plan.notes += f" | Failed modification attempt: {e}"
            return current_plan

    def confirm_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Confirms the plan, marking it ready for multi-agent execution.
        """
        plan.status = PlanStatus.CONFIRMED
        if HAS_LOGFIRE:
            logfire.info("Execution plan confirmed", plan_id=plan.plan_id)
        return plan

    def cancel_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Cancels the workflow execution plan.
        Returns cancellation state and message.
        """
        plan.status = PlanStatus.CANCELLED
        message = "user cenceled the process"
        if HAS_LOGFIRE:
            logfire.info("Execution plan cancelled", plan_id=plan.plan_id, message=message)

        return {
            "status": "cancelled",
            "message": message,
            "plan": plan
        }

    def format_plan_for_display(self, plan: ExecutionPlan) -> str:
        """
        Formats plan as a clean markdown table / string for user review.
        """
        lines = []
        lines.append(f"### 📋 Execution Plan [{plan.plan_id}] (Status: {plan.status.value.upper()})")
        lines.append(f"**Query:** {plan.query}")
        lines.append(f"**Selected Agents:** {', '.join(plan.selected_agents)}")
        if plan.notes:
            lines.append(f"**Notes:** {plan.notes}")
        lines.append("\n**Task Checklist:**")

        for t in plan.tasks:
            lines.append(f"  {t.task_id}. [{t.assigned_agent}] **{t.title}**")
            lines.append(f"     - *Description:* {t.description}")
            if t.expected_output:
                lines.append(f"     - *Expected Output:* {t.expected_output}")

        return "\n".join(lines)

    def run_interactive_flow(
        self,
        user_query: str,
        dataset_context: Optional[str] = None,
        input_func: Callable[[str], str] = input,
        print_func: Callable[[str], None] = print
    ) -> Optional[ExecutionPlan]:
        """
        Interactive loop that shows query & task list to the user,
        allowing confirm (proceed), modify (re-plan), or cancel (terminate).
        """
        print_func("\n=== 🤖 Planner Agent Generating Plan ===")
        plan = self.create_plan(user_query, dataset_context=dataset_context)

        while True:
            print_func("\n" + self.format_plan_for_display(plan))
            print_func("\nOptions:")
            print_func("  [1] Confirm / Proceed")
            print_func("  [2] Modify Plan")
            print_func("  [3] Cancel")

            user_choice = input_func("\nEnter your choice (1/2/3 or confirm/modify/cancel): ").strip().lower()

            if user_choice in ["1", "confirm", "proceed", "yes", "y"]:
                confirmed_plan = self.confirm_plan(plan)
                print_func("\n✅ Plan confirmed! Handing over to agent workflow...\n")
                return confirmed_plan

            elif user_choice in ["2", "modify", "edit", "m"]:
                mod_prompt = input_func("\nEnter modification instructions (e.g. 'Add outlier check step'): ").strip()
                if mod_prompt:
                    print_func("\n🔄 Re-planning based on your feedback...")
                    plan = self.modify_plan(plan, mod_prompt)

            elif user_choice in ["3", "cancel", "c", "exit", "quit"]:
                cancel_result = self.cancel_plan(plan)
                print_func(f"\n❌ {cancel_result['message']}\n")
                return None

            else:
                print_func("Invalid selection. Please choose 1 (Confirm), 2 (Modify), or 3 (Cancel).")


if __name__ == "__main__":
    # Self-test / Demo script
    planner = PlannerAgent()
    demo_query = "Clean sales.csv, calculate total revenue per region, plot top sales charts, and write a summary report."
    print("Testing PlannerAgent interactive flow demo (Dry-run mode)...")
    plan = planner.create_plan(demo_query)
    print(planner.format_plan_for_display(plan))
