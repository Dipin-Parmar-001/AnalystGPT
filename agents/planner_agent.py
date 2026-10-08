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
from agents.portkey_routing import create_with_thinking_route, extract_reasoning_text


try:
    import logfire
except ImportError:
    logfire = None

# Load environment variables
load_dotenv()

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")

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
    PAUSED = "paused"


def is_model_training_title(title: str) -> bool:
    """Returns True if the title clearly indicates model training/comparison rather than evaluation or tuning."""
    t = (title or "").lower().strip()
    return any(k in t for k in (
        "model training", "training & comparison", "model comparison",
        "train and compare", "candidate model", "baseline model",
        "train model", "model selection", "train classifier", "train regressor",
        "fit model"
    )) and not any(k in t for k in ("tuning", "hyperparameter", "hpt", "threshold", "final test evaluation", "holdout evaluation"))


def infer_task_type(title: str = "", description: str = "", assigned_agent: str = "") -> str:
    """Infers task_type from task title, description, and assigned agent with priority given to title intent."""
    t_lower = (title or "").lower().strip()
    d_lower = (description or "").lower().strip()
    a_lower = (assigned_agent or "").lower().strip()

    # 1. Primary classification based on task title (strongest intent signal)
    if any(k in t_lower for k in ("clean", "inspection", "missing value", "imputation", "deduplicat")):
        return "cleaning"
    if any(k in t_lower for k in ("feature engineering", "feature_engineering", "derived feature", "engineer feature", "feature transformation")):
        return "feature_engineering"
    if any(k in t_lower for k in ("hyperparameter", "tuning", "tune model", "hpt", "grid search", "random search", "randomizedsearch", "optuna")):
        return "hyperparameter_tuning"
    if any(k in t_lower for k in ("threshold selection", "threshold tuning", "validation threshold", "test evaluation", "holdout evaluation", "final evaluation", "testing and evaluation", "test metric")) or (
        any(k in t_lower for k in ("threshold", "test evaluation", "evaluation", "testing"))
        and not any(k in t_lower for k in ("train", "model training", "comparison", "tuning", "hyperparameter"))
    ):
        return "model_evaluation"
    if is_model_training_title(t_lower):
        return "model_comparison"
    if any(k in t_lower for k in ("unsupervised", "clustering", "cluster", "anomaly detection", "outlier detection", "isolation forest", "local outlier factor", "dbscan", "kmeans", "k-means", "umap")):
        return "unsupervised_analysis"
    if any(k in t_lower for k in ("descriptive stat", "stats and viz", "stats & viz", "exploratory", "eda", "visualization", "visualize", "distribution analysis", "chart", "plot")):
        return "stats_and_viz"
    if any(k in t_lower for k in ("report", "synthesis", "executive summary", "final report")):
        return "report"

    # 2. Secondary classification based on description if title was ambiguous
    if any(k in d_lower for k in ("hyperparameter_tuning", "hyperparameter tuning", "gridsearch", "randomizedsearch", "tune hyperparameter")):
        return "hyperparameter_tuning"
    if any(k in d_lower for k in ("clean", "inspection", "imputation")) and not any(k in d_lower for k in ("train", "model")):
        return "cleaning"
    if any(k in d_lower for k in ("feature engineering", "feature_engineering", "derived feature", "engineer feature")) and "train" not in d_lower:
        return "feature_engineering"
    if any(k in d_lower for k in ("unsupervised", "clustering", "isolation forest", "local outlier factor", "dbscan", "kmeans")) and "supervised" not in d_lower:
        return "unsupervised_analysis"
    if any(k in d_lower for k in ("threshold selection", "threshold tuning", "validation threshold selection", "held-out test evaluation", "evaluate on held-out test")) and not any(k in d_lower for k in ("train and compare", "model training", "train 4 models", "fit standardscaler")):
        return "model_evaluation"
    if any(k in d_lower for k in ("model", "train", "comparison", "candidate", "classifier", "regressor", "baseline")):
        return "model_comparison"
    if any(k in d_lower for k in ("report", "synthesis", "executive")):
        return "report"

    # 3. Agent-based fallback
    if "ml" in a_lower or "machinelearning" in a_lower:
        return "model_comparison"
    elif "report" in a_lower:
        return "report"
    return "stats_and_viz"


@dataclass
class TaskItem:
    task_id: Any = 1
    title: str = ""
    description: str = ""
    assigned_agent: str = "DataAnalystAgent"
    task_type: Optional[str] = None  # cleaning, feature_engineering, stats_and_viz, model_comparison, unsupervised_analysis, hyperparameter_tuning, report
    status: str = "pending"  # pending, in_progress, completed, failed, skipped, cancelled
    expected_output: str = ""
    depends_on: List[Any] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    est_tool_calls: Optional[int] = None
    primary_metric: Optional[str] = None  # e.g., "recall", "roc_auc", "pr_auc", "f1", "accuracy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskItem":
        assigned = data.get("assigned_agent") or data.get("agent", "DataAnalystAgent")
        title = data.get("title", "")
        desc = data.get("description") or data.get("objective", "")
        raw_type = data.get("task_type")
        if not raw_type:
            ttype = infer_task_type(title, desc, assigned)
        elif raw_type == "model_comparison":
            if is_model_training_title(title):
                ttype = "model_comparison"
            else:
                inferred = infer_task_type(title, desc, assigned)
                ttype = inferred if inferred in {"model_evaluation", "hyperparameter_tuning", "unsupervised_analysis"} else raw_type
        else:
            ttype = raw_type

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
            est_tool_calls=data.get("est_tool_calls"),
            primary_metric=data.get("primary_metric")
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
    cleaning_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "cleaning" or any(k in (t.title or "").lower() for k in ("clean", "inspection"))
    ]
    model_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "model_comparison" or (
            is_model_training_title(t.title or "")
            or ("model" in (t.title or "").lower() and not any(k in (t.title or "").lower() for k in ("tuning", "hyperparameter", "threshold", "test evaluation", "unsupervised", "clustering")))
        )
    ]

    # 1. Feature Engineering Check
    for idx, t in enumerate(tasks):
        if t.task_type == "feature_engineering" or ("feature" in (t.title or "").lower() and "engineering" in (t.title or "").lower()):
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
        if t.task_type == "hyperparameter_tuning" or ("tuning" in (t.title or "").lower() or "hyperparameter" in (t.title or "").lower() or "hpt" in (t.title or "").lower()):
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


TYPE_ORDER: Dict[str, int] = {
    "cleaning": 10,
    "feature_engineering": 20,
    "stats_and_viz": 30,
    "unsupervised_analysis": 35,
    "unsupervised_learning": 35,
    "clustering": 35,
    "anomaly_detection": 35,
    "model_comparison": 40,
    "hyperparameter_tuning": 50,
    "model_evaluation": 55,
    "evaluation": 55,
    "testing_and_evaluation": 55,
    "test_evaluation": 55,
    "threshold_selection": 55,
    "testing": 55,
    "report": 60,
}


def get_task_priority(task: TaskItem) -> int:
    """Returns priority weight for a task; lower values indicate earlier desired execution."""
    ttype = (task.task_type or "").lower().strip()
    title = (task.title or "").lower().strip()

    # 1. If explicit recognized task_type is present, respect it directly
    if ttype in TYPE_ORDER:
        return TYPE_ORDER[ttype]

    # 2. Check title intent
    if is_model_training_title(title):
        return TYPE_ORDER.get("model_comparison", 40)
    if any(k in title for k in ("tuning", "hyperparameter", "hpt")):
        return TYPE_ORDER.get("hyperparameter_tuning", 50)
    if any(k in title for k in ("threshold", "test evaluation", "holdout evaluation", "threshold selection", "final evaluation")):
        return TYPE_ORDER.get("model_evaluation", 55)
    if any(k in title for k in ("unsupervised", "clustering", "anomaly detection", "kmeans", "dbscan")):
        return TYPE_ORDER.get("unsupervised_analysis", 35)

    # 3. Fallback to infer_task_type
    inferred = infer_task_type(task.title, task.description, task.assigned_agent)
    return TYPE_ORDER.get(inferred, 35)


def reorder_plan_tasks(plan: ExecutionPlan) -> ExecutionPlan:
    """
    Reorders tasks so that prerequisites strictly precede dependent tasks:
    Cleaning -> Feature Engineering -> Stats/Viz -> Model Comparison -> Hyperparameter Tuning -> Report.
    Re-indexes task IDs (1, 2, ...) and reconciles depends_on / inputs.
    """
    if not plan.tasks:
        return plan

    tasks = list(plan.tasks)
    n = len(tasks)

    # 1. Normalization of IDs and task types
    for t in tasks:
        if not t.task_type:
            t.task_type = infer_task_type(t.title, t.description, t.assigned_agent)
        elif t.task_type == "model_comparison" and not is_model_training_title(t.title):
            inferred = infer_task_type(t.title, t.description, t.assigned_agent)
            if inferred in {"model_evaluation", "hyperparameter_tuning", "unsupervised_analysis"}:
                t.task_type = inferred

    # Build old ID to task index lookup
    id_to_idx: Dict[Any, int] = {}
    for idx, t in enumerate(tasks):
        id_to_idx[t.task_id] = idx
        id_to_idx[str(t.task_id)] = idx

    # 2. Identify key milestone task indices
    cleaning_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "cleaning" or any(k in (t.title or "").lower() for k in ("clean", "inspection"))
    ]
    fe_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "feature_engineering" or ("feature" in (t.title or "").lower() and "engineering" in (t.title or "").lower())
    ]
    model_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "model_comparison" or (
            is_model_training_title(t.title or "")
            or ("model" in (t.title or "").lower() and not any(k in (t.title or "").lower() for k in ("tuning", "hyperparameter", "threshold", "test evaluation", "unsupervised", "clustering")))
        )
    ]
    hpt_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "hyperparameter_tuning" or any(k in (t.title or "").lower() for k in ("tuning", "hyperparameter", "hpt", "grid search", "random search"))
    ]
    eval_indices = [
        i for i, t in enumerate(tasks)
        if i not in model_indices and i not in hpt_indices and (
            t.task_type in {"model_evaluation", "evaluation", "testing_and_evaluation", "test_evaluation", "threshold_selection", "testing"}
            or any(k in (t.title or "").lower() for k in ("threshold", "test evaluation", "holdout evaluation", "final evaluation"))
        )
    ]
    report_indices = [
        i for i, t in enumerate(tasks)
        if t.task_type == "report" or any(k in (t.title or "").lower() for k in ("report", "synthesis", "executive"))
    ]

    earliest_cleaning_idx = cleaning_indices[0] if cleaning_indices else None
    earliest_fe_idx = fe_indices[0] if fe_indices else None
    earliest_model_idx = model_indices[0] if model_indices else None
    earliest_hpt_idx = hpt_indices[0] if hpt_indices else None

    # 3. Build dependency graph (adjacency list and in-degrees)
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    in_degree: Dict[int, int] = {i: 0 for i in range(n)}

    def add_dep_edge(prereq_idx: int, dependent_idx: int):
        if prereq_idx == dependent_idx:
            return
        # Fundamental procedure invariant:
        # A task of a strictly higher pipeline stage cannot be a prerequisite for a strictly earlier pipeline stage.
        p_priority = get_task_priority(tasks[prereq_idx])
        d_priority = get_task_priority(tasks[dependent_idx])
        if p_priority > d_priority:
            # Backward edge violates ML pipeline procedure (e.g. HPT -> Model Training)
            return
        if dependent_idx not in adj[prereq_idx]:
            adj[prereq_idx].add(dependent_idx)
            in_degree[dependent_idx] += 1

    # Add explicit dependencies from task.depends_on
    for i, t in enumerate(tasks):
        raw_deps = getattr(t, "depends_on", []) or []
        for dep in raw_deps:
            if dep in id_to_idx:
                p_idx = id_to_idx[dep]
                add_dep_edge(p_idx, i)

    # Add structural prerequisite edges if not already present
    if earliest_cleaning_idx is not None:
        for fe_i in fe_indices:
            add_dep_edge(earliest_cleaning_idx, fe_i)

    if earliest_model_idx is not None:
        for hpt_i in hpt_indices:
            add_dep_edge(earliest_model_idx, hpt_i)

    # Evaluation / threshold tasks should depend on HPT (if present) or baseline model (if no HPT)
    eval_target_idx = earliest_hpt_idx if earliest_hpt_idx is not None else earliest_model_idx
    if eval_target_idx is not None:
        for ev_i in eval_indices:
            add_dep_edge(eval_target_idx, ev_i)

    # Report task should depend on all prior non-report tasks if no explicit deps or to guarantee last execution
    for rep_i in report_indices:
        for other_i in range(n):
            if other_i != rep_i and other_i not in report_indices:
                add_dep_edge(other_i, rep_i)

    # 4. Kahn's Priority-Guided Topological Sort
    scheduled_indices: List[int] = []
    scheduled_set: Set[int] = set()
    ready_pool: List[int] = [i for i in range(n) if in_degree[i] == 0]

    while len(scheduled_indices) < n:
        if not ready_pool:
            # Cycle detected or unreachable tasks; break cycle by choosing remaining task with lowest in-degree
            remaining = [i for i in range(n) if i not in scheduled_set]
            best_candidate = min(
                remaining,
                key=lambda idx: (in_degree[idx], get_task_priority(tasks[idx]), idx)
            )
            ready_pool.append(best_candidate)

        # Pick from ready_pool the task with the highest priority (lowest weight), tie-broken by original index
        ready_pool.sort(key=lambda idx: (get_task_priority(tasks[idx]), idx))
        chosen_idx = ready_pool.pop(0)

        if chosen_idx in scheduled_set:
            continue

        scheduled_indices.append(chosen_idx)
        scheduled_set.add(chosen_idx)

        for dependent_idx in adj[chosen_idx]:
            in_degree[dependent_idx] -= 1
            if in_degree[dependent_idx] == 0 and dependent_idx not in scheduled_set:
                ready_pool.append(dependent_idx)

    sorted_tasks = [tasks[idx] for idx in scheduled_indices]

    # Build old_id -> new_id map
    id_map: dict[Any, int] = {}
    for new_idx, task in enumerate(sorted_tasks, start=1):
        id_map[task.task_id] = new_idx
        id_map[str(task.task_id)] = new_idx

    # Assign new IDs and reconcile dependencies
    cleaning_new_id = None
    fe_new_id = None
    model_new_id = None
    hpt_new_id = None

    for new_idx, task in enumerate(sorted_tasks, start=1):
        task.task_id = new_idx
        if task.task_type == "cleaning":
            cleaning_new_id = new_idx
        elif task.task_type == "feature_engineering":
            fe_new_id = new_idx
        elif task.task_type == "model_comparison" and model_new_id is None:
            model_new_id = new_idx
        elif task.task_type == "hyperparameter_tuning" and hpt_new_id is None:
            hpt_new_id = new_idx

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
            if not task.inputs:
                task.inputs = ["cleaned_dataset.csv"]
            if not task.expected_output:
                task.expected_output = "feature_engineered_dataset.csv, feature_engineering_summary.json"

        elif task.task_type == "stats_and_viz":
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            if not task.inputs:
                task.inputs = [current_dataset_after_fe if fe_new_id else current_dataset]
            if not task.expected_output:
                task.expected_output = "analyst_summary.json"

        elif task.task_type in {"model_comparison", "unsupervised_analysis"}:
            if fe_new_id and fe_new_id not in remapped_deps:
                remapped_deps.append(fe_new_id)
            elif cleaning_new_id and cleaning_new_id not in remapped_deps:
                remapped_deps.append(cleaning_new_id)
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            if not task.inputs:
                task.inputs = [current_dataset_after_fe]
            if not task.expected_output:
                task.expected_output = (
                    "unsupervised_metrics.json, unsupervised_summary.json, ml_summary.json"
                    if task.task_type == "unsupervised_analysis"
                    else "model_comparison.csv, best_model.pkl, ml_summary.json, predictions.csv, feature_importance.csv"
                )

        elif task.task_type == "hyperparameter_tuning":
            if model_new_id and model_new_id not in remapped_deps:
                remapped_deps.append(model_new_id)
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            if not task.inputs:
                task.inputs = ["model_comparison.csv", "best_model.pkl"]
            if not task.expected_output:
                task.expected_output = "tuned_model_comparison.csv, best_params.json, hyperparameter_tuning_summary.json"

        elif task.task_type in {"model_evaluation", "evaluation", "testing_and_evaluation", "threshold_selection", "test_evaluation"}:
            prior_model_dep = hpt_new_id if (hpt_new_id and hpt_new_id < task.task_id) else model_new_id
            if prior_model_dep and prior_model_dep < task.task_id and prior_model_dep not in remapped_deps:
                remapped_deps.append(prior_model_dep)
            task.depends_on = sorted(list(set([d for d in remapped_deps if d < task.task_id])))
            if not task.inputs:
                task.inputs = [current_dataset_after_fe, "champion_model.json"]
            if not task.expected_output:
                task.expected_output = "validation_threshold_analysis.csv, test_metrics.json, test_predictions.csv"

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
            if isinstance(unwrapped, list):
                return {"tasks": unwrapped}
            text = str(unwrapped)
        else:
            return text

    text_str = str(text or "").strip()
    if not text_str:
        return {}

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

    try:
        parsed = json.loads(json_str)
    except Exception:
        # Try finding outer braces if json_str failed
        match_obj = re.search(r"(\{.*\})", text_str, re.DOTALL)
        if match_obj:
            try:
                parsed = json.loads(match_obj.group(1))
            except Exception:
                parsed = {}
        else:
            # Check for JSON array fallback
            array_match = re.search(r"(\[.*\])", text_str, re.DOTALL)
            if array_match:
                try:
                    parsed = json.loads(array_match.group(1))
                except Exception:
                    parsed = {}
            else:
                parsed = {}

    if isinstance(parsed, list):
        return {"tasks": parsed}
    if not isinstance(parsed, dict):
        return {}

    if "body" in parsed or "choices" in parsed:
        from agents.fallback_parser import unwrap_envelope
        inner = unwrap_envelope(parsed)
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, list):
            return {"tasks": inner}
        if isinstance(inner, str) and inner != json_str:
            return _extract_json(inner)
    return parsed


def extract_response_message(response: Any) -> Any:
    """Safely extracts message object from LLM response or Portkey envelope."""
    if response is None:
        raise ValueError("LLM returned None response")
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if choices is None or not isinstance(choices, (list, tuple)) or len(choices) == 0:
        raise ValueError(f"LLM response contained no valid choices: {response}")
    choice0 = choices[0]
    if choice0 is None:
        raise ValueError(f"LLM response choice[0] is None: {response}")
    if isinstance(choice0, dict):
        msg = choice0.get("message")
    else:
        msg = getattr(choice0, "message", None)
    if msg is None:
        raise ValueError(f"LLM choice has no message: {choice0}")
    return msg


def extract_message_content(message: Any) -> str:
    """Safely extracts string content from message."""
    if message is None:
        return ""
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", None) or "")



class PlannerAgent:
    def __init__(
        self,
        available_agents: Optional[List[str]] = None,
        model_name: str = "nemotron-3-ultra-550b-a55b",
        max_tool_calls_per_task: int = 7
    ):
        if not PORTKEY_API_KEY:
            raise ValueError("PORTKEY_API_KEY is not set in environment variables.")

        setup_logfire()

        headers = {"x-portkey-api-key": PORTKEY_API_KEY}
        if PORTKEY_CONFIG_ID:
            headers["x-portkey-config"] = PORTKEY_CONFIG_ID
            if HAS_LOGFIRE:
                logfire.info("PlannerAgent setup using Portkey Config ID", config_id=PORTKEY_CONFIG_ID)
            else:
                print(f"[PlannerAgent Setup] Using Portkey Config ID: {PORTKEY_CONFIG_ID}")
        else:
            headers["x-portkey-virtual-key"] = "analystgpt-openrouter"

        self.client = OpenAI(
            base_url="https://api.portkey.ai/v1",
            api_key=PORTKEY_API_KEY,
            default_headers=headers
        )

        # Local observability: per-call latency, served model, token metrics
        from agents.observability import instrument_openai_client
        instrument_openai_client(self.client, agent_name="PlannerAgent")

        self.available_agents = available_agents or DEFAULT_AVAILABLE_AGENTS
        self.model_name = model_name[len("nvidia/"):] if model_name.startswith("nvidia/") else model_name
        self.max_tool_calls_per_task = max_tool_calls_per_task

    def create_plan(
        self,
        user_query: str,
        dataset_context: Optional[str] = None,
        custom_agents: Optional[List[str]] = None,
        max_tool_calls_per_task: Optional[int] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
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
                delays = [5, 10, 15]
                response = None
                response_message = None
                raw_content = ""

                for attempt, delay in enumerate(delays + [0]):
                    try:
                        response = create_with_thinking_route(
                            self.client,
                            model=self.model_name,
                            messages=messages,
                            temperature=0.2,
                        )
                        response_message = extract_response_message(response)
                        raw_content = extract_message_content(response_message)
                        if not raw_content.strip():
                            raise RuntimeError("LLM response contained empty content")
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
                if step_callback and (raw_content or extract_reasoning_text(response_message)):
                    try:
                        step_callback("llm_thought", {
                            "agent": "PlannerAgent",
                            "turn": 1,
                            "thought": raw_content.strip(),
                            "reasoning": extract_reasoning_text(response_message),
                        })
                    except Exception:
                        pass
                parsed = _extract_json(raw_content)

                # Safely resolve selected_agents
                raw_selected = parsed.get("selected_agents") if isinstance(parsed, dict) else None
                if isinstance(raw_selected, list) and len(raw_selected) > 0:
                    selected_agents = [str(a).strip() for a in raw_selected if a and str(a).strip()]
                else:
                    selected_agents = []
                if not selected_agents:
                    avail = [a for a in (self.available_agents or DEFAULT_AVAILABLE_AGENTS) if a]
                    selected_agents = avail if avail else ["DataAnalystAgent", "MLAgent", "ReporterAgent"]
                default_agent = selected_agents[0]

                notes = str(parsed.get("notes") or "") if isinstance(parsed, dict) else ""

                tasks = []
                raw_tasks = parsed.get("tasks") if isinstance(parsed, dict) else None
                if not isinstance(raw_tasks, list):
                    raw_tasks = []

                for idx, task_raw in enumerate(raw_tasks, start=1):
                    if not isinstance(task_raw, dict):
                        continue
                    assigned = task_raw.get("assigned_agent")
                    if not assigned or not isinstance(assigned, str) or not assigned.strip():
                        assigned = default_agent
                    else:
                        assigned = assigned.strip()

                    title = str(task_raw.get("title") or f"Task {idx}").strip()
                    desc = str(task_raw.get("description") or "").strip()
                    raw_type = task_raw.get("task_type")
                    task_type = (str(raw_type).strip() if raw_type else None) or infer_task_type(title, desc, assigned)

                    raw_deps = task_raw.get("depends_on")
                    depends_on = [d for d in raw_deps if d is not None] if isinstance(raw_deps, list) else []

                    raw_inputs = task_raw.get("inputs")
                    inputs = [str(inp) for inp in raw_inputs if inp is not None] if isinstance(raw_inputs, list) else []

                    task = TaskItem(
                        task_id=task_raw.get("task_id", idx),
                        title=title,
                        description=desc,
                        assigned_agent=assigned,
                        task_type=task_type,
                        status="pending",
                        expected_output=str(task_raw.get("expected_output") or ""),
                        depends_on=depends_on,
                        inputs=inputs,
                        est_tool_calls=task_raw.get("est_tool_calls"),
                        primary_metric=task_raw.get("primary_metric")
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
                        delays = [5, 10, 15]
                        retry_resp = None
                        retry_message = None
                        retry_raw = ""

                        for llm_attempt, delay in enumerate(delays + [0]):
                            try:
                                retry_resp = create_with_thinking_route(
                                    self.client,
                                    model=self.model_name,
                                    messages=sanitize_messages(current_messages),
                                    temperature=0.2,
                                )
                                retry_message = extract_response_message(retry_resp)
                                retry_raw = extract_message_content(retry_message)
                                if not retry_raw.strip():
                                    raise RuntimeError("LLM returned empty content in retry")
                                break
                            except Exception as err:
                                if llm_attempt < len(delays):
                                    import time
                                    print(f"[PlannerAgent Warning] LLM request failed (attempt {llm_attempt + 1}), retrying in {delay}s... Error: {err}")
                                    time.sleep(delay)
                                    continue
                                raise err

                        if step_callback and (retry_raw or extract_reasoning_text(retry_message)):
                            try:
                                step_callback("llm_thought", {
                                    "agent": "PlannerAgent",
                                    "turn": attempt + 1,
                                    "thought": retry_raw.strip(),
                                    "reasoning": extract_reasoning_text(retry_message),
                                })
                            except Exception:
                                pass
                        current_messages.append({"role": "assistant", "content": retry_raw})
                        retry_parsed = _extract_json(retry_raw)

                        raw_retry_agents = retry_parsed.get("selected_agents") if isinstance(retry_parsed, dict) else None
                        if isinstance(raw_retry_agents, list) and len(raw_retry_agents) > 0:
                            retry_agents = [str(a).strip() for a in raw_retry_agents if a and str(a).strip()]
                        else:
                            retry_agents = plan.selected_agents or list(self.available_agents or DEFAULT_AVAILABLE_AGENTS)
                        if not retry_agents:
                            retry_agents = ["DataAnalystAgent", "MLAgent", "ReporterAgent"]
                        retry_default_agent = retry_agents[0]

                        retry_tasks = []
                        raw_retry_tasks = retry_parsed.get("tasks") if isinstance(retry_parsed, dict) else None
                        if not isinstance(raw_retry_tasks, list):
                            raw_retry_tasks = []

                        for idx, task_raw in enumerate(raw_retry_tasks, start=1):
                            if not isinstance(task_raw, dict):
                                continue
                            assigned = task_raw.get("assigned_agent")
                            if not assigned or not isinstance(assigned, str) or not assigned.strip():
                                assigned = retry_default_agent
                            else:
                                assigned = assigned.strip()

                            title = str(task_raw.get("title") or f"Task {idx}").strip()
                            desc = str(task_raw.get("description") or "").strip()
                            raw_type = task_raw.get("task_type")
                            task_type = (str(raw_type).strip() if raw_type else None) or infer_task_type(title, desc, assigned)

                            raw_deps = task_raw.get("depends_on")
                            depends_on = [d for d in raw_deps if d is not None] if isinstance(raw_deps, list) else []

                            raw_inputs = task_raw.get("inputs")
                            inputs = [str(inp) for inp in raw_inputs if inp is not None] if isinstance(raw_inputs, list) else []

                            rt = TaskItem(
                                task_id=task_raw.get("task_id", idx),
                                title=title,
                                description=desc,
                                assigned_agent=assigned,
                                task_type=task_type,
                                status="pending",
                                expected_output=str(task_raw.get("expected_output") or ""),
                                depends_on=depends_on,
                                inputs=inputs,
                                est_tool_calls=task_raw.get("est_tool_calls"),
                                primary_metric=task_raw.get("primary_metric")
                            )
                            retry_tasks.append(rt)

                        retry_notes = str(retry_parsed.get("notes") or plan.notes) if isinstance(retry_parsed, dict) else plan.notes
                        plan = ExecutionPlan(
                            plan_id=plan.plan_id,
                            query=user_query,
                            selected_agents=retry_agents,
                            tasks=retry_tasks,
                            status=PlanStatus.DRAFT,
                            notes=retry_notes,
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
                import traceback
                traceback.print_exc()
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
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
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
            delays = [5, 10, 15]
            response = None
            response_message = None
            raw_content = ""

            for attempt, delay in enumerate(delays + [0]):
                try:
                    response = create_with_thinking_route(
                        self.client,
                        model=self.model_name,
                        messages=messages,
                        temperature=0.2,
                    )
                    response_message = extract_response_message(response)
                    raw_content = extract_message_content(response_message)
                    if not raw_content.strip():
                        raise RuntimeError("LLM response contained empty content in modify_plan")
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
            if step_callback and (raw_content or extract_reasoning_text(response_message)):
                try:
                    step_callback("llm_thought", {
                        "agent": "PlannerAgent",
                        "turn": 1,
                        "thought": raw_content.strip(),
                        "reasoning": extract_reasoning_text(response_message),
                    })
                except Exception:
                    pass
            parsed = _extract_json(raw_content)

            raw_selected = parsed.get("selected_agents") if isinstance(parsed, dict) else None
            if isinstance(raw_selected, list) and len(raw_selected) > 0:
                selected_agents = [str(a).strip() for a in raw_selected if a and str(a).strip()]
            else:
                selected_agents = current_plan.selected_agents or list(self.available_agents or DEFAULT_AVAILABLE_AGENTS)
            if not selected_agents:
                selected_agents = ["DataAnalystAgent", "MLAgent", "ReporterAgent"]
            default_agent = selected_agents[0]

            notes = str(parsed.get("notes") or current_plan.notes) if isinstance(parsed, dict) else current_plan.notes

            tasks = []
            raw_tasks = parsed.get("tasks") if isinstance(parsed, dict) else None
            if not isinstance(raw_tasks, list):
                raw_tasks = []

            for idx, task_raw in enumerate(raw_tasks, start=1):
                if not isinstance(task_raw, dict):
                    continue
                assigned = task_raw.get("assigned_agent")
                if not assigned or not isinstance(assigned, str) or not assigned.strip():
                    assigned = default_agent
                else:
                    assigned = assigned.strip()

                title = str(task_raw.get("title") or f"Task {idx}").strip()
                desc = str(task_raw.get("description") or "").strip()
                raw_type = task_raw.get("task_type")
                task_type = (str(raw_type).strip() if raw_type else None) or infer_task_type(title, desc, assigned)

                raw_deps = task_raw.get("depends_on")
                depends_on = [d for d in raw_deps if d is not None] if isinstance(raw_deps, list) else []

                raw_inputs = task_raw.get("inputs")
                inputs = [str(inp) for inp in raw_inputs if inp is not None] if isinstance(raw_inputs, list) else []

                task = TaskItem(
                    task_id=task_raw.get("task_id", idx),
                    title=title,
                    description=desc,
                    assigned_agent=assigned,
                    task_type=task_type,
                    status="pending",
                    expected_output=str(task_raw.get("expected_output") or ""),
                    depends_on=depends_on,
                    inputs=inputs,
                    est_tool_calls=task_raw.get("est_tool_calls"),
                    primary_metric=task_raw.get("primary_metric")
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
            import traceback
            traceback.print_exc()
            print(f"[PlannerAgent Error] Failed to modify plan: {e}", file=sys.stderr)
            fallback_tasks = list(current_plan.tasks)
            int_ids = [t.task_id for t in fallback_tasks if isinstance(t.task_id, int)]
            next_id = (max(int_ids) + 1) if int_ids else (len(fallback_tasks) + 1)
            fallback_tasks.append(TaskItem(
                task_id=next_id,
                title=f"User Modification: {modification_prompt[:50]}",
                description=modification_prompt,
                assigned_agent=current_plan.selected_agents[0] if current_plan.selected_agents else "DataAnalystAgent",
                task_type="stats_and_viz",
                status="pending",
                expected_output="Analysis output update"
            ))
            return ExecutionPlan(
                plan_id=current_plan.plan_id,
                query=current_plan.query,
                selected_agents=current_plan.selected_agents,
                tasks=fallback_tasks,
                status=PlanStatus.MODIFIED,
                notes=current_plan.notes + f" | Fallback modification due to error: {e}",
                revision_count=current_plan.revision_count + 1
            )

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
        status_val = plan.status.value if hasattr(plan.status, "value") else str(plan.status)
        lines.append(f"### 📋 Execution Plan [{plan.plan_id}] (Status: {status_val.upper()})")
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
