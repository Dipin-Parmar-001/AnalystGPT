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


@dataclass
class TaskItem:
    task_id: Any = 1
    title: str = ""
    description: str = ""
    assigned_agent: str = "DataAnalystAgent"
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
        return cls(
            task_id=data.get("task_id", 1),
            title=data.get("title", ""),
            description=data.get("description") or data.get("objective", ""),
            assigned_agent=data.get("assigned_agent") or data.get("agent", "DataAnalystAgent"),
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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2
                )
                if cancellation_token:
                    cancellation_token.check_cancelled(step_name="PlannerAgent after LLM plan generation")

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

                if HAS_LOGFIRE:
                    logfire.info(
                        "PlannerAgent Created Plan",
                        plan_id=plan.plan_id,
                        selected_agents=selected_agents,
                        task_count=len(tasks),
                        task_summary=[{"id": t.task_id, "title": t.title, "agent": t.assigned_agent} for t in tasks],
                        notes=notes
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )
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
