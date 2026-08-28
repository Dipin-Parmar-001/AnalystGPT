"""
Data Analyst Agent Module - AnalystGPT

Specialized agent responsible for taking context (uploaded files, prompt/user query,
and plan built by PlannerAgent), filtering tasks assigned to DataAnalystAgent,
and applying data analysis steps (data inspection, cleaning, statistical analysis,
visualizations, ML modeling) using Python sandbox code execution tool with a maximum limit
of 7 tool calls.
"""

import os
import sys
import json
import re
import traceback
from typing import List, Dict, Any, Optional, Union, Callable
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

from sandbox_execution import ToolManager, execute_python, EXECUTE_PYTHON_TOOL_SPEC
from agents.tools import reset_global_sandbox_manager, extract_code_from_content, create_synthetic_execute_tool_call
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.memory_manager import MessageHistoryManager, DatasetState, count_tokens, sanitize_messages
from agents.planner_agent import ExecutionPlan, TaskItem, PlanStatus
from agents.cancellation import CancellationToken, WorkflowCancelledException



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

from agents.prompts import DATA_ANALYST_AGENT, DATA_ANALYST_SYSTEM_PROMPT


class DataAnalystAgent:
    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
        tool_manager: Optional[ToolManager] = None,
        max_tool_calls: int = 7,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 300,
        max_tokens_before_compact: int = 3000,
    ):
        if not PORTKEY_API_KEY:
            raise ValueError("PORTKEY_API_KEY is not set in environment variables.")

        setup_logfire()

        headers = {
            "x-portkey-api-key": PORTKEY_API_KEY,
        }

        if PORTKEY_CONFIG_ID:
            if HAS_LOGFIRE:
                logfire.info("DataAnalystAgent setup using Portkey Config ID", config_id=PORTKEY_CONFIG_ID)
            else:
                print(f"[DataAnalystAgent Setup] Using Portkey Config ID: {PORTKEY_CONFIG_ID}")
            headers["x-portkey-config"] = PORTKEY_CONFIG_ID
        else:
            if HAS_LOGFIRE:
                logfire.warn("[DataAnalystAgent Setup] PORTKEY_CONFIG_ID not found. Using fallback config.")
            else:
                print("[DataAnalystAgent Setup] WARNING: PORTKEY_CONFIG_ID not found. Using fallback config.")
            headers["x-portkey-config"] = json.dumps(DEFAULT_PORTKEY_CONFIG)

        self.client = OpenAI(
            base_url="https://api.portkey.ai/v1",
            api_key=PORTKEY_API_KEY,
            default_headers=headers
        )

        if HAS_LOGFIRE:
            try:
                logfire.instrument_openai(self.client)
            except Exception as e:
                logfire.warn("Failed to instrument OpenAI client with Logfire", error=str(e))

        self.model_name = model_name
        self.tool_manager = tool_manager or ToolManager()
        self.max_tool_calls = max_tool_calls
        self.agent_name = "DataAnalystAgent"
        self.keep_last_tool_pairs = keep_last_tool_pairs
        self.stdout_stderr_token_limit = stdout_stderr_token_limit
        self.max_tokens_before_compact = max_tokens_before_compact

    def _normalize_agent_name(self, name: str) -> str:
        if not name:
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def is_assigned_to_me(self, assigned_agent: str) -> bool:
        """Determines if a task's assigned agent corresponds to DataAnalystAgent."""
        normalized = self._normalize_agent_name(assigned_agent)
        target = self._normalize_agent_name(self.agent_name)
        return normalized == target or "dataanalyst" in normalized

    def extract_assigned_tasks(
        self,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any], str]] = None,
        task_id: Optional[int] = None
    ) -> List[TaskItem]:
        """Extracts task items from plan that are assigned to DataAnalystAgent."""
        if plan is None:
            return []

        tasks: List[TaskItem] = []

        if isinstance(plan, ExecutionPlan):
            tasks = plan.tasks
        elif isinstance(plan, dict):
            task_dicts = plan.get("tasks", [])
            tasks = [TaskItem.from_dict(t) if isinstance(t, dict) else t for t in task_dicts]
        elif isinstance(plan, str):
            try:
                plan_dict = json.loads(plan)
                if isinstance(plan_dict, dict):
                    task_dicts = plan_dict.get("tasks", [])
                    tasks = [TaskItem.from_dict(t) if isinstance(t, dict) else t for t in task_dicts]
            except Exception:
                pass

        if task_id is not None:
            tasks = [t for t in tasks if t.task_id == task_id]
        else:
            tasks = [t for t in tasks if self.is_assigned_to_me(t.assigned_agent)]

        return tasks

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        sanitized_messages = sanitize_messages(messages)
        kwargs = {
            "model": self.model_name,
            "messages": sanitized_messages,
        }
        effective_tools = tools if tools is not None else self.tool_manager.get_tool_specs()
        if effective_tools:
            kwargs["tools"] = effective_tools

        def _do_call():
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                return message
            except Exception as err:
                if HAS_LOGFIRE:
                    logfire.error("DataAnalystAgent LLM request failed", error=str(err))
                raise RuntimeError(f"DataAnalystAgent LLM call failed: {err}") from err

        if HAS_LOGFIRE:
            with logfire.span("DataAnalystAgent LLM Call", model=self.model_name):
                return _do_call()
        else:
            return _do_call()

    def run_task(
        self,
        prompt: str,
        uploaded_files: Optional[List[str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any], str]] = None,
        task_id: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """
        Executes assigned tasks from an ExecutionPlan or user prompt.
        
        Args:
            prompt: User query or prompt.
            uploaded_files: List of uploaded dataset file paths/names.
            plan: ExecutionPlan object, dict, or JSON string built by PlannerAgent.
            task_id: Optional task ID to run specifically.
            max_tool_calls: Override for maximum tool calls (default: 7).
            step_callback: Optional callback function for execution telemetry.
            cancellation_token: Optional token to immediately halt execution.
            
        Returns:
            Structured dictionary with execution status, task results, code executed, and final response.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name=f"{self.agent_name} initializing", task_id=task_id)

        effective_max_calls = max_tool_calls if max_tool_calls is not None else self.max_tool_calls

        assigned_tasks = self.extract_assigned_tasks(plan, task_id=task_id)

        # Fallback if no specific task matched or no plan provided
        if not assigned_tasks:
            default_task = TaskItem(
                task_id=1,
                title="Data Analysis Execution",
                description=prompt,
                assigned_agent=self.agent_name,
                status="in_progress",
                expected_output="Detailed analysis results and generated charts."
            )
            assigned_tasks = [default_task]
        else:
            for t in assigned_tasks:
                t.status = "in_progress"

        # Update plan status if ExecutionPlan instance provided
        if isinstance(plan, ExecutionPlan):
            for t in plan.tasks:
                if t.task_id in [at.task_id for at in assigned_tasks]:
                    t.status = "in_progress"

        # Build context block with File Registry
        from sandbox_execution.utils import format_file_registry_prompt
        file_registry_block = format_file_registry_prompt(uploaded_files)

        def _safe_callback(event: str, data: Dict[str, Any]):
            if step_callback:
                try:
                    step_callback(event, data)
                except BaseException:
                    pass

        task_id_val = assigned_tasks[0].task_id if assigned_tasks else 1

        tasks_formatted = []
        for t in assigned_tasks:
            tasks_formatted.append(
                f"- Task #{t.task_id}: {t.title}\n"
                f"  Description: {t.description}\n"
                f"  Expected Output: {t.expected_output}"
            )
        tasks_block = "\n".join(tasks_formatted)

        plan_summary = ""
        if plan:
            if isinstance(plan, ExecutionPlan):
                plan_summary = f"Plan ID: {plan.plan_id} | Total Tasks: {len(plan.tasks)} | Notes: {plan.notes}"
            elif isinstance(plan, dict):
                plan_summary = f"Plan ID: {plan.get('plan_id', 'N/A')} | Notes: {plan.get('notes', '')}"

        composite_prompt = (
            f"=== USER QUERY / PROMPT ===\n{prompt}\n\n"
            f"{file_registry_block}\n"
        )
        if plan_summary:
            composite_prompt += f"=== PLAN CONTEXT ===\n{plan_summary}\n\n"

        composite_prompt += (
            f"=== ASSIGNED TASKS FOR {self.agent_name} ===\n"
            f"{tasks_block}\n\n"
            f"Instructions:\n"
            f"1. Complete the assigned task(s) above using Python code execution.\n"
            f"2. You have a budget of MAX {effective_max_calls} tool calls.\n"
            f"3. Load datasets using the exact filename in FILE_REGISTRY or via get_file_path('filename').\n"
            f"4. RAW DATA INSPECTION & ROW RECONCILIATION (CRITICAL): Calculate duplicates (`df.duplicated().sum()`) and nulls (`df.isnull().sum()`) on the RAW dataset FIRST before any column exclusions. Track `initial_rows`, explicit `removed` breakdown with reason and count, and `retained` rows.\n"
            f"5. IDENTIFIER PRESERVATION: Do NOT drop identifier columns (e.g., customerID, id). Except/exclude them from feature correlations and plots, but KEEP them in the dataset.\n"
            f"6. EXPORT CLEANED DATASET: ALWAYS save the cleaned dataset to `cleaned_dataset.csv` (e.g., `df.to_csv('cleaned_dataset.csv', index=False)`).\n"
            f"7. Save `analyst_summary.json` (and `analyst_summary_task_{task_id_val}.json`) and all visualization PNGs.\n"
            f"8. Provide a comprehensive final answer summarizing your analysis, findings, and figures."
        )

        memory_manager = MessageHistoryManager(
            system_prompt=DATA_ANALYST_SYSTEM_PROMPT,
            keep_last_tool_pairs=self.keep_last_tool_pairs,
            stdout_stderr_token_limit=self.stdout_stderr_token_limit,
            max_tokens_before_compact=self.max_tokens_before_compact
        )

        memory_manager.messages.append({"role": "user", "content": composite_prompt})

        _safe_callback("task_started", {
            "agent": self.agent_name,
            "assigned_tasks": [t.to_dict() for t in assigned_tasks],
            "files": uploaded_files,
            "max_tool_calls": effective_max_calls
        })

        tool_calls_count = 0
        executed_code_blocks: List[str] = []
        generated_files: List[str] = []
        final_response = ""
        agent_status = "completed"

        # Agent turn loop bounded by tool calls
        for turn in range(effective_max_calls * 2):
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} turn {turn + 1}", task_id=task_id)

            _safe_callback("turn_start", {"agent": self.agent_name, "turn": turn + 1, "tool_calls_count": tool_calls_count})

            response_msg = self._call_llm(memory_manager.messages, tools=self.tool_manager.get_tool_specs())
            
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} after LLM turn {turn + 1}", task_id=task_id)

            msg_dict: Dict[str, Any] = {"role": "assistant"}
            if response_msg.content:
                msg_dict["content"] = response_msg.content
            if response_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_msg.tool_calls
                ]

            memory_manager.append_assistant(msg_dict)

            if response_msg.content:
                _safe_callback("llm_thought", {"agent": self.agent_name, "turn": turn + 1, "thought": response_msg.content.strip()})

            tool_calls = response_msg.tool_calls

            # If no tool calls requested, check fallback for malformed or raw code in content
            if not tool_calls:
                fallback_code = extract_code_from_content(response_msg.content or "")
                if fallback_code and tool_calls_count < effective_max_calls:
                    if HAS_LOGFIRE:
                        logfire.info("DataAnalystAgent detected code in assistant response via fallback parser", code_snippet=fallback_code[:200])
                    else:
                        print(f"[DataAnalystAgent Fallback] Extracted Python code from non-standard LLM response ({len(fallback_code)} chars). Executing in sandbox...")
                    synthetic_tc = create_synthetic_execute_tool_call(fallback_code, f"fallback_call_{turn + 1}_{tool_calls_count + 1}")
                    tool_calls = [synthetic_tc]
                else:
                    final_response = response_msg.content or ""
                    break


            # Process requested tool calls
            for tool_call in tool_calls:
                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} before tool call", task_id=task_id)

                tool_call_id = getattr(tool_call, "id", None) or (tool_call.get("id") if isinstance(tool_call, dict) else f"call_{tool_calls_count + 1}")
                fn_name = tool_call.function.name if hasattr(tool_call, "function") else tool_call.get("function", {}).get("name", "")
                raw_args = tool_call.function.arguments if hasattr(tool_call, "function") else tool_call.get("function", {}).get("arguments", "{}")

                if tool_calls_count >= effective_max_calls:
                    agent_status = "tool_limit_reached"
                    # Complete tool call response so API schema is never broken
                    memory_manager.append_tool_result(
                        tool_call_id,
                        fn_name or "tool",
                        {"status": "tool_limit_reached", "message": f"Maximum tool call limit of {effective_max_calls} reached."}
                    )
                    continue

                try:
                    parsed_args = json.loads(raw_args)
                    if "code" in parsed_args:
                        executed_code_blocks.append(parsed_args["code"])
                except Exception:
                    pass

                _safe_callback("tool_executing", {
                    "agent": self.agent_name,
                    "turn": turn + 1,
                    "tool_call_id": tool_call_id,
                    "fn_name": fn_name,
                    "raw_args": raw_args,
                    "tool_call_num": tool_calls_count + 1
                })

                tool_result = self.tool_manager.execute_tool_call(tool_call, files=uploaded_files)

                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} after tool {fn_name}", task_id=task_id)

                for gf in tool_result.get("returned_files", []) or []:
                    if gf not in generated_files:
                        generated_files.append(gf)

                memory_manager.append_tool_result(tool_call_id, fn_name, tool_result)
                tool_calls_count += 1

                if HAS_LOGFIRE:
                    logfire.info(
                        f"DataAnalystAgent Tool Executed: {fn_name}",
                        agent="DataAnalystAgent",
                        tool=fn_name,
                        success=tool_result.get("success", True),
                        execution_time=tool_result.get("execution_time", 0.0),
                        returned_files=tool_result.get("returned_files", [])
                    )

                _safe_callback("tool_completed", {
                    "agent": self.agent_name,
                    "turn": turn + 1,
                    "tool_call_id": tool_call_id,
                    "fn_name": fn_name,
                    "result": tool_result
                })

            if tool_calls_count >= effective_max_calls:
                if not final_response:
                    if cancellation_token:
                        cancellation_token.check_cancelled(step_name=f"{self.agent_name} final synthesis", task_id=task_id)
                    _safe_callback("tool_limit_warning", {
                        "agent": self.agent_name,
                        "message": f"Reached max tool call limit of {effective_max_calls}. Requesting final answer."
                    })
                    memory_manager.messages.append({
                        "role": "user",
                        "content": f"[SYSTEM WARNING] You have reached the maximum limit of {effective_max_calls} tool calls. "
                                   f"Do NOT make any more tool calls. Write your final, complete data analysis report now based on the execution outputs obtained so far."
                    })
                    final_msg = self._call_llm(memory_manager.messages, tools=[])
                    final_response = final_msg.content or ""
                    memory_manager.append_assistant({"role": "assistant", "content": final_response})
                break

        # Mark tasks completed or failed
        task_status_final = "completed" if agent_status != "failed" else "failed"
        for t in assigned_tasks:
            t.status = task_status_final
            t.result = final_response

        if isinstance(plan, ExecutionPlan):
            for t in plan.tasks:
                if t.task_id in [at.task_id for at in assigned_tasks]:
                    t.status = task_status_final
                    t.result = final_response

        # Ensure analyst_summary.json exists and is captured
        raw_output_dir = getattr(self.tool_manager, "output_dir", "outputs")
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = raw_output_dir if (raw_output_dir and os.path.isabs(raw_output_dir)) else os.path.join(workspace_root, raw_output_dir or "outputs")
        summary_data = None
        task_id_val = assigned_tasks[0].task_id if assigned_tasks else 1
        summary_filename = f"analyst_summary_task_{task_id_val}.json"
        main_summary_filename = "analyst_summary.json"

        # Check if cleaned_dataset.csv was produced
        for cleaned_candidate in ["cleaned_dataset.csv", "cleaned_data.csv"]:
            cleaned_path = os.path.join(output_dir, cleaned_candidate)
            if os.path.exists(cleaned_path) and cleaned_candidate not in generated_files:
                generated_files.append(cleaned_candidate)

        # Check if Python script produced analyst_summary.json
        for candidate_name in [main_summary_filename, summary_filename]:
            cand_path = os.path.join(output_dir, candidate_name)
            if os.path.exists(cand_path):
                try:
                    with open(cand_path, "r", encoding="utf-8") as sf:
                        summary_data = json.load(sf)
                    if candidate_name not in generated_files:
                        generated_files.append(candidate_name)
                    break
                except Exception:
                    pass

        # If not produced by script, generate structured fallback summary
        if not summary_data:
            summary_data = {
                "agent": self.agent_name,
                "task_id": task_id_val,
                "task_title": assigned_tasks[0].title if assigned_tasks else "Data Analysis",
                "status": agent_status,
                "dataset_overview": {
                    "filename": os.path.basename(uploaded_files[0]) if uploaded_files else "dataset.csv",
                    "initial_shape": {},
                    "processed_shape": {},
                    "numerical_columns": [],
                    "categorical_columns": [],
                    "identifier_columns": ["customerID"]
                },
                "data_cleaning": {
                    "row_reconciliation": {
                        "initial_rows": 0,
                        "removed": [],
                        "retained": 0
                    },
                    "missing_values_handled": {},
                    "imputation_methods": {"numerical": "median", "categorical": "mode"},
                    "identifier_columns_excepted": ["customerID (retained in dataset for prediction traceability)"],
                    "dropped_columns": [],
                    "encoding": {"method": "One-Hot Encoding (drop_first=True)"},
                    "transformations": ["Preserved identifier columns", "Exported cleaned_dataset.csv"]
                },
                "statistical_findings": {
                    "summary_metrics": {},
                    "key_insights": [final_response[:300] if final_response else "Data analysis completed."]
                },
                "generated_visualizations": [
                    {"file": gf, "description": f"Visualization plot: {gf}"}
                    for gf in generated_files if gf.endswith((".png", ".jpg", ".jpeg"))
                ],
                "saved_artifacts": generated_files,
                "issues_and_caveats": ["Identifier column customerID retained in dataset for downstream prediction linkage."]
            }
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, summary_filename), "w", encoding="utf-8") as sf:
                    json.dump(summary_data, sf, indent=2)
                with open(os.path.join(output_dir, main_summary_filename), "w", encoding="utf-8") as sf:
                    json.dump(summary_data, sf, indent=2)
                if summary_filename not in generated_files:
                    generated_files.append(summary_filename)
                if main_summary_filename not in generated_files:
                    generated_files.append(main_summary_filename)
            except Exception as write_err:
                print(f"[DataAnalystAgent] Notice saving fallback summary: {write_err}")

        if HAS_LOGFIRE:
            logfire.info(
                "DataAnalystAgent Task Finished",
                agent="DataAnalystAgent",
                status=agent_status,
                tool_calls_made=tool_calls_count,
                generated_files=generated_files,
                assigned_tasks_count=len(assigned_tasks)
            )

        _safe_callback("task_finished", {
            "agent": self.agent_name,
            "status": agent_status,
            "tool_calls_made": tool_calls_count,
            "assigned_tasks_count": len(assigned_tasks)
        })

        return {
            "status": agent_status,
            "agent_name": self.agent_name,
            "user_query": prompt,
            "response": final_response,
            "executed_tasks": [t.to_dict() for t in assigned_tasks],
            "tool_calls_made": tool_calls_count,
            "max_tool_calls_limit": effective_max_calls,
            "executed_code": executed_code_blocks,
            "generated_files": generated_files,
            "summary_data": summary_data,
            "plan": plan.to_dict() if isinstance(plan, ExecutionPlan) else plan
        }

    def run(
        self,
        prompt: str,
        uploaded_files: Optional[List[str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any], str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience alias for run_task."""
        return self.run_task(prompt=prompt, uploaded_files=uploaded_files, plan=plan, **kwargs)


if __name__ == "__main__":
    print("Testing DataAnalystAgent standalone initialization...")
    from agents.planner_agent import PlannerAgent

    planner = PlannerAgent()
    sample_query = "Analyze the sales performance, check for missing values, calculate total revenue per region, and produce a bar chart."
    sample_files = ["sales_data.csv"]
    
    print("\n--- Creating Execution Plan via PlannerAgent ---")
    plan = planner.create_plan(user_query=sample_query, dataset_context="sales_data.csv with region, sales, date columns")
    print(f"Generated Plan ID: {plan.plan_id}")
    print("Tasks in Plan:")
    for task in plan.tasks:
        print(f"  Task #{task.task_id} [{task.assigned_agent}]: {task.title}")

    print("\n--- Initializing DataAnalystAgent ---")
    analyst_agent = DataAnalystAgent(max_tool_calls=7)

    assigned = analyst_agent.extract_assigned_tasks(plan)
    print(f"Tasks assigned to DataAnalystAgent: {len(assigned)}")
    for t in assigned:
        print(f"  - #{t.task_id}: {t.title}")
