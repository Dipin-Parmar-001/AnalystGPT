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
import time
from typing import List, Dict, Any, Optional, Union, Callable
from dotenv import load_dotenv
from openai import OpenAI


def is_fragmented_or_unparsable_response(message: Any) -> bool:
    """
    Detect if the LLL response is fragmented, unparsable, or contains only
    token usage metadata instead of actual content.

    Returns True if the response should be retried.
    """
    if message is None:
        return True

    # Check if content is None or empty
    content = getattr(message, "content", None)
    if content is None or (isinstance(content, str) and not content.strip()):
        # If no content and no tool calls, it's likely an incomplete response
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return True

    # Check if content looks like token usage metadata (JSON with token fields)
    if isinstance(content, str):
        content_stripped = content.strip()
        # Check for the specific token usage metadata pattern
        if content_stripped.startswith("{") and content_stripped.endswith("}"):
            try:
                parsed = json.loads(content_stripped)
                # Check for token usage metadata keys
                token_keys = {"completion_tokens", "prompt_tokens", "total_tokens",
                             "completion_tokens_details", "prompt_tokens_details",
                             "cost", "is_byok", "cost_details"}
                if isinstance(parsed, dict) and token_keys.intersection(parsed.keys()):
                    # This is token usage metadata, not actual content
                    return True
            except json.JSONDecodeError:
                pass

        # Check for other common fragmented response patterns
        # e.g., empty object, just whitespace, or clearly incomplete
        if content_stripped in ["{}", "[]", "null", ""]:
            return True

    # Check finish_reason if available (some providers include this in message)
    finish_reason = getattr(message, "finish_reason", None)
    if finish_reason in ["length", "content_filter", "function_call"]:
        # These might indicate incomplete responses
        pass

    return False

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sandbox_execution import ToolManager, execute_python, EXECUTE_PYTHON_TOOL_SPEC
from agents.tools import reset_global_sandbox_manager, extract_code_from_content, extract_action_code_from_content, create_synthetic_execute_tool_call
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.memory_manager import MessageHistoryManager, DatasetState, count_tokens, sanitize_messages, convert_to_plain_chat
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

from agents.prompts import DATA_ANALYST_AGENT, DATA_ANALYST_SYSTEM_PROMPT, resolve_budget_placeholders


class DataAnalystAgent:
    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
        tool_manager: Optional[ToolManager] = None,
        max_tool_calls: int = 7,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 300,
        max_tokens_before_compact: int = 100000,
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

        # Local observability: per-call latency, served model, token metrics
        from agents.observability import instrument_openai_client
        instrument_openai_client(self.client, agent_name="DataAnalystAgent")

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
        if getattr(self, "_plain_chat_mode", False):
            # Provider (e.g. Gemini) already rejected the OpenAI tool-message schema;
            # send plain user/assistant history while still exposing tools.
            sanitized_messages = convert_to_plain_chat(messages)
        else:
            sanitized_messages = sanitize_messages(messages)
        kwargs = {
            "model": self.model_name,
            "messages": sanitized_messages,
        }
        effective_tools = tools if tools is not None else self.tool_manager.get_tool_specs()
        if effective_tools:
            kwargs["tools"] = effective_tools

        def _do_call():
            delays = [10, 20, 30]
            for attempt, delay in enumerate(delays + [0]):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    message = response.choices[0].message

                    # Check for fragmented or unparsable response (token usage metadata, etc.)
                    if is_fragmented_or_unparsable_response(message):
                        if HAS_LOGFIRE:
                            logfire.warn("Detected fragmented/unparsable LLM response (token metadata only), retrying...",
                                         content_snippet=str(getattr(message, "content", None))[:200])
                        else:
                            print(f"[DataAnalystAgent] Detected fragmented/unparsable LLM response, retrying...")
                        # Treat as an error to trigger retry
                        raise RuntimeError("Fragmented or unparsable LLM response received (token metadata only)")

                    return message
                except Exception as err:
                    err_str = str(err).lower()
                    # Provider rejected the OpenAI tool-calling message schema (e.g. Gemini:
                    # "Role 'function' is not supported") - retry with plain user/assistant
                    # history. Tools are preserved so the agent loop keeps working.
                    is_schema_rejection = (
                        "role 'function'" in err_str
                        or 'role "function"' in err_str
                        or "role 'tool'" in err_str
                        or 'role "tool"' in err_str
                        or ("role" in err_str and "not supported" in err_str)
                        or ("tool" in err_str and "invalid" in err_str)
                    )
                    if is_schema_rejection:
                        if HAS_LOGFIRE:
                            logfire.warn("Portkey/Provider rejected tool message schema. Retrying with plain chat format (tools preserved).", error=str(err))
                        else:
                            print(f"[DataAnalystAgent] Provider tool schema rejected ({err}). Retrying with plain chat format (tools preserved)...")

                        plain_msgs = convert_to_plain_chat(messages)
                        fallback_kwargs = {
                            "model": self.model_name,
                            "messages": plain_msgs,
                        }
                        if effective_tools:
                            fallback_kwargs["tools"] = effective_tools
                        try:
                            fallback_resp = self.client.chat.completions.create(**fallback_kwargs)
                            self._plain_chat_mode = True
                            return fallback_resp.choices[0].message
                        except Exception as fallback_err:
                            if HAS_LOGFIRE:
                                logfire.error("DataAnalystAgent fallback LLM request failed", error=str(fallback_err))
                            raise RuntimeError(f"DataAnalystAgent LLM call failed after fallback: {fallback_err}") from fallback_err


                    if attempt < len(delays):
                        if HAS_LOGFIRE:
                            logfire.warn(f"LLM request failed (attempt {attempt + 1}), retrying in {delay}s...", error=str(err))
                        else:
                            print(f"[Agent Warning] LLM request failed (attempt {attempt + 1}), retrying in {delay}s... Error: {err}")
                        time.sleep(delay)
                        continue

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

        primary_task = assigned_tasks[0] if assigned_tasks else None
        task_type = getattr(primary_task, "task_type", None) or "cleaning"

        if task_type == "feature_engineering":
            task_specific_instructions = (
                f"Instructions for Feature Engineering:\n"
                f"1. You have a budget of MAX {effective_max_calls} tool calls.\n"
                f"2. Load `cleaned_dataset.csv` directly from FILE_REGISTRY.\n"
                f"3. Scope discipline: Construct ONLY the derived features named in the task description above.\n"
                f"4. Preserve all existing columns and identifiers (e.g. customerID, user_id).\n"
                f"5. Reload-and-assert: Reload `feature_engineered_dataset.csv` and verify shape and columns before finishing.\n"
                f"6. MANDATORY OUTPUTS: Save the engineered dataset as `feature_engineered_dataset.csv`, the engineering summary as `feature_engineering_summary.json`, AND your own task summary as `analyst_summary_task_{task_id_val}.json` plus `analyst_summary.json` — both must include the engineered feature definitions, rationale, and verified output shape.\n"
                f"7. Provide a concise summary of the engineered features and verified dataset shape."
            )
        elif task_type == "stats_and_viz":
            task_specific_instructions = (
                f"Instructions for Descriptive Statistics & Visualization:\n"
                f"1. You have a budget of MAX {effective_max_calls} tool calls.\n"
                f"2. Load the input dataset named in your task inputs (e.g. `feature_engineered_dataset.csv` or `cleaned_dataset.csv`).\n"
                f"3. Compute summary statistics, correlations, and grouped aggregations.\n"
                f"4. Generate ONLY the specific charts requested in the task description and save as PNG files.\n"
                f"5. COMPUTED STATS ARTIFACT (MANDATORY): in the same script, dump every segment rate, per-class "
                f"median/mean, and correlation you will report to `stats_findings.json` — each segment entry with "
                f"`feature`, `category`, `count`, `positives`, `rate`. Every number in `key_insights` and every chart "
                f"description MUST come from that file, never retyped. If you reconstruct categorical columns from "
                f"one-hot dummies, save the analysis frame as `analysis_dataset.csv` and claim ITS saved shape.\n"
                f"6. MANDATORY SUMMARY: Save your own task summary as `analyst_summary_task_{task_id_val}.json` AND `analyst_summary.json` with complete metric breakdowns.\n"
                f"7. Provide a comprehensive summary of key statistical insights and findings."
            )
        else:
            task_specific_instructions = (
                f"Instructions for Data Cleaning & Preparation:\n"
                f"1. You have a budget of MAX {effective_max_calls} tool calls.\n"
                f"2. Load datasets using the exact filename in FILE_REGISTRY.\n"
                f"3. RAW DATA INSPECTION & ROW RECONCILIATION (CRITICAL): Calculate duplicates (`df.duplicated().sum()`) and nulls (`df.isnull().sum()`) on the RAW dataset FIRST before any column exclusions. Track `initial_rows`, explicit `removed` breakdown with reason and count, and `retained` rows.\n"
                f"4. IDENTIFIER PRESERVATION: Do NOT drop identifier columns (e.g., customerID, id). Except/exclude them from feature correlations and plots, but KEEP them in the dataset.\n"
                f"5. EXPORT CLEANED DATASET: ALWAYS save the cleaned dataset to `cleaned_dataset.csv` (e.g., `df.to_csv('cleaned_dataset.csv', index=False)`).\n"
                f"6. MANDATORY SUMMARY: In addition to any intermediate summary (e.g. `cleaning_summary.json`), you MUST save your own task summary as `analyst_summary_task_{task_id_val}.json` AND `analyst_summary.json` with the complete metric breakdowns.\n"
                f"7. Provide a comprehensive final answer summarizing your cleaning, findings, and figures."
            )

        composite_prompt += (
            f"=== ASSIGNED TASKS FOR {self.agent_name} ===\n"
            f"{tasks_block}\n\n"
            f"{task_specific_instructions}"
        )

        memory_manager = MessageHistoryManager(
            system_prompt=resolve_budget_placeholders(DATA_ANALYST_SYSTEM_PROMPT, effective_max_calls),
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
        successful_executions = 0
        failed_executions = 0
        generated_files: List[str] = []
        final_response = ""
        agent_status = "in_progress"

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

            # Structural turn classification from the provider response:
            # - tool_calls present  -> pass to the execution layer below
            # - content only        -> non-execution response (markdown/JSON
            #   summary or prose). It is NEVER executed as code; the loop ends
            #   and task completion is decided by actual execution results.
            # Only a provider-issued 'Malformed function call:' body is parsed
            # for recoverable code.
            if not tool_calls:
                fallback_code = extract_action_code_from_content(response_msg.content or "")
                if fallback_code and tool_calls_count < effective_max_calls:
                    if HAS_LOGFIRE:
                        logfire.info("DataAnalystAgent recovered malformed function call via fallback parser", code_snippet=fallback_code[:200])
                    else:
                        print(f"[DataAnalystAgent Fallback] Recovered malformed function call ({len(fallback_code)} chars). Executing in sandbox...")
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
                    "tool_call_num": tool_calls_count + 1,
                    "tool_calls_used": tool_calls_count,
                    "tool_calls_remaining": max(0, effective_max_calls - tool_calls_count - 1),
                    "max_tool_calls": effective_max_calls
                })

                tool_result = self.tool_manager.execute_tool_call(tool_call, files=uploaded_files)

                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} after tool {fn_name}", task_id=task_id)

                for gf in tool_result.get("returned_files", []) or []:
                    if gf not in generated_files:
                        generated_files.append(gf)

                memory_manager.append_tool_result(tool_call_id, fn_name, tool_result)
                tool_calls_count += 1

                # Execution results are recorded from the tool layer, not inferred
                # from any LLM-generated text.
                if tool_result.get("success", True):
                    successful_executions += 1
                else:
                    failed_executions += 1

                if HAS_LOGFIRE:
                    logfire.info(
                        f"DataAnalystAgent Tool Executed: {fn_name}",
                        agent="DataAnalystAgent",
                        tool=fn_name,
                        success=tool_result.get("success", True),
                        execution_time=tool_result.get("execution_time", 0.0),
                        returned_files=tool_result.get("returned_files", []),
                        tool_call_num=tool_calls_count,
                        tool_calls_remaining=max(0, effective_max_calls - tool_calls_count),
                        max_tool_calls=effective_max_calls
                    )
                else:
                    print(
                        f"[DataAnalystAgent] Tool call {tool_calls_count}/{effective_max_calls} executed: {fn_name} "
                        f"(remaining tool calls: {max(0, effective_max_calls - tool_calls_count)})",
                        flush=True
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

        # Task completion must be grounded in actual execution: at least one tool
        # action passed to the execution layer and succeeded. The content of the
        # LLM response alone (a markdown summary, a status field, or code that
        # "would" create artifacts) never marks a task completed. A response with
        # zero successful executions is a non-execution response -> failed, so
        # the orchestrator's artifact validation triggers a retry.
        if agent_status != "tool_limit_reached":
            agent_status = "completed" if successful_executions > 0 else "no_execution"
        task_status_final = "completed" if successful_executions > 0 else "failed"
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

        # If not produced by script, generate structured fallback summary from artifacts
        if not summary_data:
            clean_shape = {}
            init_shape = {}
            num_cols = []
            cat_cols = []
            total_rows_val = 0
            
            # Inspect cleaned_dataset.csv or raw uploaded dataset
            target_ds = os.path.join(output_dir, "cleaned_dataset.csv")
            if not os.path.exists(target_ds) and uploaded_files and os.path.exists(uploaded_files[0]):
                target_ds = uploaded_files[0]
            
            if os.path.exists(target_ds):
                try:
                    import pandas as pd
                    df_inspect = pd.read_csv(target_ds)
                    total_rows_val = int(df_inspect.shape[0])
                    clean_shape = {"rows": total_rows_val, "columns": int(df_inspect.shape[1])}
                    init_shape = clean_shape
                    num_cols = df_inspect.select_dtypes(include=["number"]).columns.tolist()
                    cat_cols = df_inspect.select_dtypes(exclude=["number"]).columns.tolist()
                except Exception:
                    pass

            summary_data = {
                "agent": self.agent_name,
                "task_id": task_id_val,
                "task_title": assigned_tasks[0].title if assigned_tasks else "Data Analysis",
                "status": agent_status,
                # Honesty marker: this summary was assembled structurally from
                # on-disk artifacts, NOT written by the analyst agent. Downstream
                # consumers must not treat unspecified details as verified.
                "source": "fallback_structural_summary",
                "dataset_overview": {
                    "filename": os.path.basename(target_ds) if target_ds else "dataset.csv",
                    "initial_shape": init_shape,
                    "processed_shape": clean_shape,
                    "numerical_columns": num_cols,
                    "categorical_columns": cat_cols,
                    "identifier_columns": ["customerID"] if "customerID" in cat_cols else []
                },
                "data_cleaning": {
                    "row_reconciliation": {
                        "initial_rows": total_rows_val,
                        "removed": [],
                        "retained": total_rows_val
                    },
                    "missing_values_handled": {},
                    # NOT fabricated: leave imputation/encoding empty unless actually
                    # observed. An empty value means "not recorded by this fallback",
                    # never "median/mode imputation was applied".
                    "imputation_methods": {},
                    "identifier_columns_excepted": ["customerID (retained in dataset for prediction traceability)"] if "customerID" in cat_cols else [],
                    "dropped_columns": [],
                    "encoding": {},
                    "transformations": [
                        name for name in ["Exported cleaned_dataset.csv"]
                        if os.path.exists(os.path.join(output_dir, "cleaned_dataset.csv"))
                    ]
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
                "issues_and_caveats": [
                    "FALLBACK SUMMARY: generated structurally from artifacts because the agent did not produce its own summary JSON. Preprocessing details (imputation, encoding) are intentionally left empty rather than assumed.",
                    "Data analysis execution completed."
                ]
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

        # Enforce that the task generated ITS OWN designated summary (not a borrowed
        # intermediate from another task type). This check applies to summaries only.
        required_task_summary = {
            "cleaning": "cleaning_summary.json",
            "feature_engineering": "feature_engineering_summary.json",
            "stats_and_viz": summary_filename,
        }.get(task_type, summary_filename)
        missing_required_summary = None
        if not os.path.exists(os.path.join(output_dir, required_task_summary)):
            missing_required_summary = required_task_summary
            if HAS_LOGFIRE:
                logfire.warn(
                    "DataAnalystAgent task did not produce its own required summary",
                    agent="DataAnalystAgent",
                    task_type=task_type,
                    required_summary=required_task_summary,
                    generated_files=generated_files,
                )
            else:
                print(f"[DataAnalystAgent] WARNING: task_type '{task_type}' did not produce its required summary '{required_task_summary}'.", flush=True)

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
            "missing_required_summary": missing_required_summary,
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
