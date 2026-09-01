"""
Reporter Agent Module - AnalystGPT

Specialized synthesis and verification agent responsible for reading structured
JSON summaries, metrics, and artifact lists from preceding agents (Data Analyst, ML),
verifying data consistency/avoiding contradictions via stateful Python execution tools,
and producing an executive, clear, publication-grade markdown analysis report for stakeholders.
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Union, Callable
from dotenv import load_dotenv
from openai import OpenAI
from agents.logger import setup_logfire, HAS_LOGFIRE

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sandbox_execution import ToolManager, execute_python, EXECUTE_PYTHON_TOOL_SPEC, build_verified_metrics
from agents.prompts import REPORTER_AGENT, REPORTER_SYSTEM_PROMPT
from agents.memory_manager import MessageHistoryManager, sanitize_messages, convert_to_plain_chat
from agents.planner_agent import ExecutionPlan
from agents.cancellation import CancellationToken, WorkflowCancelledException
from agents.numeric_verifier import verify_report_numbers, attach_verification_warning

try:
    import logfire
    HAS_LOGFIRE = True
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
                "max_tokens": 2500
            }
        },
        {
            "virtual_key": "analystgpt-openrouter",
            "override_params": {
                "model": "poolside/laguna-xs-2.1:free",
                "temperature": 0.2,
                "max_tokens": 2500
            }
        },
        {
            "virtual_key": "analystgpt-nvidia",
            "override_params": {
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "temperature": 0.2,
                "max_tokens": 2500
            }
        },
        {
            "virtual_key": "analystgpt-gemini",
            "override_params": {
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "max_tokens": 2500
            }
        }
    ]
}


class ReporterAgent:
    """
    Reporter Agent that verifies metrics in the stateful sandbox and synthesizes
    multi-agent pipeline outputs into an executive, stakeholder-friendly markdown report.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
        tool_manager: Optional[ToolManager] = None,
        max_tool_calls: int = 10,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 1000,
        max_tokens_before_compact: int = 4000,
    ):
        if not PORTKEY_API_KEY:
            raise ValueError("PORTKEY_API_KEY is not set in environment variables.")

        setup_logfire()

        headers = {
            "x-portkey-api-key": PORTKEY_API_KEY,
        }

        if PORTKEY_CONFIG_ID:
            if HAS_LOGFIRE:
                logfire.info("ReporterAgent setup using Portkey Config ID", config_id=PORTKEY_CONFIG_ID)
            else:
                print(f"[ReporterAgent Setup] Using Portkey Config ID: {PORTKEY_CONFIG_ID}")
            headers["x-portkey-config"] = PORTKEY_CONFIG_ID
        else:
            if HAS_LOGFIRE:
                logfire.warn("[ReporterAgent Setup] PORTKEY_CONFIG_ID not found. Using fallback config.")
            else:
                print("[ReporterAgent Setup] WARNING: PORTKEY_CONFIG_ID not found. Using fallback config.")
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
        self.agent_name = "ReporterAgent"
        self.keep_last_tool_pairs = keep_last_tool_pairs
        self.stdout_stderr_token_limit = stdout_stderr_token_limit
        self.max_tokens_before_compact = max_tokens_before_compact

    def _normalize_agent_name(self, name: str) -> str:
        if not name:
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def is_assigned_to_me(self, assigned_agent: str) -> bool:
        """Determines if a task's assigned agent corresponds to ReporterAgent."""
        normalized = self._normalize_agent_name(assigned_agent)
        target = self._normalize_agent_name(self.agent_name)
        return (
            normalized == target
            or normalized in ["reporter", "reporteragent", "reportingagent", "reporting"]
            or "report" in normalized
        )

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        sanitized_messages = sanitize_messages(messages)
        kwargs = {
            "model": self.model_name,
            "messages": sanitized_messages,
            "temperature": 0.2,
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
                    return message
                except Exception as err:
                    err_str = str(err).lower()
                    # If provider fails on tool calling schema (e.g. Google Role 'function' error or tool format rejection)
                    if "role 'function'" in err_str or "not supported" in err_str or "invalid_argument" in err_str or "400" in err_str:
                        if HAS_LOGFIRE:
                            logfire.warn("Portkey/Provider rejected tool schema in ReporterAgent. Falling back to plain chat format.", error=str(err))
                        else:
                            print(f"[ReporterAgent] Provider tool schema rejected ({err}). Falling back to robust plain chat format...")
                        
                        plain_msgs = convert_to_plain_chat(messages)
                        fallback_kwargs = {
                            "model": self.model_name,
                            "messages": plain_msgs,
                            "temperature": 0.2,
                        }
                        try:
                            fallback_resp = self.client.chat.completions.create(**fallback_kwargs)
                            return fallback_resp.choices[0].message
                        except Exception as fallback_err:
                            if HAS_LOGFIRE:
                                logfire.error("ReporterAgent fallback LLM request failed", error=str(fallback_err))
                            raise RuntimeError(f"ReporterAgent LLM call failed after fallback: {fallback_err}") from fallback_err

                    if attempt < len(delays):
                        import time
                        if HAS_LOGFIRE:
                            logfire.warn(f"LLM request failed (attempt {attempt + 1}), retrying in {delay}s...", error=str(err))
                        else:
                            print(f"[Agent Warning] LLM request failed (attempt {attempt + 1}), retrying in {delay}s... Error: {err}")
                        time.sleep(delay)
                        continue

                if HAS_LOGFIRE:
                    logfire.error("ReporterAgent LLM request failed", error=str(err))
                raise RuntimeError(f"ReporterAgent LLM call failed: {err}") from err

        if HAS_LOGFIRE:
            with logfire.span("ReporterAgent LLM Call", model=self.model_name):
                return _do_call()
        else:
            return _do_call()

    def generate_report(
        self,
        user_goal: str,
        analyst_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        ml_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        artifacts: Optional[List[Union[str, Dict[str, str]]]] = None,
        task_results: Optional[List[Dict[str, Any]]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any]]] = None,
        max_tool_calls: Optional[int] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        verified_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Synthesizes structured task summaries and sandbox artifacts into the final executive markdown report.
        Grounds every numeric claim in VERIFIED_METRICS and performs a deterministic post-generation verification loop.
        
        Args:
            user_goal: The user's original request/goal.
            analyst_summaries: One or more analyst_summary.json objects from DataAnalyst tasks.
            ml_summaries: One or more ml_summary.json objects from ML tasks.
            artifacts: List of generated artifact paths/dictionaries.
            task_results: Optional fallback list of completed task result items.
            file_contents: Optional file contents (legacy/fallback).
            plan: Optional original execution plan.
            max_tool_calls: Override for maximum tool calls (default: 10).
            step_callback: Optional execution callback.
            cancellation_token: Optional token to halt execution.
            verified_metrics: Optional pre-computed ground-truth verified metrics block.
            
        Returns:
            Executive analysis markdown report.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name="ReporterAgent initializing synthesis")

        effective_max_calls = max_tool_calls if max_tool_calls is not None else self.max_tool_calls

        def _safe_callback(event: str, data: Dict[str, Any]):
            if step_callback:
                try:
                    step_callback(event, data)
                except BaseException:
                    pass

        _safe_callback("report_generation_started", {
            "agent": self.agent_name,
            "user_goal": user_goal,
            "max_tool_calls": effective_max_calls
        })

        # Ensure ground-truth verified metrics exist
        raw_output_dir = getattr(self.tool_manager, "output_dir", "outputs")
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = raw_output_dir if (raw_output_dir and os.path.isabs(raw_output_dir)) else os.path.join(workspace_root, raw_output_dir or "outputs")

        if not verified_metrics:
            try:
                verified_metrics = build_verified_metrics(output_dir)
            except Exception as vm_err:
                print(f"[ReporterAgent Warning] build_verified_metrics notice: {vm_err}")
                verified_metrics = {}

        plan_info = ""
        if plan:
            if isinstance(plan, ExecutionPlan):
                plan_info = f"Plan ID: {plan.plan_id} | Notes: {plan.notes}"
            elif isinstance(plan, dict):
                plan_info = f"Plan ID: {plan.get('plan_id', 'N/A')} | Notes: {plan.get('notes', '')}"

        user_content_parts = [
            f"=== ORIGINAL USER QUERY / GOAL ===\n{user_goal}\n"
        ]

        if plan_info:
            user_content_parts.append(f"=== PLAN CONTEXT ===\n{plan_info}\n")

        # 1. Authoritative Ground Truth Verified Metrics
        if verified_metrics:
            verified_str = json.dumps(verified_metrics, indent=2, default=str)
            user_content_parts.append(f"=== VERIFIED_METRICS (GROUND TRUTH — COMPUTED DIRECTLY FROM ARTIFACTS) ===\n{verified_str}\n")

        # 2. Data Analyst Summaries (Context Only)
        if analyst_summaries:
            analyst_str = json.dumps(analyst_summaries, indent=2, default=str)
            user_content_parts.append(f"=== DATA ANALYST SUMMARY / FINDINGS (`analyst_summary.json` - CONTEXT ONLY) ===\n{analyst_str}\n")

        # 3. Machine Learning Summaries (Context Only)
        if ml_summaries:
            ml_str = json.dumps(ml_summaries, indent=2, default=str)
            user_content_parts.append(f"=== MACHINE LEARNING SUMMARY / MODEL EVALUATION (`ml_summary.json` - CONTEXT ONLY) ===\n{ml_str}\n")

        # 4. Fallback: If neither analyst_summaries nor ml_summaries was provided, use task_results or file_contents
        if not analyst_summaries and not ml_summaries:
            if file_contents:
                file_blocks = []
                for fname, fcontent in file_contents.items():
                    file_blocks.append(f"--- FILE: {fname} ---\n{fcontent[:4000]}\n")
                user_content_parts.append("=== GROUND-TRUTH METRIC FILES ===\n" + "\n".join(file_blocks))
            elif task_results:
                clean_results = []
                for tr in task_results:
                    clean_results.append({
                        "task_id": tr.get("task_id"),
                        "title": tr.get("title"),
                        "agent": tr.get("agent"),
                        "status": tr.get("status"),
                        "summary": tr.get("output", "")[:500] if isinstance(tr.get("output"), str) else tr.get("summary")
                    })
                user_content_parts.append(f"=== COMPLETED TASK SUMMARIES ===\n{json.dumps(clean_results, indent=2)}\n")

        # 5. Artifacts List
        artifacts_str = json.dumps(artifacts or [], indent=2, default=str)
        user_content_parts.append(f"=== GENERATED ARTIFACTS LIST IN STATEFUL ENVIRONMENT ===\n{artifacts_str}\n")

        user_content_parts.append(
            "Instructions:\n"
            "1. You have access to `execute_python` in the shared stateful Modal Cloud Sandbox containing ALL generated artifacts.\n"
            f"2. You have a budget of up to {effective_max_calls} tool calls. You are not forced to execute only 1 call, but you should combine related checks efficiently.\n"
            "3. STRICT GROUNDING: Every numeric value you write in the report MUST come from `VERIFIED_METRICS` verbatim. Do not compute, round, or estimate any number yourself. If a number the report needs isn't present in `VERIFIED_METRICS`, write \"not verified in current run\" instead of guessing.\n"
            "4. On contradiction: `VERIFIED_METRICS` is authoritative. Note any upstream discrepancies in Operational Caveats.\n"
            "5. Generate the complete executive markdown report following the required structure."
        )

        user_content = "\n\n".join(user_content_parts)

        memory_manager = MessageHistoryManager(
            system_prompt=REPORTER_SYSTEM_PROMPT,
            keep_last_tool_pairs=self.keep_last_tool_pairs,
            stdout_stderr_token_limit=self.stdout_stderr_token_limit,
            max_tokens_before_compact=self.max_tokens_before_compact
        )
        memory_manager.messages.append({"role": "user", "content": user_content})

        tool_calls_count = 0
        final_report = ""
        turn = 0

        while tool_calls_count < effective_max_calls:
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} turn {turn + 1}")

            turn += 1
            _safe_callback("turn_start", {"agent": self.agent_name, "turn": turn, "tool_calls_used": tool_calls_count})

            assistant_msg = self._call_llm(memory_manager.messages, tools=self.tool_manager.get_tool_specs())

            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} after LLM turn {turn}")

            thought = assistant_msg.content or ""
            tool_calls = assistant_msg.tool_calls or []

            if thought:
                _safe_callback("llm_thought", {"agent": self.agent_name, "turn": turn, "thought": thought})

            msg_dict: Dict[str, Any] = {"role": "assistant"}
            if assistant_msg.content:
                msg_dict["content"] = assistant_msg.content
            if tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", ""),
                            "arguments": tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "{}")
                        }
                    }
                    for i, tc in enumerate(tool_calls)
                ]

            memory_manager.append_assistant(msg_dict)

            # If model produced report without tool calls, we're done with tool execution
            if not tool_calls:
                final_report = thought
                break

            # Execute tool calls
            for tc in tool_calls:
                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} before tool call")

                fn_name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
                fn_args_raw = tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "{}")
                tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else f"call_{tool_calls_count + 1}")

                if tool_calls_count >= effective_max_calls:
                    memory_manager.append_tool_result(
                        tool_call_id=tc_id,
                        name=fn_name or "tool",
                        tool_result={"status": "tool_limit_reached", "message": f"Maximum tool call limit of {effective_max_calls} reached."}
                    )
                    continue

                tool_calls_count += 1

                _safe_callback("tool_executing", {
                    "agent": self.agent_name,
                    "turn": turn,
                    "tool_call_id": tc_id,
                    "fn_name": fn_name,
                    "raw_args": fn_args_raw,
                    "tool_call_num": tool_calls_count
                })

                exec_result = self.tool_manager.execute_tool_call(tool_call=tc)

                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} after tool {fn_name}")

                memory_manager.append_tool_result(
                    tool_call_id=tc_id,
                    name=fn_name,
                    tool_result=exec_result
                )

                if HAS_LOGFIRE:
                    logfire.info(
                        f"ReporterAgent Tool Executed: {fn_name}",
                        agent="ReporterAgent",
                        tool=fn_name,
                        success=exec_result.get("success", True),
                        execution_time=exec_result.get("execution_time", 0.0),
                        returned_files=exec_result.get("returned_files", [])
                    )

                _safe_callback("tool_completed", {
                    "agent": self.agent_name,
                    "turn": turn,
                    "tool_call_id": tc_id,
                    "fn_name": fn_name,
                    "result": exec_result
                })

        if not final_report and tool_calls_count >= effective_max_calls:
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} final report synthesis")
            memory_manager.messages.append({
                "role": "user",
                "content": f"[SYSTEM WARNING] You have reached the maximum limit of {effective_max_calls} tool calls. "
                           f"Do NOT make any more tool calls. Synthesize and output the final executive markdown report now based on VERIFIED_METRICS and all verified findings."
            })
            assistant_msg = self._call_llm(memory_manager.messages, tools=[])
            final_report = assistant_msg.content or ""
            memory_manager.append_assistant({"role": "assistant", "content": final_report})

        # ==============================================================================
        # POST-GENERATION DETERMINISTIC NUMERIC VERIFICATION & CORRECTION LOOP
        # ==============================================================================
        if final_report and verified_metrics:
            _safe_callback("verification_loop_started", {
                "agent": self.agent_name,
                "report_length": len(final_report)
            })

            max_correction_turns = 2
            correction_turn = 0
            verification_res = verify_report_numbers(final_report, verified_metrics)

            while not verification_res.is_valid and correction_turn < max_correction_turns:
                correction_turn += 1
                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} verification correction turn {correction_turn}")

                _safe_callback("verification_correction_turn", {
                    "agent": self.agent_name,
                    "correction_turn": correction_turn,
                    "unverified_count": verification_res.unverified_count,
                    "unverified_claims": [c.raw_str for c in verification_res.unverified_claims[:10]]
                })

                if HAS_LOGFIRE:
                    logfire.warn(
                        f"ReporterAgent Post-Verification Discrepancies (Turn {correction_turn})",
                        unverified_count=verification_res.unverified_count,
                        unverified_samples=[c.raw_str for c in verification_res.unverified_claims[:5]]
                    )

                # Send targeted deterministic correction prompt to LLM
                memory_manager.messages.append({
                    "role": "user",
                    "content": verification_res.correction_prompt
                })

                assistant_msg = self._call_llm(memory_manager.messages, tools=[])
                corrected_report = assistant_msg.content or ""
                if corrected_report:
                    final_report = corrected_report
                    memory_manager.append_assistant({"role": "assistant", "content": final_report})

                # Re-verify the revised report
                verification_res = verify_report_numbers(final_report, verified_metrics)

            # If still unverified after max correction turns, attach fallback warning banner
            if not verification_res.is_valid:
                if HAS_LOGFIRE:
                    logfire.warn(
                        "ReporterAgent Verification Max Turns Exceeded, attaching warning banner",
                        remaining_unverified=verification_res.unverified_count
                    )
                final_report = attach_verification_warning(final_report, verification_res.unverified_claims)

            _safe_callback("verification_completed", {
                "agent": self.agent_name,
                "is_valid": verification_res.is_valid,
                "total_claims": verification_res.total_claims_count,
                "verified_count": verification_res.verified_count,
                "unverified_count": verification_res.unverified_count,
                "correction_turns_used": correction_turn
            })

        if HAS_LOGFIRE:
            logfire.info(
                "ReporterAgent Synthesized Executive Report",
                agent="ReporterAgent",
                report_length=len(final_report),
                tool_calls_used=tool_calls_count,
                artifact_count=len(artifacts or [])
            )

        _safe_callback("report_generated", {
            "agent": self.agent_name,
            "report_length": len(final_report),
            "tool_calls_used": tool_calls_count
        })

        return final_report

    def run_task(
        self,
        prompt: str,
        analyst_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        ml_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        task_results: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[List[Union[str, Dict[str, str]]]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any]]] = None,
        max_tool_calls: Optional[int] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        verified_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Unified entry point to run the reporting task.
        """
        report_md = self.generate_report(
            user_goal=prompt,
            analyst_summaries=analyst_summaries,
            ml_summaries=ml_summaries,
            task_results=task_results,
            artifacts=artifacts,
            file_contents=file_contents,
            plan=plan,
            max_tool_calls=max_tool_calls,
            step_callback=step_callback,
            cancellation_token=cancellation_token,
            verified_metrics=verified_metrics,
        )

        return {
            "status": "completed",
            "agent": self.agent_name,
            "final_report": report_md,
            "summary": "Executive summary report generated successfully."
        }


if __name__ == "__main__":
    print("Testing ReporterAgent initialization...")
    reporter = ReporterAgent()
    print("ReporterAgent initialized successfully with stateful sandbox verification tool support!")
