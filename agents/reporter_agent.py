"""
Reporter Agent Module - AnalystGPT

Specialized synthesis agent responsible for reading structured JSON summaries,
metrics, and artifact lists from preceding agents (Data Analyst, ML) and producing
an executive, clear, publication-grade markdown analysis report for stakeholders.
Does not execute code or require sandbox tools.
"""

import os
import sys
import json
import re
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

from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.planner_agent import ExecutionPlan, TaskItem, PlanStatus
from agents.prompts import REPORTER_AGENT, REPORTER_SYSTEM_PROMPT
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
    Reporter Agent that synthesizes multi-agent pipeline outputs into an
    executive, stakeholder-friendly markdown report without code execution.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
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
        self.agent_name = "ReporterAgent"

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

    def generate_report(
        self,
        user_goal: str,
        analyst_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        ml_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        artifacts: Optional[List[Union[str, Dict[str, str]]]] = None,
        task_results: Optional[List[Dict[str, Any]]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any]]] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        """
        Synthesizes structured task summaries into the final executive markdown report.
        
        Args:
            user_goal: The user's original request/goal.
            analyst_summaries: One or more analyst_summary.json objects from DataAnalyst tasks.
            ml_summaries: One or more ml_summary.json objects from ML tasks.
            artifacts: List of generated artifact paths/dictionaries.
            task_results: Optional fallback list of completed task result items.
            file_contents: Optional file contents (legacy/fallback).
            plan: Optional original execution plan.
            step_callback: Optional execution callback.
            cancellation_token: Optional token to halt execution.
            
        Returns:
            Executive analysis markdown report.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name="ReporterAgent initializing synthesis")

        if step_callback:
            step_callback("report_generation_started", {"agent": self.agent_name, "user_goal": user_goal})

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

        # 1. Data Analyst Summaries
        if analyst_summaries:
            analyst_str = json.dumps(analyst_summaries, indent=2, default=str)
            user_content_parts.append(f"=== DATA ANALYST SUMMARY / FINDINGS (`analyst_summary.json`) ===\n{analyst_str}\n")

        # 2. Machine Learning Summaries
        if ml_summaries:
            ml_str = json.dumps(ml_summaries, indent=2, default=str)
            user_content_parts.append(f"=== MACHINE LEARNING SUMMARY / MODEL EVALUATION (`ml_summary.json`) ===\n{ml_str}\n")

        # 3. Fallback: If neither analyst_summaries nor ml_summaries was provided, use task_results or file_contents
        if not analyst_summaries and not ml_summaries:
            if file_contents:
                file_blocks = []
                for fname, fcontent in file_contents.items():
                    # Truncate content to avoid token overflow
                    file_blocks.append(f"--- FILE: {fname} ---\n{fcontent[:2000]}\n")
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

        # 4. Artifacts List
        artifacts_str = json.dumps(artifacts or [], indent=2, default=str)
        user_content_parts.append(f"=== GENERATED ARTIFACTS LIST ===\n{artifacts_str}\n")

        user_content_parts.append(
            "Instructions:\n"
            "Synthesize the summaries above into the structured executive markdown report. "
            "STRICT GROUNDING: Directly quote exact metrics, baseline numbers, confusion matrix counts, sample counts, and feature importance rankings from the summaries above. "
            "Do NOT recalculate, assume, or alter any numbers. Ensure all report sections are detailed and executive-ready."
        )

        user_content = "\n\n".join(user_content_parts)

        messages = [
            {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        if cancellation_token:
            cancellation_token.check_cancelled(step_name="ReporterAgent before LLM synthesis call")

        def _do_call():
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2
                )
                return response.choices[0].message.content or ""
            except Exception as err:
                if HAS_LOGFIRE:
                    logfire.error("ReporterAgent LLM request failed", error=str(err))
                raise RuntimeError(f"ReporterAgent LLM call failed: {err}") from err

        if HAS_LOGFIRE:
            with logfire.span("ReporterAgent Generate Report", user_goal=user_goal[:200]):
                report_md = _do_call()
                logfire.info(
                    "ReporterAgent Synthesized Executive Report",
                    agent="ReporterAgent",
                    report_length=len(report_md),
                    artifact_count=len(artifacts or [])
                )
        else:
            report_md = _do_call()

        if cancellation_token:
            cancellation_token.check_cancelled(step_name="ReporterAgent after LLM synthesis call")

        if step_callback:
            step_callback("report_generated", {"agent": self.agent_name, "report_length": len(report_md)})

        return report_md

    def run_task(
        self,
        prompt: str,
        analyst_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        ml_summaries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        task_results: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[List[Union[str, Dict[str, str]]]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any]]] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
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
            step_callback=step_callback,
            cancellation_token=cancellation_token,
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
    print("ReporterAgent initialized successfully (Pure Synthesis, No Code Sandbox)!")
