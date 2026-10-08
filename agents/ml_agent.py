"""
Machine Learning Agent Module - AnalystGPT

Specialized agent responsible for training, evaluating, comparing machine learning
models, hyperparameter tuning, metrics logging (model_comparison.csv), and
generating model artifacts/predictions in the Modal cloud sandbox.
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
    Detect if the LLM response is fragmented, unparsable, or contains only
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
from agents.tools import (
    reset_global_sandbox_manager,
    extract_code_from_content,
    extract_action_code_from_content,
    create_synthetic_execute_tool_call,
    parse_and_clean_tool_args,
    validate_or_retry_tool_calls,
)
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.memory_manager import MessageHistoryManager, DatasetState, count_tokens, sanitize_messages, convert_to_plain_chat
from agents.planner_agent import ExecutionPlan, TaskItem, PlanStatus
from agents.prompts import ML_AGENT, ML_SYSTEM_PROMPT, resolve_budget_placeholders
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


class MLAgent:
    """
    Machine Learning Agent that executes model training, evaluation, comparison,
    and inference tasks inside the secure Modal Cloud Sandbox.
    """

    def __init__(
        self,
        model_name: str = "nemotron-3-ultra-550b-a55b",
        tool_manager: Optional[ToolManager] = None,
        max_tool_calls: int = 7,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 1200,
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
                logfire.info("MLAgent setup using Portkey Config ID", config_id=PORTKEY_CONFIG_ID)
            else:
                print(f"[MLAgent Setup] Using Portkey Config ID: {PORTKEY_CONFIG_ID}")
            headers["x-portkey-config"] = PORTKEY_CONFIG_ID
        else:
            if HAS_LOGFIRE:
                logfire.warn("[MLAgent Setup] PORTKEY_CONFIG_ID not found. Using fallback config.")
            else:
                print("[MLAgent Setup] WARNING: PORTKEY_CONFIG_ID not found. Using fallback config.")
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
        instrument_openai_client(self.client, agent_name="MLAgent")

        self.model_name = model_name[len("nvidia/"):] if model_name.startswith("nvidia/") else model_name
        self.tool_manager = tool_manager or ToolManager()
        self.max_tool_calls = max_tool_calls
        self.agent_name = "MLAgent"
        self.keep_last_tool_pairs = keep_last_tool_pairs
        self.stdout_stderr_token_limit = stdout_stderr_token_limit
        self.max_tokens_before_compact = max_tokens_before_compact

    def _normalize_agent_name(self, name: str) -> str:
        if not name:
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def is_assigned_to_me(self, assigned_agent: str) -> bool:
        """Determines if a task's assigned agent corresponds to MLAgent."""
        normalized = self._normalize_agent_name(assigned_agent)
        target = self._normalize_agent_name(self.agent_name)
        return (
            normalized == target
            or normalized in ["ml", "mlagent", "machinelearning", "machinelearningagent"]
            or "ml" in normalized
        )

    def extract_assigned_tasks(
        self,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any], str]] = None,
        task_id: Optional[Union[int, str]] = None
    ) -> List[TaskItem]:
        """Extracts task items from plan that are assigned to MLAgent."""
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
            tasks = [t for t in tasks if str(t.task_id) == str(task_id)]
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
                            print(f"[MLAgent] Detected fragmented/unparsable LLM response, retrying...")
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
                            logfire.warn("Portkey/Provider rejected tool message schema in MLAgent. Retrying with plain chat format (tools preserved).", error=str(err))
                        else:
                            print(f"[MLAgent] Provider tool schema rejected ({err}). Retrying with plain chat format (tools preserved)...")

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
                                logfire.error("MLAgent fallback LLM request failed", error=str(fallback_err))
                            raise RuntimeError(f"MLAgent LLM call failed after fallback: {fallback_err}") from fallback_err


                    if attempt < len(delays):
                        if HAS_LOGFIRE:
                            logfire.warn(f"LLM request failed (attempt {attempt + 1}), retrying in {delay}s...", error=str(err))
                        else:
                            print(f"[Agent Warning] LLM request failed (attempt {attempt + 1}), retrying in {delay}s... Error: {err}")
                        time.sleep(delay)
                        continue

                if HAS_LOGFIRE:
                    logfire.error("MLAgent LLM request failed", error=str(err))
                raise RuntimeError(f"MLAgent LLM call failed: {err}") from err

        if HAS_LOGFIRE:
            with logfire.span("MLAgent LLM Call", model=self.model_name):
                return _do_call()
        else:
            return _do_call()

    def run_task(
        self,
        prompt: str,
        uploaded_files: Optional[List[str]] = None,
        plan: Optional[Union[ExecutionPlan, Dict[str, Any], str]] = None,
        task_id: Optional[Union[int, str]] = None,
        max_tool_calls: Optional[int] = None,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """
        Executes machine learning tasks from an ExecutionPlan or prompt.
        """
        if cancellation_token:
            cancellation_token.check_cancelled(step_name=f"{self.agent_name} initializing", task_id=task_id if isinstance(task_id, int) else None)

        effective_max_calls = max_tool_calls if max_tool_calls is not None else self.max_tool_calls

        assigned_tasks = self.extract_assigned_tasks(plan, task_id=task_id)

        if not assigned_tasks:
            default_task = TaskItem(
                task_id=2,
                title="Machine Learning Modeling & Comparison",
                description=prompt,
                assigned_agent=self.agent_name,
                status="in_progress",
                expected_output="model_comparison.csv, predictions.csv, feature_importance.png"
            )
            assigned_tasks = [default_task]
        else:
            for t in assigned_tasks:
                t.status = "in_progress"

        if isinstance(plan, ExecutionPlan):
            for t in plan.tasks:
                if t.task_id in [at.task_id for at in assigned_tasks]:
                    t.status = "in_progress"

        files_str = ", ".join(uploaded_files) if uploaded_files else "None provided"

        # Build context block with File Registry
        from sandbox_execution.utils import format_file_registry_prompt
        file_registry_block = format_file_registry_prompt(uploaded_files)

        def _safe_callback(event: str, data: Dict[str, Any]):
            if step_callback:
                try:
                    step_callback(event, data)
                except BaseException:
                    pass

        tasks_formatted = []
        for t in assigned_tasks:
            tasks_formatted.append(
                f"- Task #{t.task_id}: {t.title}\n"
                f"  Objective: {t.description}\n"
                f"  Inputs: {getattr(t, 'inputs', [])}\n"
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
            f"=== USER QUERY / GOAL ===\n{prompt}\n\n"
            f"{file_registry_block}\n"
        )
        if plan_summary:
            composite_prompt += f"=== PLAN CONTEXT ===\n{plan_summary}\n\n"

        primary_task = assigned_tasks[0] if assigned_tasks else None
        task_type = getattr(primary_task, "task_type", None) or "model_comparison"

        if task_type == "unsupervised_analysis":
            task_specific_instructions = (
                f"Instructions for Unsupervised Analysis:\n"
                f"1. Use the task objective to choose clustering, anomaly detection, or dimensionality reduction. Do not invent a target or labels. If the objective is underspecified, choose a defensible method and record the assumption.\n"
                f"2. Load the requested dataset and exclude identifiers, free-text IDs, and non-feature metadata from the feature matrix. Impute missing values and scale numeric features when the chosen algorithm is distance- or scale-sensitive; record every transformation.\n"
                f"3. For clustering, compare a small defensible set of algorithms or cluster counts. Report Silhouette (higher is better), Davies-Bouldin (lower), and Calinski-Harabasz (higher) when mathematically defined. Also report cluster sizes, noise/uncategorized share, and stability across seeds or resamples. Do not compare raw inertia across different feature spaces.\n"
                f"4. For anomaly detection, if ground-truth labels exist and were not used for fitting, report PR-AUC and ROC-AUC plus precision@k at a stated review capacity. Without labels, report the score distribution, chosen threshold and its rationale, flagged count/rate, and a small reviewable list of flagged record IDs; do not claim accuracy.\n"
                f"5. For dimensionality reduction, report explained variance for PCA where applicable, and trustworthiness or reconstruction error when appropriate. State what information the projection may lose.\n"
                f"6. If independent reference labels exist, ARI/NMI may be reported as external validation only; never use them to fit or choose the representation without clearly labeling that evaluation. Metrics are not interchangeable: explain which are applicable and why any are omitted.\n"
                f"7. Save `unsupervised_metrics.json` as JSON with `task_type`, `features_used`, `preprocessing`, `candidate_results` (list of objects with `algorithm`, `configuration`, `metrics`, `optimization_directions`, and `limitations`), and `selected_result` (object with `algorithm`, `configuration`, `metrics`, `reason`, and `limitations`). Use consistent snake_case metric keys. Save `unsupervised_summary.json` with plain-language findings. Save `ml_summary.json` and `ml_summary_task_{assigned_tasks[0].task_id if assigned_tasks else 2}.json`. Save `cluster_assignments.csv` for clustering with preserved identifier columns and the assigned label in a `cluster` column; for anomaly detection save scores/flags in `anomaly_scores.csv` when applicable. Save reusable fitted model/preprocessing artifacts where supported. Do not create supervised-only champion, confusion-matrix, baseline, or feature-importance artifacts.\n"
                f"8. You have a budget of MAX {effective_max_calls} tool calls. Ground every metric in executed code and clearly label unavailable metrics instead of fabricating values."
            )
        elif task_type == "hyperparameter_tuning":
            # Get primary_metric from task spec
            primary_metric = getattr(primary_task, "primary_metric", None) if primary_task else None
            metric_hint = f" (Planner specified: {primary_metric})" if primary_metric else ""

            task_specific_instructions = (
                f"Instructions for Hyperparameter Tuning:\n"
                f"1. You have a budget of MAX {effective_max_calls} tool calls.\n"
                f"2. Load `model_comparison.csv` and the champion model artifact (from `champion_model.json[\"artifact_filename\"]`) directly from FILE_REGISTRY.\n"
                f"3. Reconstruct training data using identical split and preprocessing from baseline.\n"
                f"4. Bounded search: Run GridSearchCV or RandomizedSearchCV with cv=5 over the bounded parameter search space specified in your task description. Use scoring='{primary_metric or 'roc_auc'}'{metric_hint} for the search -- this MUST match the primary_metric from the task spec / champion_model.json.\n"
                f"5. Honest reporting: Only update `champion_model.json` to point to the tuned model if the tuned model's primary metric strictly outperforms baseline.\n"
                f"6. MANDATORY OUTPUT ARTIFACTS:\n"
                f"   - `tuned_model_comparison.csv` (Comparing baseline vs tuned metrics side by side)\n"
                f"   - `best_params.json` (Winning hyperparameter combination and best CV score)\n"
                f"   - `hyperparameter_tuning_summary.json` (Structured tuning summary JSON)\n"
                f"   - `ml_summary_task_{assigned_tasks[0].task_id if assigned_tasks else 2}.json` AND `ml_summary.json` (Your own task summary with tuned-vs-baseline metric deltas — both files required)\n"
                f"   - Versioned tuned model artifact (e.g., `logistic_regression_p1_tuned.pkl`)\n"
                f"7. Provide a concise summary of tuning results, parameter improvements, and test metric deltas."
            )
        else:
            # Get primary_metric from task spec
            primary_metric = getattr(primary_task, "primary_metric", None) if primary_task else None
            metric_hint = f" (Planner specified: {primary_metric})" if primary_metric else ""

            task_specific_instructions = (
                f"Instructions for Model Training & Comparison:\n"
                f"1. Dataset Loading: If `feature_engineered_dataset.csv` or `cleaned_dataset.csv` is available in FILE_REGISTRY or inputs, load it directly. If not available, use the main dataset provided.\n"
                f"2. Identifier Preservation: Retain identifier columns (e.g., `customerID`, `id`) in `predictions.csv` for entity traceability, but EXCLUDE them from training feature matrix `X`.\n"
                f"3. Preprocessing: Use One-Hot Encoding (`pd.get_dummies(..., drop_first=True)` or `OneHotEncoder`) for nominal features, `LabelEncoder()` strictly for classification target `y`, `StandardScaler()` fit strictly on `X_train`, and `train_test_split(..., test_size=0.2, random_state=42, stratify=y if classification else None)`.\n"
                f"4. Metrics: For classification, evaluate Accuracy, Precision, Recall, F1, ROC-AUC, **PR-AUC** (`average_precision_score`), and **5-Fold Stratified Cross-Validation** (`CV_Mean`, `CV_Std`). For regression, evaluate RMSE, MAE, R2, MSE, and 5-Fold CV R2.\n"
                f"5. Directional Feature Importance: Extract signed `coefficient`, `absolute_importance`, and `direction` (e.g., `'reduces_churn'` / `'increases_churn'` or `'reduces_target'` / `'increases_target'`) and save to `feature_importance.csv`.\n"
                f"6. Business-Aware Model Selection: The primary metric for selection is {primary_metric or 'roc_auc'}{metric_hint}. Provide a comparative multi-metric rationale and select the model that scores highest on this primary metric (or lowest for error metrics like RMSE).\n"
                f"7. MANDATORY ARTIFACTS: Your script MUST save all core files:\n"
                f"   - `model_comparison.csv` (Comparing models across Accuracy, Precision, Recall, F1, ROC_AUC, PR_AUC, CV_Mean, CV_Std, **model_artifact**)\n"
                f"   - `baseline_score.json` (Dummy baseline benchmark using `DummyClassifier(strategy=\"most_frequent\")` labeled \"Majority Class Baseline\")\n"
                f"   - `feature_importance.csv` (Columns: feature, coefficient, absolute_importance, direction)\n"
                f"   - `confusion_matrix.json` (Confusion matrix data for best model)\n"
                f"   - `predictions.csv` (Predictions with customerID, actual, predicted, probability)\n"
                f"   - Versioned model artifacts (e.g., `logistic_regression_p0.pkl`, `random_forest_p0.pkl` — NOT `best_model.pkl`)\n"
                f"   - `ml_summary.json` AND `ml_summary_task_{assigned_tasks[0].task_id if assigned_tasks else 2}.json` (Structured machine learning summary — both files required)\n"
                f"   - `model_registry.json` and `champion_model.json`\n"
                f"8. You have a budget of MAX {effective_max_calls} tool calls.\n"
                f"9. Output the final structured JSON summary and metric details."
            )

        composite_prompt += (
            f"=== ASSIGNED ML TASKS FOR {self.agent_name} ===\n"
            f"{tasks_block}\n\n"
            f"{task_specific_instructions}"
        )

        memory_manager = MessageHistoryManager(
            system_prompt=resolve_budget_placeholders(ML_SYSTEM_PROMPT, effective_max_calls),
            keep_last_tool_pairs=self.keep_last_tool_pairs,
            stdout_stderr_token_limit=self.stdout_stderr_token_limit,
            max_tokens_before_compact=self.max_tokens_before_compact,
        )

        memory_manager.prepare_user_prompt(composite_prompt, files=uploaded_files)

        tool_calls_count = 0
        executed_codes: List[str] = []
        successful_executions = 0
        failed_executions = 0
        generated_files: List[str] = []
        turn = 0
        final_text = ""

        _safe_callback("task_started", {
            "agent": self.agent_name,
            "task_count": len(assigned_tasks),
            "tasks": [t.to_dict() for t in assigned_tasks]
        })

        while tool_calls_count < effective_max_calls:
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} turn {turn + 1}", task_id=task_id if isinstance(task_id, int) else None)

            turn += 1
            _safe_callback("turn_start", {"agent": self.agent_name, "turn": turn, "tool_calls_used": tool_calls_count})

            assistant_msg = self._call_llm(memory_manager.messages, tools=self.tool_manager.get_tool_specs())
            
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} after LLM turn {turn}", task_id=task_id if isinstance(task_id, int) else None)

            # Validate tool call arguments; if parsing fails, retry with LLM directly without corrupting history
            assistant_msg, tool_calls = validate_or_retry_tool_calls(
                agent_name=self.agent_name,
                assistant_msg=assistant_msg,
                memory_manager=memory_manager,
                call_llm_fn=self._call_llm,
                tool_specs=self.tool_manager.get_tool_specs(),
                cancellation_token=cancellation_token,
                step_callback=_safe_callback,
            )

            thought = assistant_msg.content or ""
            if thought:
                _safe_callback("llm_thought", {"agent": self.agent_name, "turn": turn, "thought": thought})

            if not tool_calls:
                # Structural turn classification: only a provider-issued
                # 'Malformed function call:' body is recovered as code. A plain
                # content response (markdown summary, JSON, prose) is a
                # non-execution turn and is never executed as Python.
                fallback_code = extract_action_code_from_content(thought)
                if fallback_code and tool_calls_count < effective_max_calls:
                    if HAS_LOGFIRE:
                        logfire.info("MLAgent recovered malformed function call via fallback parser", code_snippet=fallback_code[:200])
                    else:
                        print(f"[MLAgent Fallback] Recovered malformed function call ({len(fallback_code)} chars). Executing in sandbox...")
                    synthetic_tc = create_synthetic_execute_tool_call(fallback_code, f"fallback_call_{turn}_{tool_calls_count + 1}")
                    tool_calls = [synthetic_tc]
                else:
                    final_text = thought
                    memory_manager.append_assistant_response(assistant_msg)
                    break

            memory_manager.append_assistant_response(assistant_msg)


            for tc in tool_calls:
                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} before tool call", task_id=task_id if isinstance(task_id, int) else None)

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
                    "tool_call_num": tool_calls_count,
                    "tool_calls_used": tool_calls_count,
                    "tool_calls_remaining": max(0, effective_max_calls - tool_calls_count),
                    "max_tool_calls": effective_max_calls
                })

                exec_result = self.tool_manager.execute_tool_call(
                    tool_call=tc,
                    files=uploaded_files
                )

                if cancellation_token:
                    cancellation_token.check_cancelled(step_name=f"{self.agent_name} after tool {fn_name}", task_id=task_id if isinstance(task_id, int) else None)

                for gf in exec_result.get("returned_files", []) or []:
                    if gf not in generated_files:
                        generated_files.append(gf)

                ok_args, args_dict, _ = parse_and_clean_tool_args(fn_args_raw)
                if ok_args and "code" in args_dict:
                    executed_codes.append(args_dict["code"])

                memory_manager.append_tool_result(
                    tool_call_id=tc_id,
                    name=fn_name,
                    tool_result=exec_result
                )

                # Execution results are recorded from the tool layer, not inferred
                # from any LLM-generated text.
                if exec_result.get("success", True):
                    successful_executions += 1
                else:
                    failed_executions += 1

                if HAS_LOGFIRE:
                    logfire.info(
                        f"MLAgent Tool Executed: {fn_name}",
                        agent="MLAgent",
                        tool=fn_name,
                        success=exec_result.get("success", True),
                        execution_time=exec_result.get("execution_time", 0.0),
                        returned_files=exec_result.get("returned_files", []),
                        tool_call_num=tool_calls_count,
                        tool_calls_remaining=max(0, effective_max_calls - tool_calls_count),
                        max_tool_calls=effective_max_calls
                    )
                else:
                    print(
                        f"[MLAgent] Tool call {tool_calls_count}/{effective_max_calls} executed: {fn_name} "
                        f"(remaining tool calls: {max(0, effective_max_calls - tool_calls_count)})",
                        flush=True
                    )

                _safe_callback("tool_completed", {
                    "agent": self.agent_name,
                    "turn": turn,
                    "tool_call_id": tc_id,
                    "fn_name": fn_name,
                    "result": exec_result
                })

        if not final_text and tool_calls_count >= effective_max_calls:
            if cancellation_token:
                cancellation_token.check_cancelled(step_name=f"{self.agent_name} final warning response", task_id=task_id if isinstance(task_id, int) else None)
            memory_manager.messages.append({
                "role": "user",
                "content": f"[SYSTEM WARNING] You have reached the maximum limit of {effective_max_calls} tool calls. "
                           f"Do NOT make any more tool calls. Write your final machine learning summary, best model details, and metrics comparison now."
            })
            assistant_msg = self._call_llm(memory_manager.messages, tools=[])
            final_text = assistant_msg.content or ""
            memory_manager.append_assistant_response(assistant_msg)

        # Task completion must be grounded in actual execution: at least one tool
        # action passed to the execution layer and succeeded. The content of the
        # LLM response alone (a markdown summary, a status field, or code that
        # "would" create artifacts) never marks a task completed. A response with
        # zero successful executions is a non-execution response -> failed, so
        # the orchestrator's artifact validation triggers a retry.
        final_status = "completed" if successful_executions > 0 else "failed"
        for t in assigned_tasks:
            t.status = final_status
            t.result = final_text

        if isinstance(plan, ExecutionPlan):
            for t in plan.tasks:
                if t.task_id in [at.task_id for at in assigned_tasks]:
                    t.status = final_status
                    t.result = final_text

        # Ensure ml_summary.json exists and is captured
        raw_output_dir = getattr(self.tool_manager, "output_dir", "outputs")
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = raw_output_dir if (raw_output_dir and os.path.isabs(raw_output_dir)) else os.path.join(workspace_root, raw_output_dir or "outputs")
        summary_data = None
        task_id_val = assigned_tasks[0].task_id if assigned_tasks else 2
        summary_filename = f"ml_summary_task_{task_id_val}.json"
        main_summary_filename = "ml_summary.json"

        # Prefer this task's own summary and reject a stale supervised summary
        # when the current task is unsupervised.
        summary_candidates = (
            [summary_filename, main_summary_filename]
            if task_type == "unsupervised_analysis"
            else [main_summary_filename, summary_filename]
        )
        for candidate_name in summary_candidates:
            cand_path = os.path.join(output_dir, candidate_name)
            if os.path.exists(cand_path):
                try:
                    with open(cand_path, "r", encoding="utf-8") as sf:
                        candidate_summary = json.load(sf)
                    if task_type == "unsupervised_analysis" and isinstance(candidate_summary, dict):
                        summary_type = str(candidate_summary.get("task_type", "")).lower()
                        has_unsupervised_data = "unsupervised_metrics" in candidate_summary
                        if not has_unsupervised_data and not any(
                            tag in summary_type for tag in ("unsupervised", "cluster", "anomal", "dimensionality")
                        ):
                            continue
                    summary_data = candidate_summary
                    if candidate_name not in generated_files:
                        generated_files.append(candidate_name)
                    break
                except Exception:
                    pass

        # If not produced directly by script, assemble from generated artifact files
        if not summary_data and task_type == "unsupervised_analysis":
            unsupervised_metrics = {}
            unsupervised_summary = {}
            for artifact_name, destination in (
                ("unsupervised_metrics.json", "metrics"),
                ("unsupervised_summary.json", "summary"),
            ):
                try:
                    with open(os.path.join(output_dir, artifact_name), "r", encoding="utf-8") as af:
                        value = json.load(af)
                    if isinstance(value, dict):
                        if destination == "metrics":
                            unsupervised_metrics = value
                        else:
                            unsupervised_summary = value
                except Exception:
                    pass
            summary_data = {
                "agent": self.agent_name,
                "task_id": task_id_val,
                "task_title": assigned_tasks[0].title if assigned_tasks else "Unsupervised Analysis",
                "status": final_status,
                "task_type": "unsupervised_analysis",
                "source": "fallback_from_unsupervised_artifacts",
                "unsupervised_metrics": unsupervised_metrics,
                "unsupervised_summary": unsupervised_summary,
                "generated_artifacts": [
                    {"file": gf, "description": f"Generated artifact: {gf}"}
                    for gf in generated_files
                ],
                "issues_and_caveats": [
                    "Fallback summary copied the task's unsupervised metric and summary artifacts; values were not independently recomputed."
                ],
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
                print(f"[MLAgent] Notice saving unsupervised fallback summary: {write_err}")

        if not summary_data and task_type != "unsupervised_analysis":
            # 1. Baseline Score
            baseline_data = {}
            base_path = os.path.join(output_dir, "baseline_score.json")
            if os.path.exists(base_path):
                try:
                    with open(base_path, "r", encoding="utf-8") as bf:
                        baseline_data = json.load(bf)
                except Exception:
                    pass

            # 2. Confusion Matrix
            conf_data = {}
            conf_path = os.path.join(output_dir, "confusion_matrix.json")
            if os.path.exists(conf_path):
                try:
                    with open(conf_path, "r", encoding="utf-8") as cmf:
                        conf_data = json.load(cmf)
                except Exception:
                    pass

            # 3. Model Comparison CSV (head/summary)
            comp_list = []
            comp_path = os.path.join(output_dir, "model_comparison.csv")
            if os.path.exists(comp_path):
                try:
                    import pandas as pd
                    df_comp = pd.read_csv(comp_path)
                    comp_list = df_comp.to_dict(orient="records")
                except Exception:
                    pass

            # 4. Feature Importance CSV (top features with coefficients & direction)
            top_feats = []
            feat_path = os.path.join(output_dir, "feature_importance.csv")
            if os.path.exists(feat_path):
                try:
                    import pandas as pd
                    df_feat = pd.read_csv(feat_path)
                    # Normalize columns if needed
                    for rec in df_feat.head(10).to_dict(orient="records"):
                        feat_name = rec.get("feature") or rec.get("Feature") or rec.get("feature_name") or "feature"
                        coeff_val = rec.get("coefficient") if rec.get("coefficient") is not None else rec.get("importance", 0.0)
                        abs_val = rec.get("absolute_importance") if rec.get("absolute_importance") is not None else abs(float(coeff_val) if isinstance(coeff_val, (int, float)) else 0.0)
                        direction_val = rec.get("direction") or ("reduces_target" if (isinstance(coeff_val, (int, float)) and coeff_val < 0) else "increases_target")
                        top_feats.append({
                            "feature": feat_name,
                            "coefficient": coeff_val,
                            "absolute_importance": abs_val,
                            "direction": direction_val
                        })
                except Exception:
                    pass

            # 5. Prediction Summary from predictions.csv if available
            pred_summary = {}
            pred_path = os.path.join(output_dir, "predictions.csv")
            if os.path.exists(pred_path):
                try:
                    import pandas as pd
                    df_pred = pd.read_csv(pred_path)
                    pred_summary["test_samples"] = len(df_pred)
                    id_cols = [c for c in df_pred.columns if any(k in c.lower() for k in ["id", "customer", "user"])]
                    if id_cols:
                        pred_summary["identifiers_retained"] = id_cols
                except Exception:
                    pass

            # Determine best model and comparative rationale
            best_model_name = comp_list[0].get("Model", "Best Model") if comp_list else "Trained Model"
            best_metrics = comp_list[0] if comp_list else {}

            # Construct business-aware comparative explanation
            comparative_parts = []
            if len(comp_list) > 1:
                # Find top performers per metric if classification
                try:
                    models_summary = ", ".join([f"{m.get('Model')}: ROC-AUC={m.get('ROC_AUC', m.get('roc_auc', 'N/A'))}, F1={m.get('F1', m.get('f1', 'N/A'))}" for m in comp_list[:3]])
                    comparative_explanation = (
                        f"{best_model_name} demonstrated top overall discrimination capability and stability. "
                        f"Evaluated models ({models_summary}). {best_model_name} was selected because primary metric ROC-AUC and PR-AUC were prioritized for robust classification."
                    )
                except Exception:
                    comparative_explanation = f"{best_model_name} was selected as the top-performing candidate based on multi-metric evaluation."
            else:
                comparative_explanation = f"{best_model_name} was selected based on holistic multi-metric performance."

            source_file = "cleaned_dataset.csv" if os.path.exists(os.path.join(output_dir, "cleaned_dataset.csv")) else (os.path.basename(uploaded_files[0]) if uploaded_files else "dataset.csv")

            summary_data = {
                "agent": self.agent_name,
                "task_id": task_id_val,
                "task_title": assigned_tasks[0].title if assigned_tasks else "Machine Learning Modeling",
                "status": final_status,
                "task_type": "classification" if conf_data else "regression",
                # Honesty marker: structurally assembled from artifacts, NOT written
                # by the ML agent. Unspecified details were NOT verified.
                "source": "fallback_structural_summary",
                "target_column": "target",
                "features_used": [f.get("feature") for f in top_feats if isinstance(f, dict)],
                "identifier_columns": [],
                "preprocessing": {
                    "dataset_source": source_file,
                    # NOT fabricated: preprocessing details are left empty unless the
                    # agent recorded them. Empty means "not recorded by this fallback".
                    "imputation_applied": {},
                    "encoding_applied": {},
                    "target_encoding": "",
                    "scaling_applied": "",
                    "train_test_split": {},
                    "transformations": [
                        f"Loaded dataset from {source_file}"
                    ]
                },
                "baseline_performance": baseline_data or {"model": "Dummy baseline", "metrics": {}},
                "model_comparison": comp_list,
                "model_selection": {
                    "selection_metric": "ROC-AUC",
                    "secondary_metrics": ["PR-AUC", "F1", "Recall", "Accuracy"],
                    "best_model": best_model_name,
                    "summary_explanation": comparative_explanation,
                    "business_priority": "Balanced discrimination, ranking capability, and precision"
                },
                "best_model": {
                    "name": best_model_name,
                    "test_metrics": best_metrics,
                    "confusion_matrix": conf_data
                },
                "prediction_summary": pred_summary,
                "top_features": top_feats,
                "generated_artifacts": [
                    {"file": gf, "description": f"Generated artifact: {gf}"}
                    for gf in generated_files
                ],
                "issues_and_caveats": [
                    "FALLBACK SUMMARY: generated structurally from artifacts because the agent did not produce its own summary JSON. Preprocessing details (imputation, encoding, scaling, split) are intentionally left empty rather than assumed."
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
                print(f"[MLAgent] Notice saving fallback summary: {write_err}")

        # Enforce that the task generated ITS OWN designated summary (not a borrowed
        # intermediate from another task type). This check applies to summaries only.
        required_task_summary = (
            "hyperparameter_tuning_summary.json"
            if task_type == "hyperparameter_tuning"
            else "ml_summary.json"
        )
        missing_required_summary = None
        if not os.path.exists(os.path.join(output_dir, required_task_summary)):
            missing_required_summary = required_task_summary
            if HAS_LOGFIRE:
                logfire.warn(
                    "MLAgent task did not produce its own required summary",
                    agent="MLAgent",
                    task_type=task_type,
                    required_summary=required_task_summary,
                    generated_files=generated_files,
                )
            else:
                print(f"[MLAgent] WARNING: task_type '{task_type}' did not produce its required summary '{required_task_summary}'.", flush=True)

        if HAS_LOGFIRE:
            logfire.info(
                "MLAgent Task Finished",
                agent="MLAgent",
                status=final_status,
                tool_calls_used=tool_calls_count,
                generated_files=generated_files,
                assigned_tasks_count=len(assigned_tasks)
            )

        return {
            "status": final_status,
            "agent": self.agent_name,
            "tasks": [t.to_dict() for t in assigned_tasks],
            "tool_calls_used": tool_calls_count,
            "executed_codes": executed_codes,
            "generated_files": generated_files,
            "final_response": final_text,
            "summary_data": summary_data,
            "missing_required_summary": missing_required_summary,
            "token_usage": memory_manager.token_usage(),
        }


if __name__ == "__main__":
    print("Testing MLAgent initialization...")
    ml_agent = MLAgent()
    print("MLAgent initialized successfully with Modal Cloud Sandbox Tool!")
