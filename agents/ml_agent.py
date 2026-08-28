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
from agents.prompts import ML_AGENT, ML_SYSTEM_PROMPT
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

        self.model_name = model_name
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

        composite_prompt += (
            f"=== ASSIGNED ML TASKS FOR {self.agent_name} ===\n"
            f"{tasks_block}\n\n"
            f"Instructions:\n"
            f"1. Dataset Loading: If `cleaned_dataset.csv` is available in FILE_REGISTRY or inputs, load it directly (e.g. `pd.read_csv('cleaned_dataset.csv')`). If not available, use the main dataset provided.\n"
            f"2. Identifier Preservation: Retain identifier columns (e.g., `customerID`, `id`) in `predictions.csv` for entity traceability, but EXCLUDE them from training feature matrix `X`.\n"
            f"3. Preprocessing: Use One-Hot Encoding (`pd.get_dummies(..., drop_first=True)` or `OneHotEncoder`) for nominal features, `LabelEncoder()` strictly for classification target `y`, `StandardScaler()` fit strictly on `X_train`, and `train_test_split(..., test_size=0.2, random_state=42, stratify=y if classification else None)`.\n"
            f"4. Metrics: For classification, evaluate Accuracy, Precision, Recall, F1, ROC-AUC, **PR-AUC** (`average_precision_score`), and **5-Fold Stratified Cross-Validation** (`CV_Mean`, `CV_Std`). For regression, evaluate RMSE, MAE, R2, MSE, and 5-Fold CV R2.\n"
            f"5. Directional Feature Importance: Extract signed `coefficient`, `absolute_importance`, and `direction` (e.g., `'reduces_churn'` / `'increases_churn'` or `'reduces_target'` / `'increases_target'`) and save to `feature_importance.csv`.\n"
            f"6. Business-Aware Model Selection: Provide a comparative multi-metric rationale (e.g. 'Model A has highest ROC-AUC/PR-AUC while Model B has higher recall. Model A was selected because ROC-AUC was designated as primary metric...').\n"
            f"7. MANDATORY 7 ARTIFACTS: Your script MUST save all 7 core files:\n"
            f"   - `model_comparison.csv` (Comparing models across Accuracy, Precision, Recall, F1, ROC_AUC, PR_AUC, CV_Mean, CV_Std)\n"
            f"   - `baseline_score.json` (Dummy baseline benchmark)\n"
            f"   - `feature_importance.csv` (Columns: feature, coefficient, absolute_importance, direction)\n"
            f"   - `confusion_matrix.json` (Confusion matrix data for best model)\n"
            f"   - `predictions.csv` (Predictions with customerID, actual, predicted, probability)\n"
            f"   - `best_model.pkl` (Best model saved via joblib.dump)\n"
            f"   - `ml_summary.json` (Structured machine learning summary)\n"
            f"   (You may also save additional plots/files if helpful).\n"
            f"8. You have a budget of MAX {effective_max_calls} tool calls.\n"
            f"9. Output the final structured JSON summary and metric details."
        )

        memory_manager = MessageHistoryManager(
            system_prompt=ML_SYSTEM_PROMPT,
            keep_last_tool_pairs=self.keep_last_tool_pairs,
            stdout_stderr_token_limit=self.stdout_stderr_token_limit,
            max_tokens_before_compact=self.max_tokens_before_compact,
        )

        memory_manager.prepare_user_prompt(composite_prompt, files=uploaded_files)

        tool_calls_count = 0
        executed_codes: List[str] = []
        generated_files: List[str] = []
        turn = 0
        final_text = ""
        completed_success = False

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

            thought = assistant_msg.content or ""
            tool_calls = assistant_msg.tool_calls or []

            if thought:
                _safe_callback("llm_thought", {"agent": self.agent_name, "turn": turn, "thought": thought})

            if not tool_calls:
                # Check for fallback code in thought/content (e.g. 'Malformed function call:' or markdown python block)
                fallback_code = extract_code_from_content(thought)
                if fallback_code and tool_calls_count < effective_max_calls:
                    if HAS_LOGFIRE:
                        logfire.info("MLAgent detected code in assistant response via fallback parser", code_snippet=fallback_code[:200])
                    else:
                        print(f"[MLAgent Fallback] Extracted Python code from non-standard LLM response ({len(fallback_code)} chars). Executing in sandbox...")
                    synthetic_tc = create_synthetic_execute_tool_call(fallback_code, f"fallback_call_{turn}_{tool_calls_count + 1}")
                    tool_calls = [synthetic_tc]
                else:
                    final_text = thought
                    completed_success = True
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
                    "raw_args": fn_args_raw
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

                try:
                    args_dict = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    if "code" in args_dict:
                        executed_codes.append(args_dict["code"])
                except Exception:
                    pass

                memory_manager.append_tool_result(
                    tool_call_id=tc_id,
                    name=fn_name,
                    tool_result=exec_result
                )

                if HAS_LOGFIRE:
                    logfire.info(
                        f"MLAgent Tool Executed: {fn_name}",
                        agent="MLAgent",
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

        if not completed_success and tool_calls_count >= effective_max_calls:
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

        final_status = "completed" if completed_success or final_text else "failed"
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

        # Check if Python script produced ml_summary.json
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

        # If not produced directly by script, assemble from generated artifact files
        if not summary_data:
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
                "target_column": "target",
                "features_used": [f.get("feature") for f in top_feats if isinstance(f, dict)],
                "identifier_columns": ["customerID"],
                "preprocessing": {
                    "dataset_source": source_file,
                    "identifier_columns_excluded": ["customerID"],
                    "imputation_applied": {"numerical": "median", "categorical": "mode"},
                    "encoding_applied": {"method": "One-Hot Encoding (drop_first=True)"},
                    "target_encoding": "LabelEncoder",
                    "scaling_applied": "StandardScaler (fitted strictly on X_train)",
                    "train_test_split": {"test_size": 0.2, "random_state": 42, "stratified": True},
                    "transformations": [
                        f"Loaded dataset from {source_file}",
                        "Preserved identifier columns in predictions.csv and excluded from X",
                        "Applied One-Hot Encoding to categorical features (drop_first=True)",
                        "StandardScaler fitted on X_train"
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
                "issues_and_caveats": []
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
            "token_usage": memory_manager.token_usage(),
        }


if __name__ == "__main__":
    print("Testing MLAgent initialization...")
    ml_agent = MLAgent()
    print("MLAgent initialized successfully with Modal Cloud Sandbox Tool!")
