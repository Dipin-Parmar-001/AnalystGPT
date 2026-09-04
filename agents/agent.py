import os
import re
import sys
import json
import traceback
import time
from typing import List, Dict, Any, Optional, Union, Callable
from dotenv import load_dotenv
from openai import OpenAI


# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sandbox_execution import ToolManager, execute_python, EXECUTE_PYTHON_TOOL_SPEC
from agents.tools import reset_global_sandbox_manager, extract_code_from_content, create_synthetic_execute_tool_call
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.memory_manager import MessageHistoryManager, DatasetState, count_tokens, sanitize_messages, convert_to_plain_chat



try:
    import logfire
except ImportError:
    logfire = None

# Load environment variables
load_dotenv()

PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Legacy dictionary export for backwards compatibility
AVAILABLE_TOOLS = {
    "execute_python": execute_python
}

# Default Portkey Configuration for fallback & multi-provider routing
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

from agents.prompts import SYSTEM_PROMPT

class DataAnalystAgent:
    def __init__(
        self,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 200,
        max_tokens_before_compact: int = 3000,
        tool_manager: Optional[ToolManager] = None,
    ):
        if not PORTKEY_API_KEY:
            raise ValueError("PORTKEY_API_KEY is not set in environment variables.")

        setup_logfire()

        # Set up standard Portkey headers
        headers = {
            "x-portkey-api-key": PORTKEY_API_KEY,
        }

        if PORTKEY_CONFIG_ID:
            if HAS_LOGFIRE:
                logfire.info("DataAnalystAgent Setup using Portkey Config ID", config_id=PORTKEY_CONFIG_ID)
            else:
                print(f"[Agent Setup] Using Portkey Config ID: {PORTKEY_CONFIG_ID}")
            headers["x-portkey-config"] = PORTKEY_CONFIG_ID
        else:
            if HAS_LOGFIRE:
                logfire.warn("PORTKEY_CONFIG_ID not found in .env. Attempting inline fallback config...")
            else:
                print("[Agent Setup] WARNING: PORTKEY_CONFIG_ID not found in .env. Attempting inline fallback config...")
            headers["x-portkey-config"] = json.dumps(DEFAULT_PORTKEY_CONFIG)

        # Single OpenAI client routing through Portkey Gateway API
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

        # Tool execution workflow manager
        self.tool_manager = tool_manager or ToolManager()

        # Memory and history manager instance
        self.memory_manager = MessageHistoryManager(
            system_prompt=SYSTEM_PROMPT,
            keep_last_tool_pairs=keep_last_tool_pairs,
            stdout_stderr_token_limit=stdout_stderr_token_limit,
            max_tokens_before_compact=max_tokens_before_compact,
        )

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self.memory_manager.messages

    @messages.setter
    def messages(self, val: List[Dict[str, Any]]) -> None:
        self.memory_manager.messages = val

    def reset_memory(self, keep_state: bool = True, keep_checklist: bool = True):
        """Resets agent conversation transcript while optionally retaining dataset state and checklist."""
        self.memory_manager.reset_conversation(keep_state=keep_state, keep_checklist=keep_checklist)
        if not keep_state and reset_global_sandbox_manager is not None:
            reset_global_sandbox_manager()

    def token_usage(self) -> Dict[str, int]:
        """Returns token usage telemetry."""
        return self.memory_manager.token_usage()

    def _call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Calls Portkey Gateway which handles multi-provider fallbacks, 
        parameter overrides, and provider routing seamlessly.
        """
        if getattr(self, "_plain_chat_mode", False):
            # Provider (e.g. Gemini) already rejected the OpenAI tool-message schema;
            # send plain user/assistant history while still exposing tools.
            sanitized_messages = convert_to_plain_chat(messages)
        else:
            sanitized_messages = sanitize_messages(messages)
        kwargs = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
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

                    if HAS_LOGFIRE:
                        tool_calls_summary = []
                        if message.tool_calls:
                            for tc in message.tool_calls:
                                tool_calls_summary.append({
                                    "id": tc.id,
                                    "function": tc.function.name,
                                "arguments_raw": tc.function.arguments,
                            })
                    logfire.info(
                        "LLM Completion Received",
                        role=message.role,
                        has_thought=bool(message.content),
                        thought_snippet=message.content[:200] if message.content else None,
                        tool_call_count=len(message.tool_calls or []),
                        tool_calls=tool_calls_summary,
                        finish_reason=getattr(response.choices[0], "finish_reason", None),
                        usage=getattr(response, "usage", None),
                    )
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
                            logfire.warn("Portkey/Provider rejected tool message schema in agent.py. Retrying with plain chat format (tools preserved).", error=str(err))
                        else:
                            print(f"[Agent] Provider tool schema rejected ({err}). Retrying with plain chat format (tools preserved)...")

                        plain_msgs = convert_to_plain_chat(messages)
                        fallback_kwargs = {
                            "model": kwargs.get("model", "nvidia/nemotron-3-ultra-550b-a55b"),
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
                                logfire.error("Agent fallback LLM request failed", error=str(fallback_err))
                            raise RuntimeError(f"Agent LLM call failed after fallback: {fallback_err}") from fallback_err


                if attempt < len(delays):
                    if HAS_LOGFIRE:
                        logfire.warn(f"LLM request failed (attempt {attempt + 1}), retrying in {delay}s...", error=str(err))
                    else:
                        print(f"[{agent} Warning] LLM request failed (attempt {attempt + 1}), retrying in {delay}s... Error: {err}")
                    time.sleep(delay)
                    continue

                if HAS_LOGFIRE:
                    logfire.error(
                        "Portkey Gateway LLM Request Failed",
                        error=str(err),
                        error_type=type(err).__name__,
                        traceback=traceback.format_exc(),
                        model=kwargs.get("model"),
                        message_count=len(messages),
                    )
                else:
                    print(f"[Agent Error] Portkey Gateway request failed: {err}\n{traceback.format_exc()}")
                raise RuntimeError(f"Portkey Gateway call failed: {err}") from err

        if HAS_LOGFIRE:
            tool_names = [t.get("function", {}).get("name") for t in effective_tools] if effective_tools else []
            with logfire.span(
                "LLM Completion Request",
                model="nvidia/nemotron-3-ultra-550b-a55b",
                message_count=len(messages),
                tools_available=tool_names,
            ):
                return _do_call()
        else:
            print("[Agent LLM] Requesting Portkey Gateway...")
            return _do_call()

    def extract_python_code(self, text: str) -> Optional[str]:
        """Extracts executable Python code snippets inside ```python ... ``` markdown blocks."""
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def prompt_agent(
        self, 
        user_prompt: str, 
        files: Optional[Union[List[str], Dict[str, Any]]] = None,
        max_turns: int = 10,
        step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        """
        Sends a user prompt to the agent and runs an autonomous tool execution loop 
        using Modal cloud sandbox tools, managed with token-budgeted memory compaction and rich Logfire telemetry.
        """
        def _safe_callback(event: str, data: Dict[str, Any]):
            if step_callback:
                try:
                    step_callback(event, data)
                except BaseException:
                    pass

        self.memory_manager.prepare_user_prompt(user_prompt, files=files)
        _safe_callback("prompt_started", {"user_prompt": user_prompt, "files": files})

        def _run_turns():
            for turn in range(max_turns):
                _safe_callback("turn_start", {"turn": turn + 1, "max_turns": max_turns})

                def _do_turn():
                    response_msg = self._call_llm(self.memory_manager.messages, tools=self.tool_manager.get_tool_specs())
                    self.memory_manager.append_assistant_response(response_msg)

                    if response_msg.content:
                        if HAS_LOGFIRE:
                            logfire.info("AI Assistant Thought", thought=response_msg.content.strip())
                        else:
                            print(f"\n[AI Assistant Thought]:\n{response_msg.content.strip()}\n", flush=True)
                        _safe_callback("llm_thought", {"turn": turn + 1, "thought": response_msg.content.strip()})

                    tool_calls = response_msg.tool_calls
                    if not tool_calls:
                        fallback_code = extract_code_from_content(response_msg.content or "")
                        if fallback_code:
                            if HAS_LOGFIRE:
                                logfire.info("Agent detected code in assistant response via fallback parser", code_snippet=fallback_code[:200])
                            else:
                                print(f"[Agent Fallback] Extracted Python code from non-standard LLM response ({len(fallback_code)} chars). Executing in sandbox...")
                            synthetic_tc = create_synthetic_execute_tool_call(fallback_code, f"fallback_call_{turn + 1}")
                            tool_calls = [synthetic_tc]
                        else:
                            if HAS_LOGFIRE:
                                logfire.info("Agent turn loop finished with final response")
                            _safe_callback("turn_finished", {"turn": turn + 1, "response": response_msg.content or ""})
                            return response_msg.content or ""


                    if HAS_LOGFIRE:
                        logfire.info("Agent requested tool execution", tool_count=len(tool_calls))
                    else:
                        print(f"\n==================== [Agent Turn {turn + 1}] ====================", flush=True)
                        print(f"[Agent] Model requested {len(tool_calls)} tool execution(s).", flush=True)

                    _safe_callback("tool_calls_requested", {"turn": turn + 1, "tool_count": len(tool_calls)})

                    for tool_call in tool_calls:
                        tool_call_id = getattr(tool_call, "id", None) or tool_call.get("id", "")
                        fn_name = tool_call.function.name if hasattr(tool_call, "function") else tool_call.get("function", {}).get("name", "")
                        raw_args = tool_call.function.arguments if hasattr(tool_call, "function") else tool_call.get("function", {}).get("arguments", "{}")

                        _safe_callback("tool_executing", {"turn": turn + 1, "tool_call_id": tool_call_id, "fn_name": fn_name, "raw_args": raw_args})
                        
                        tool_result = self.tool_manager.execute_tool_call(tool_call, files=files)
                        self.memory_manager.append_tool_result(tool_call_id, fn_name, tool_result)

                        _safe_callback("tool_completed", {"turn": turn + 1, "tool_call_id": tool_call_id, "fn_name": fn_name, "result": tool_result})

                    return None

                if HAS_LOGFIRE:
                    with logfire.span(f"Agent Turn {turn + 1}", turn_number=turn + 1, token_usage=self.memory_manager.token_usage()):
                        turn_result = _do_turn()
                        if turn_result is not None:
                            return turn_result
                else:
                    turn_result = _do_turn()
                    if turn_result is not None:
                        return turn_result

            last_msg = self.memory_manager.messages[-1]
            final_txt = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
            if step_callback:
                step_callback("prompt_finished", {"turn": max_turns, "response": final_txt})
            return final_txt

        if HAS_LOGFIRE:
            with logfire.span("Prompt Agent Run", user_prompt=user_prompt[:200], files=files, token_usage=self.memory_manager.token_usage()):
                return _run_turns()
        else:
            return _run_turns()


    def send_execution_results(self, stdout: str, stderr: str, success: bool = True) -> str:
        """
        Feeds external execution outputs (stdout, stderr, status) from a cloud sandbox
        or runner environment back into the agent conversation loop.
        """
        self.memory_manager.append_external_feedback(stdout, stderr, success)
        response_msg = self._call_llm(self.memory_manager.messages)
        res_text = response_msg.content or ""
        self.memory_manager.append_assistant_response(response_msg)
        return res_text


if __name__ == "__main__":
    print("Initializing Data Analyst Agent with Modal Cloud Sandbox Tool...")
    agent = DataAnalystAgent()

    dataset_file = "uploads/customer_churn_dataset.csv"
    user_request = f"""You are an expert data scientist.

Analyze the attached customer churn dataset '{dataset_file}' and perform a complete machine learning workflow.

Requirements:

1. Load and inspect the dataset.
2. Check for missing values, duplicate rows, and incorrect data types.
3. Perform exploratory data analysis with meaningful visualizations.
4. Explain the relationships between important features and customer churn.
5. Encode categorical variables appropriately.
6. Split the dataset into training and testing sets.
7. Train multiple machine learning classification models including:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost (if available)
8. Perform hyperparameter tuning where appropriate.
9. Compare all models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
   - Confusion Matrix
10. Select the best-performing model based on the evaluation metrics.
11. Display feature importance (or model coefficients if applicable).
12. Predict churn probability for the entire dataset using the best model.
13. Save the predictions as:
    predictions.csv
14. Save every generated plot as PNG files.
15. Write a detailed summary explaining:
    - Which model performed best
    - Why it performed best
    - The most important factors affecting churn
    - Practical business recommendations for reducing churn

Important:
- Do not stop after training one model.
- Automatically recover from minor errors if they occur.
- Produce publication-quality plots.
- Save every output file."""

    print(f"\n--- User Request: {user_request} ---")
    final_analysis = agent.prompt_agent(user_prompt=user_request, files=[dataset_file])
    print("\n================ FINAL AGENT ANALYSIS ================")
    print(final_analysis)
