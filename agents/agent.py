import os
import re
import sys
import json
import traceback
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.tools import execute_python, EXECUTE_PYTHON_TOOL_SPEC, reset_global_sandbox_manager
from agents.logger import setup_logfire, HAS_LOGFIRE
from agents.memory_manager import MessageHistoryManager, DatasetState, count_tokens


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

# Registry of tools available to the agent
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

SYSTEM_PROMPT = """You are AnalystGPT, an expert AI Data Analyst and Business Intelligence assistant.
 
You analyse datasets using a Python execution tool (execute_python) that runs code inside a secure
Modal cloud sandbox. Uploaded files are already in the working directory — read them directly by
filename, e.g. pd.read_csv("sales.csv"), never pd.read_csv("uploads/sales.csv").
 
The sandbox already has: pandas, numpy, matplotlib, scipy, scikit-learn, openpyxl. Only request other
packages if genuinely required.
 
==================================================
HARD RULE: ONE SCRIPT, NOT A CHECKLIST
==================================================
 
No matter how many analysis steps the user's request lists (inspection, cleaning, statistics, groupby
metrics, outliers, trends, multiple charts, insights, recommendations...), that is a description of what
ONE Python script should contain — not a sequence of separate tool calls. Inspection, cleaning,
computation, and chart generation are all steps INSIDE a single script, executed together, not separate
turns.
 
Budget: call execute_python at most 2 times per user request.
  - Call 1: a single script that loads the data, inspects it, cleans it if needed, computes every
    requested metric, detects outliers/trends, and generates and saves every requested chart — all in
    one run, printing clearly labeled results (e.g. "REVENUE_BY_REGION:", "OUTLIERS:") at the end.
  - Call 2 (only if call 1 errors, or the user asks for something new afterward): a corrected or
    additional script. Fix the actual traceback cause — do not re-run unchanged code.
 
Never call execute_python a 3rd time for the same request. If something is still missing at that point,
say so in your final answer and explain what would be needed, rather than making another call.
 
Do not use separate tool calls to "peek" at the dataset (e.g. df.head(), df.info() alone) before the
real analysis — fold that inspection into the same script as the rest of the work.

==================================================
EXECUTION STATE
==================================================

All execute_python calls within the same user request share the same Python session and working directory.

Variables, imported libraries, loaded DataFrames and generated files remain available between retries.

Do not reload the dataset or recreate files unless necessary.

If a retry is required, continue from the existing session whenever possible.
 
==================================================
HOW TO STRUCTURE THE ONE SCRIPT
==================================================
 
Within that single script:
1. Load the data and check shape/dtypes/missing values/duplicates.
2. Clean only what's actually broken (never alter data that doesn't need it).
3. Compute every statistic, aggregation, and metric the request asks for.
4. Detect outliers/anomalies/trends if requested.
5. Generate and save every requested chart (bar, line, scatter, histogram, box, heatmap, pie — pick
   what fits the data and question).
6. Print a clearly labeled summary of every result so you can write the final report from the output —
   don't dump raw unlabeled tables.
 
==================================================
ERROR HANDLING
==================================================
 
If execution fails: read the traceback, fix only the actual cause, and use your one remaining retry.
Never retry identical code.
 
==================================================
FINAL RESPONSE
==================================================
 
Never fabricate or estimate a number that Python could calculate — base every figure on executed output.
Turn the printed results into a professional report, not a data dump: summarise tables, explain what each
chart shows, and separate facts from insights from recommendations. Structure it with sections as
relevant: Executive Summary, Dataset Overview, Data Quality, Key Metrics, Analysis, Visualisations,
Insights, Recommendations, Limitations. Be concise but thorough — write like a senior data analyst, not
a checklist-completion bot.
"""

class DataAnalystAgent:
    def __init__(
        self,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 200,
        max_tokens_before_compact: int = 3000,
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
        kwargs = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        def _do_call():
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
            tool_names = [t.get("function", {}).get("name") for t in tools] if tools else []
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
        max_turns: int = 10
    ) -> str:
        """
        Sends a user prompt to the agent and runs an autonomous tool execution loop 
        using Modal cloud sandbox tools, managed with token-budgeted memory compaction and rich Logfire telemetry.
        """
        prompt_content = self.memory_manager.build_prompt_content(user_prompt, files=files)
        self.memory_manager.append_user(prompt_content)

        def _run_turns():
            for turn in range(max_turns):
                def _do_turn():
                    response_msg = self._call_llm(self.memory_manager.messages, tools=[EXECUTE_PYTHON_TOOL_SPEC])
                    
                    # Serialize Assistant Message to dict for history
                    msg_dict: Dict[str, Any] = {
                        "role": "assistant",
                        "content": response_msg.content,
                    }
                    if response_msg.tool_calls:
                        msg_dict["tool_calls"] = [
                            tc.model_dump() if hasattr(tc, "model_dump") else tc
                            for tc in response_msg.tool_calls
                        ]
                    
                    self.memory_manager.append_assistant(msg_dict)

                    if response_msg.content:
                        if HAS_LOGFIRE:
                            logfire.info("AI Assistant Thought", thought=response_msg.content.strip())
                        else:
                            print(f"\n[AI Assistant Thought]:\n{response_msg.content.strip()}\n", flush=True)

                    tool_calls = response_msg.tool_calls
                    if not tool_calls:
                        if HAS_LOGFIRE:
                            logfire.info("Agent turn loop finished with final response")
                        return response_msg.content or ""

                    if HAS_LOGFIRE:
                        logfire.info("Agent requested tool execution", tool_count=len(tool_calls))
                    else:
                        print(f"\n==================== [Agent Turn {turn + 1}] ====================", flush=True)
                        print(f"[Agent] Model requested {len(tool_calls)} tool execution(s).", flush=True)

                    for tool_call in tool_calls:
                        tool_call_id = tool_call.id
                        fn_name = tool_call.function.name
                        raw_args = tool_call.function.arguments or "{}"
                        
                        # Robust JSON decoding with explicit Logfire error capturing
                        parse_error = None
                        try:
                            fn_args = json.loads(raw_args)
                        except Exception as parse_err:
                            fn_args = {}
                            parse_error = f"{type(parse_err).__name__}: {str(parse_err)}"
                            if HAS_LOGFIRE:
                                logfire.error(
                                    "LLM Tool Call JSON Argument Parsing Failed",
                                    tool_call_id=tool_call_id,
                                    fn_name=fn_name,
                                    raw_arguments=raw_args,
                                    parse_error=parse_error,
                                    traceback=traceback.format_exc(),
                                )
                            else:
                                print(f"[Tool Parse Error] Failed to parse JSON for '{fn_name}' (ID: {tool_call_id}). Raw: {raw_args}. Error: {parse_error}")

                        def _execute_tool():
                            if fn_name in AVAILABLE_TOOLS:
                                if files and "files" not in fn_args:
                                    fn_args["files"] = files

                                try:
                                    tool_res = AVAILABLE_TOOLS[fn_name](**fn_args)
                                except Exception as err:
                                    tool_res = {
                                        "success": False,
                                        "error": f"Tool execution raised exception: {str(err)}",
                                        "stderr": traceback.format_exc(),
                                    }
                                    if HAS_LOGFIRE:
                                        logfire.error(
                                            "Tool Execution Raised Exception",
                                            fn_name=fn_name,
                                            tool_call_id=tool_call_id,
                                            error=str(err),
                                            traceback=traceback.format_exc(),
                                        )
                            else:
                                tool_res = {
                                    "success": False,
                                    "error": f"Unknown tool requested by model: '{fn_name}'",
                                }
                                if HAS_LOGFIRE:
                                    logfire.warn(
                                        "Model requested unknown tool",
                                        requested_tool=fn_name,
                                        available_tools=list(AVAILABLE_TOOLS.keys()),
                                    )

                            is_success = tool_res.get("success", False)
                            exec_time = tool_res.get("execution_time", 0.0)
                            returned_files = tool_res.get("returned_files", [])
                            stdout_str = str(tool_res.get("stdout", ""))
                            stderr_str = str(tool_res.get("stderr", ""))

                            if HAS_LOGFIRE:
                                log_func = logfire.info if is_success else logfire.error
                                log_func(
                                    f"Tool Execution {'SUCCESS' if is_success else 'FAILED'}: {fn_name}",
                                    fn_name=fn_name,
                                    tool_call_id=tool_call_id,
                                    success=is_success,
                                    execution_time=exec_time,
                                    code=fn_args.get("code", "") if isinstance(fn_args, dict) and "code" in fn_args else None,
                                    returned_files=returned_files,
                                    stdout_snippet=stdout_str[:500],
                                    stderr_snippet=stderr_str[-500:] if len(stderr_str) > 500 else stderr_str,
                                    error=tool_res.get("error"),
                                    parse_error=parse_error,
                                )
                            else:
                                print(f"[Tool Result]: status={'SUCCESS' if is_success else 'FAILED'} (time: {exec_time}s)", flush=True)

                            return tool_res

                        if HAS_LOGFIRE:
                            with logfire.span(
                                f"Tool Call: {fn_name}",
                                fn_name=fn_name,
                                tool_call_id=tool_call_id,
                                raw_arguments=raw_args,
                                parsed_arguments=fn_args,
                                parse_error=parse_error,
                            ):
                                tool_result = _execute_tool()
                        else:
                            print(f"\n[Tool Execution Request]: '{fn_name}' (ID: {tool_call_id})", flush=True)
                            tool_result = _execute_tool()

                        self.memory_manager.append_tool_result(tool_call_id, fn_name, tool_result)

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
            return last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")

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
        output_summary = f"Execution Status: {'Success' if success else 'Failed'}\n"
        if stdout:
            output_summary += f"STDOUT:\n{stdout}\n"
        if stderr:
            output_summary += f"STDERR:\n{stderr}\n"

        feedback_message = (
            f"Code Execution Result:\n{output_summary}\n"
            "Analyze these results, organize findings, and output the next step summary & code snippet (or final insights if complete)."
        )

        self.memory_manager.append_user(feedback_message)
        response_msg = self._call_llm(self.memory_manager.messages)
        res_text = response_msg.content or ""
        self.memory_manager.append_assistant({"role": "assistant", "content": res_text})
        return res_text




if __name__ == "__main__":
    print("Initializing Data Analyst Agent with Modal Cloud Sandbox Tool...")
    agent = DataAnalystAgent()

    dataset_file = "uploads/test.csv"
    user_request = f"""You are an expert business analyst.

The uploaded CSV "{dataset_file}" contains company sales data.

Perform a complete business analysis.

Your workflow should be:

1. Load the CSV using Python.
2. Clean the data if required.
3. Validate column types.
4. Compute descriptive statistics.
5. Calculate:
   - Total Revenue
   - Total Profit
   - Profit Margin
   - Average Order Value
   - Revenue by Region
   - Revenue by Category
   - Revenue by Product
   - Revenue by Salesperson
6. Identify:
   - Best selling products
   - Least selling products
   - Most profitable products
   - Least profitable products
   - Highest performing region
   - Lowest performing region
7. Detect outliers.
8. Find unusual trends.
9. Create appropriate visualisations.
10. Explain every chart.
11. Produce executive insights.
12. Recommend at least 10 actionable business recommendations.

Return:
- Summary
- Tables
- Charts
- Insights
- Recommendations

Use Python whenever calculations or visualisations are required."""

    print(f"\n--- User Request: {user_request} ---")
    final_analysis = agent.prompt_agent(user_prompt=user_request, files=[dataset_file])
    print("\n================ FINAL AGENT ANALYSIS ================")
    print(final_analysis)

