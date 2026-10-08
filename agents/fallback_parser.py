"""
Fallback Parser & Code Extraction Utilities - AnalystGPT

Handles non-standard, malformed, or envelope-wrapped LLM responses across providers
(e.g., Gemini / OpenRouter / Portkey returning 'Malformed function call:', raw markdown code blocks,
or HTTP response envelopes) and extracts executable Python code or JSON payloads seamlessly.
"""

import re
import json
from typing import Optional, Dict, Any, Union, List, Tuple, Callable

try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    HAS_LOGFIRE = False


class SyntheticFunction:
    def __init__(self, name: str, arguments: Union[str, Dict[str, Any]]):
        self.name = name
        if isinstance(arguments, dict):
            self.arguments = json.dumps(arguments)
        else:
            self.arguments = str(arguments)

    def __getitem__(self, item):
        if item == "name":
            return self.name
        elif item == "arguments":
            return self.arguments
        raise KeyError(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default


class SyntheticToolCall:
    """
    Object compatible with both OpenAI SDK ChatCompletionMessageToolCall
    (attribute access: tc.id, tc.function.name, tc.function.arguments)
    and dictionary access (tc['id'], tc['function']['name']).
    """
    def __init__(self, id: str, name: str, arguments: Union[str, Dict[str, Any]]):
        self.id = id
        self.type = "function"
        self.function = SyntheticFunction(name, arguments)

    def __getitem__(self, item):
        if item == "id":
            return self.id
        elif item == "type":
            return self.type
        elif item == "function":
            return self.function
        raise KeyError(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments
            }
        }


def unwrap_envelope(response_or_text: Any) -> Any:
    """
    Unwraps Portkey/OpenAI HTTP REST envelopes like:
    {
      "status": 200,
      "body": {
        "choices": [
          { "message": { "content": "..." } }
        ]
      }
    }
    or returns the original object/string.
    """
    if isinstance(response_or_text, str):
        text = response_or_text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                return unwrap_envelope(parsed)
            except Exception:
                return response_or_text
        return response_or_text

    if isinstance(response_or_text, dict):
        # Case: { "status": 200, "body": { "choices": ... } }
        if "body" in response_or_text and isinstance(response_or_text["body"], dict):
            return unwrap_envelope(response_or_text["body"])
        
        # Case: { "choices": [ { "message": { ... } } ] }
        if "choices" in response_or_text and isinstance(response_or_text["choices"], list) and len(response_or_text["choices"]) > 0:
            first_choice = response_or_text["choices"][0]
            if isinstance(first_choice, dict):
                msg = first_choice.get("message", {})
                if isinstance(msg, dict):
                    return msg.get("content", "")
                return str(first_choice)

        # Case: { "message": { "content": "..." } }
        if "message" in response_or_text and isinstance(response_or_text["message"], dict):
            return response_or_text["message"].get("content", "")

    return response_or_text


def extract_code_from_content(content: Optional[str]) -> Optional[str]:
    """
    Extracts executable Python code ONLY from responses the provider itself
    flagged as failed tool calls ('Malformed function call: ...').

    Plain assistant content — markdown reports, JSON summaries, prose, or
    fenced code blocks shown for documentation — is a NON-EXECUTION response
    and is never returned as executable code. Whether a task is complete is
    decided by the execution layer (tool_calls executed successfully and
    artifacts validated), never by inspecting response text.

    Handles (inside a 'Malformed function call:' body only):
    1. A JSON payload: {"name": "execute_python", "arguments": {"code": "..."}}
    2. Function-call syntax: execute_python(code=...)
    3. A ```python fenced block
    4. Raw Python script text
    """
    if not content:
        return None

    # First unwrap any JSON envelope string if present
    unwrapped = unwrap_envelope(content)
    if isinstance(unwrapped, str):
        raw_text = unwrapped.strip()
    else:
        raw_text = str(unwrapped).strip()

    if not raw_text:
        return None

    # Only a provider-issued malformed-function-call marker may yield code.
    malformed_prefix_match = re.match(r"^Malformed function call:?\s*", raw_text, re.IGNORECASE)
    if not malformed_prefix_match:
        # Plain content (markdown summary, JSON, prose) is not executable.
        return None

    stripped_body = raw_text[malformed_prefix_match.end():].strip()
    if not stripped_body:
        return None

    # 1a. The remainder is a JSON tool-call payload
    if stripped_body.startswith("{") and stripped_body.endswith("}"):
        try:
            parsed_json = json.loads(stripped_body)
            if isinstance(parsed_json, dict):
                if "arguments" in parsed_json and isinstance(parsed_json["arguments"], dict):
                    if "code" in parsed_json["arguments"]:
                        return parsed_json["arguments"]["code"].strip()
                if "code" in parsed_json:
                    return str(parsed_json["code"]).strip()
        except Exception:
            pass

    # 1b. Function call syntax execute_python(...)
    fn_syntax_match = re.match(r"^execute_python\s*\(\s*(?:code\s*=\s*)?([\"']{1,3})(.*?)\1\s*\)$", stripped_body, re.DOTALL)
    if fn_syntax_match:
        return fn_syntax_match.group(2).strip()

    # 1c. An explicitly python-fenced block inside the malformed body
    py_block_match = re.search(r"```python[ \t]*\r?\n(.*?)```", stripped_body, re.DOTALL | re.IGNORECASE)
    if py_block_match and py_block_match.group(1).strip():
        return py_block_match.group(1).strip()

    # 1d. The remainder itself is raw Python code. Lenient here: the provider
    # already declared this body to be a function call, so a single code
    # indicator is sufficient.
    if looks_like_python_code(stripped_body, lenient=True):
        return stripped_body

    return None


# Language that signals a final report/summary rather than an action response.
_SUMMARY_MARKER_RE = re.compile(
    r"(summary of findings|task complete|artifacts generated|executive summary|"
    r"key insights|data overview|next steps|findings:)",
    re.IGNORECASE,
)


def _is_markdown_report(text: str) -> bool:
    """True when unfenced content looks like a markdown report/summary.

    Used ONLY for unfenced raw content: fenced ```python blocks are always
    execution candidates regardless of the prose around them, because a
    deliberately python-labelled block is code the agent wants to run (and it
    may legitimately be a small snippet inside a larger message).
    """
    if not text:
        return False
    if _SUMMARY_MARKER_RE.search(text):
        return True
    md_lines = [l for l in text.splitlines() if l.lstrip().startswith(("#", "-", "* ", "|"))]
    return len(md_lines) >= 4


def extract_action_code_from_content(content: Optional[str]) -> Optional[str]:
    """
    Accurately detects when the agent put executable code directly in the
    response content (no tool_calls, no malformed-call marker) and returns
    that code. Returns None for non-execution responses.

    Accuracy rules:
    - An explicitly ```python-fenced block is always an execution candidate:
      the model deliberately labelled it as Python, and it may be a small
      snippet inside a larger message. It is returned if it looks like
      runnable Python.
    - Unfenced raw content is executed only when it strictly looks like Python
      (multiple code indicators) and is not a markdown report/summary.
    - Unfenced markdown report/summary or JSON content is a non-execution
      response and never returned as code.
    """
    if not content:
        return None

    unwrapped = unwrap_envelope(content)
    raw_text = unwrapped.strip() if isinstance(unwrapped, str) else str(unwrapped).strip()
    if not raw_text:
        return None

    # Provider-flagged malformed calls are handled by the dedicated path.
    if re.match(r"^Malformed function call:?\s*", raw_text, re.IGNORECASE):
        return extract_code_from_content(raw_text)

    # Explicitly python-fenced blocks: always execution candidates.
    block_pattern = re.compile(r"```python[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    for b in block_pattern.findall(raw_text):
        b = b.strip()
        if b and looks_like_python_code(b):
            return b
    if block_pattern.search(raw_text):
        # A python-fenced block exists but doesn't look runnable: not code.
        return None

    # No fences: the entire content must strictly be a Python script, not prose.
    if looks_like_python_code(raw_text) and not _is_markdown_report(raw_text):
        return raw_text

    return None


def interpret_response(response_or_message: Any) -> Dict[str, Any]:
    """
    Structurally classifies an LLM response using provider-level fields only
    (message.tool_calls and finish_reason) — never by inspecting the text.

    Returns:
        {
          "mode": "tool_calls" | "final_text",
          "tool_calls": [...] or None,
          "content": str,
          "finish_reason": str or None,
        }

    "tool_calls" means the provider requested actual tool execution and the
    response must be passed to the execution layer. "final_text" means the
    provider ended its turn with a natural-language/summary response: this is
    a non-execution turn and must never be executed as code.
    """
    message = unwrap_envelope(response_or_message)

    # Drill into chat.completion envelopes to the message level
    if isinstance(message, dict) and "choices" in message:
        first = message["choices"][0]
        finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
        message = first.get("message", {}) if isinstance(first, dict) else {}
    else:
        finish_reason = None

    content = ""
    tool_calls = None

    if isinstance(message, dict):
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or None
    else:
        content = getattr(message, "content", None) or ""
        tool_calls = getattr(message, "tool_calls", None) or None
        if finish_reason is None:
            # Messages may carry the finish_reason from the enclosing choice
            finish_reason = getattr(message, "finish_reason", None)

    mode = "tool_calls" if tool_calls else "final_text"

    return {
        "mode": mode,
        "tool_calls": tool_calls,
        "content": content,
        "finish_reason": finish_reason,
    }


def looks_like_python_code(text: str, lenient: bool = False) -> bool:
    """
    Heuristics to determine if a string contains executable Python code.

    Conservative by default: markdown/prose summaries and JSON payloads that
    merely *mention* pandas/sklearn calls must never be treated as code.
    With lenient=True (used only for provider-flagged malformed function-call
    bodies, where the provider already declared the content to be a call),
    a single code indicator is sufficient.
    """
    if not text:
        return False

    # Reject markdown documents: headers, bullet lists, tables, nested fences.
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    md_markers = ("#", "-", "* ", "> ", "|", "+ ", "```")
    md_lines = sum(1 for l in lines if l.lstrip().startswith(md_markers))
    if lines and md_lines / len(lines) > 0.3:
        return False

    # Indicators weighted toward code that actually manipulates data/objects,
    # not prose (e.g. `print(` alone can appear in quoted snippets in prose).
    code_indicators = [
        r"^\s*import\s+[a-zA-Z0-9_\.]+",
        r"^\s*from\s+[a-zA-Z0-9_.]+\s+import\s+",
        r"pd\.read_csv\s*\(",
        r"pd\.DataFrame\s*\(",
        r"np\.[a-zA-Z_]+\s*\(",
        r"plt\.[a-zA-Z_]+\s*\(",
        r"sns\.[a-zA-Z_]+\s*\(",
        r"sklearn",
        r"train_test_split\s*\(",
        r"\.fit\s*\(",
        r"\.predict\s*\(",
        r"\.to_csv\s*\(",
        r"joblib\.dump\s*\(",
        r"json\.dump\s*\(",
        r"def\s+[a-zA-Z0-9_]+\s*\(.*\)\s*:",
        r"^if\s+.*:\s*$",
        r"^for\s+[a-zA-Z_]\w*\s+in\s+.*:\s*$",
    ]

    matches = 0
    for pattern in code_indicators:
        if re.search(pattern, text, re.MULTILINE):
            matches += 1
            if matches >= 2 or (lenient and matches >= 1):
                return True

    # If it starts with import/from and has several code-like lines
    if len(lines) >= 3:
        first_line = lines[0].strip()
        if first_line.startswith("import ") or first_line.startswith("from "):
            return True

    return False


def create_synthetic_execute_tool_call(code: str, call_id: Optional[str] = None) -> SyntheticToolCall:
    """
    Creates a SyntheticToolCall for execute_python that can be processed
    identically to an OpenAI function tool call.
    """
    tid = call_id or f"fallback_call_{abs(hash(code)) % 1000000}"
    return SyntheticToolCall(
        id=tid,
        name="execute_python",
        arguments={"code": code}
    )


def repair_json_escapes(raw_str: str) -> str:
    r"""
    Repairs invalid escape sequences in JSON strings.
    RFC 8259 allows only \", \\, \/, \b, \f, \n, \r, \t, and \uXXXX.
    LLMs frequently output Python code with lone backslashes (line continuations '\\ ', '\\\n'),
    escaped single quotes (\\'), or regex escapes (\\d, \\w, \\s, etc.) which cause JSONDecodeError.
    This replaces any backslash NOT followed by a valid JSON escape character with double-backslash (\\\\).
    """
    if not raw_str:
        return raw_str
    pattern = r'\\(?!(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))'
    return re.sub(pattern, r'\\\\', raw_str)


def parse_and_clean_tool_args(raw_args: Any) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Robustly parses and sanitizes LLM tool call arguments into a Python dictionary.

    Returns:
        (is_valid, parsed_dict, error_message)
    """
    if isinstance(raw_args, dict):
        return True, raw_args, None

    if raw_args is None:
        return True, {}, None

    raw_str = str(raw_args).strip()
    if not raw_str or raw_str == "{}":
        return True, {}, None

    # Check for provider prefix "Malformed function call: ..."
    if re.match(r"^Malformed function call:?\s*", raw_str, re.IGNORECASE):
        code = extract_code_from_content(raw_str)
        if code:
            return True, {"code": code}, None

    # Attempt 1: Standard json.loads
    first_err = None
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, dict):
            return True, parsed, None
    except Exception as e1:
        first_err = e1

    # Attempt 2: json.loads with strict=False (allows unescaped control characters)
    try:
        parsed = json.loads(raw_str, strict=False)
        if isinstance(parsed, dict):
            return True, parsed, None
    except Exception:
        pass

    # Attempt 3: Repair invalid backslash escapes + strict=False
    try:
        repaired = repair_json_escapes(raw_str)
        parsed = json.loads(repaired, strict=False)
        if isinstance(parsed, dict):
            return True, parsed, None
    except Exception:
        pass

    # Attempt 4: If JSON was wrapped in markdown ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_str)
    if fence_match:
        fenced_content = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced_content, strict=False)
            if isinstance(parsed, dict):
                return True, parsed, None
        except Exception:
            try:
                repaired = repair_json_escapes(fenced_content)
                parsed = json.loads(repaired, strict=False)
                if isinstance(parsed, dict):
                    return True, parsed, None
            except Exception:
                pass

    # Attempt 5: Extraction of "code" field if it looks like an execute_python tool call
    code_field_match = re.search(r'["\']code["\']\s*:\s*(?:"""|\'\'\'|")([\s\S]*?)(?:"""|\'\'\'|")\s*(?:,\s*["\']files["\']|\}\s*$)', raw_str)
    if code_field_match:
        extracted = code_field_match.group(1).strip()
        if looks_like_python_code(extracted, lenient=True):
            return True, {"code": extracted}, None

    err_msg = f"{type(first_err).__name__}: {str(first_err)}" if first_err else "Failed to parse tool arguments as JSON"
    return False, {}, err_msg


def validate_or_retry_tool_calls(
    agent_name: str,
    assistant_msg: Any,
    memory_manager: Any,
    call_llm_fn: Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], Any],
    tool_specs: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 2,
    cancellation_token: Optional[Any] = None,
    step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[Any, List[Any]]:
    """
    Validates tool call arguments on assistant_msg.
    If parsing fails for any tool call:
    - Retries immediately with the LLM providing the exact JSON parse error.
    - NEVER adds the unparseable tool call or its failed execution to memory_manager,
      preventing message history corruption.
    - Returns (final_assistant_msg, clean_tool_calls).
    """
    curr_msg = assistant_msg
    curr_tool_calls = getattr(curr_msg, "tool_calls", None) or []
    if not curr_tool_calls and isinstance(curr_msg, dict):
        curr_tool_calls = curr_msg.get("tool_calls", []) or []

    if not curr_tool_calls:
        return curr_msg, []

    for retry_idx in range(max_retries + 1):
        if cancellation_token:
            cancellation_token.check_cancelled(step_name=f"{agent_name} validate tool calls (retry {retry_idx})")

        all_valid = True
        first_bad_tool = None
        first_bad_err = None

        cleaned_tool_calls = []
        for tc in curr_tool_calls:
            fn_name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
            raw_args = tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "{}")

            ok, parsed_args, err = parse_and_clean_tool_args(raw_args)
            if not ok:
                all_valid = False
                first_bad_tool = (fn_name, raw_args)
                first_bad_err = err
                break
            else:
                canonical_args = json.dumps(parsed_args)
                if hasattr(tc, "function"):
                    tc.function.arguments = canonical_args
                elif isinstance(tc, dict) and "function" in tc:
                    tc["function"]["arguments"] = canonical_args
                cleaned_tool_calls.append(tc)

        if all_valid:
            if hasattr(curr_msg, "tool_calls"):
                curr_msg.tool_calls = cleaned_tool_calls
            elif isinstance(curr_msg, dict):
                curr_msg["tool_calls"] = cleaned_tool_calls
            return curr_msg, cleaned_tool_calls

        # If retries exhausted, exit retry loop
        if retry_idx >= max_retries:
            break

        # A tool call failed parsing: DO NOT add to memory_manager!
        fn_name, raw_args = first_bad_tool
        err_msg = str(first_bad_err)

        if HAS_LOGFIRE:
            logfire.warn(
                f"{agent_name} tool argument parse error, retrying without corrupting history",
                agent=agent_name,
                tool=fn_name,
                parse_error=err_msg,
                raw_args_snippet=str(raw_args)[:300],
                retry_attempt=retry_idx + 1,
                max_retries=max_retries,
            )
        else:
            print(
                f"[{agent_name} Parse Error] Tool '{fn_name}' arguments failed to parse ({err_msg}). "
                f"Retrying LLM (attempt {retry_idx + 1}/{max_retries}) without corrupting history...",
                flush=True
            )

        if step_callback:
            try:
                step_callback("tool_parse_error", {
                    "agent": agent_name,
                    "tool": fn_name,
                    "error": err_msg,
                    "raw_args": str(raw_args)[:300],
                    "retry_attempt": retry_idx + 1,
                })
            except Exception:
                pass

        retry_prompt = (
            f"[SYSTEM ERROR: Tool Call Argument Parsing Failed]\n"
            f"Your previous attempt to call tool '{fn_name}' failed because the arguments were malformed or invalid JSON:\n"
            f"Error: {err_msg}\n\n"
            f"Raw arguments received:\n{str(raw_args)[:500]}\n\n"
            f"Please retry immediately and re-issue the tool call with strictly valid, properly escaped JSON (RFC 8259). "
            f"Ensure all quotes, newlines, and backslashes in code or arguments are properly escaped."
        )

        # Construct retry messages from clean memory_manager history
        base_messages = getattr(memory_manager, "messages", memory_manager)
        retry_messages = list(base_messages)
        thought_content = (getattr(curr_msg, "content", None) or (curr_msg.get("content") if isinstance(curr_msg, dict) else "") or "").strip()
        if thought_content:
            retry_messages.append({"role": "assistant", "content": thought_content})
        retry_messages.append({"role": "user", "content": retry_prompt})

        curr_msg = call_llm_fn(retry_messages, tools=tool_specs)
        curr_tool_calls = getattr(curr_msg, "tool_calls", None) or []
        if not curr_tool_calls and isinstance(curr_msg, dict):
            curr_tool_calls = curr_msg.get("tool_calls", []) or []

        if not curr_tool_calls:
            return curr_msg, []

    # If retries exhausted and still invalid:
    fn_name, raw_args = first_bad_tool
    fallback_code = extract_action_code_from_content(str(raw_args)) or extract_action_code_from_content(getattr(curr_msg, "content", None) or "")
    if fallback_code and (fn_name == "execute_python" or fn_name == "unknown" or not fn_name):
        if HAS_LOGFIRE:
            logfire.info(f"{agent_name} recovered unparseable tool call via fallback parser after retries")
        else:
            print(f"[{agent_name} Fallback] Recovered unparseable tool call via fallback code extraction.", flush=True)
        synthetic_tc = create_synthetic_execute_tool_call(fallback_code, "fallback_call_repair")
        return curr_msg, [synthetic_tc]

    print(f"[{agent_name} Error] Tool arguments could not be parsed after {max_retries} retries. Skipping corrupted tool call to preserve history.", flush=True)
    if hasattr(curr_msg, "tool_calls"):
        curr_msg.tool_calls = []
    elif isinstance(curr_msg, dict):
        curr_msg["tool_calls"] = []
    return curr_msg, []
