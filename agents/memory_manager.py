"""
memory_manager.py - Context and Memory Management for AnalystGPT Agent Loop.

Problem solved:
    Prevents agent message history from growing without bound across turns and calls.
    Verbatim tool output (stdout/stderr) and assistant turns are budgeted and compacted.

Approach:
    - Maintains a compact, durable `DatasetState` (files, schema, row counts, key facts, last error)
      outside raw chat transcript.
    - Keeps the last K tool call/result pairs verbatim in messages.
    - Converts older completed turns into a compact ACTION CHECKLIST so key completed steps
      (e.g., loaded dataset, cleaned missing values, generated charts) survive compaction.
    - Budgets token usage using real BPE token counts (`tiktoken`), with a fallback estimate if unavailable.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except Exception:
    _ENCODING = None
    HAS_TIKTOKEN = False

try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    logfire = None
    HAS_LOGFIRE = False



def count_tokens(text: str) -> int:
    """Real token count when tiktoken is available; a conservative
    char/4 estimate otherwise so behavior degrades gracefully."""
    if not text:
        return 0
    if HAS_TIKTOKEN and _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_message_tokens(msg: Dict[str, Any]) -> int:
    """Estimates token budget for a single message dictionary in chat format."""
    total = 4
    content = msg.get("content")
    if content:
        total += count_tokens(content if isinstance(content, str) else json.dumps(content, default=str))
    if msg.get("tool_calls"):
        total += count_tokens(json.dumps(msg["tool_calls"], default=str))
    if msg.get("name"):
        total += count_tokens(msg["name"])
    return total


# ---------------------------------------------------------------------------
# 1. Structured state that survives across turns/prompts, kept OUT of messages
# ---------------------------------------------------------------------------

@dataclass
class DatasetState:
    """Small, durable facts about working dataset(s). Injected as a compact
    context block instead of forcing the model to re-derive from raw tool output."""
    files: List[str] = field(default_factory=list)
    schema: Dict[str, str] = field(default_factory=dict)         # column -> dtype
    row_count: Optional[int] = None
    computed_facts: Dict[str, Any] = field(default_factory=dict) # e.g. {"total_revenue": 128400}
    generated_files: List[str] = field(default_factory=list)     # charts/csvs already produced
    last_error: Optional[str] = None

    def as_context_block(self) -> str:
        if not self.files:
            return ""
        lines = [
            "[DATASET STATE — already established, do not re-derive]",
            f"Files: {self.files}",
        ]
        if self.schema:
            lines.append(f"Schema: {self.schema}")
        if self.row_count is not None:
            lines.append(f"Rows: {self.row_count}")
        if self.computed_facts:
            lines.append(f"Known facts: {json.dumps(self.computed_facts, default=str)}")
        if self.generated_files:
            lines.append(f"Already generated: {self.generated_files}")
        if self.last_error:
            lines.append(f"Last error (fix before retrying same approach): {self.last_error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Action extraction — turn code execution into a structured checklist
# ---------------------------------------------------------------------------

_ACTION_PATTERNS = [
    (r'pd\.read_csv\(\s*["\']([^"\']+)["\']', "Loaded {0}"),
    (r'pd\.read_excel\(\s*["\']([^"\']+)["\']', "Loaded {0}"),
    (r'\.drop_duplicates\(', "Removed duplicates"),
    (r'\.dropna\(', "Dropped missing values"),
    (r'\.fillna\(', "Filled missing values"),
    (r'\.astype\(', "Converted column dtypes"),
    (r'\.groupby\(', "Aggregated data (groupby)"),
    (r'\.merge\(|\.join\(', "Merged datasets"),
    (r'\.to_csv\(\s*["\']([^"\']+)["\']', "Exported {0}"),
    (r'\.to_excel\(\s*["\']([^"\']+)["\']', "Exported {0}"),
    (r'plt\.savefig\(\s*["\']([^"\']+)["\']', "Generated plot {0}"),
    (r'\.corr\(', "Computed correlations"),
    (r'\.describe\(', "Computed summary statistics"),
    (r'sklearn|LinearRegression|KMeans|train_test_split', "Ran ML step"),
]


def extract_actions_from_code(code: str) -> List[str]:
    """Extracts high-level action summaries from executed Python code snippets."""
    if not code:
        return []
    actions: List[str] = []
    seen = set()
    for pattern, template in _ACTION_PATTERNS:
        for match in re.finditer(pattern, code):
            groups = match.groups()
            label = template.format(*groups) if groups else template
            if label not in seen:
                seen.add(label)
                actions.append(label)
    return actions


def _extract_code_from_tool_call(tool_call: Dict[str, Any]) -> str:
    try:
        fn = tool_call.get("function", {})
        args = json.loads(fn.get("arguments", "{}"))
        return args.get("code", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. History Manager: Bounds raw transcript and compacts old turns
# ---------------------------------------------------------------------------

def convert_to_plain_chat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts message history containing tool_calls and tool messages into
    a strictly alternating user/assistant plain chat transcript with markdown code blocks.
    
    This format is guaranteed to work seamlessly with ANY LLM provider (Google Gemini,
    OpenRouter, Nvidia, Mistral, OpenAI, etc.) even when Portkey or the provider's API
    rejects native tool/function roles.
    """
    if not messages:
        return [{"role": "user", "content": "Please proceed with the assigned task."}]

    system_contents: List[str] = []
    plain_conv: List[Dict[str, Any]] = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            c = m.get("content", "")
            if c:
                system_contents.append(str(c))
        elif role == "user":
            plain_conv.append({"role": "user", "content": str(m.get("content", ""))})
        elif role == "assistant":
            content = str(m.get("content") or "").strip()
            tool_calls = m.get("tool_calls", [])
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                args_str = fn.get("arguments", "{}") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    if isinstance(args, dict) and "code" in args:
                        content += f"\n\n```python\n{args['code']}\n```"
                except Exception:
                    pass
            plain_conv.append({"role": "assistant", "content": content or "Executing required tool..."})
        elif role in ("tool", "function"):
            fn_name = m.get("name", "execute_python")
            tool_content = m.get("content", "")
            if isinstance(tool_content, (dict, list)):
                tool_content = json.dumps(tool_content, default=str)
            else:
                tool_content = str(tool_content)
            plain_conv.append({
                "role": "user",
                "content": f"[Tool Execution Result for {fn_name}]:\n{tool_content}\n\nPlease analyze these execution results and proceed with the next step, code snippet, or final summary."
            })

    # Merge consecutive messages with same role to ensure strict alternating turn structure
    merged_conv: List[Dict[str, Any]] = []
    for msg in plain_conv:
        if not merged_conv:
            merged_conv.append(msg)
        else:
            prev = merged_conv[-1]
            if prev["role"] == msg["role"]:
                prev["content"] = prev["content"] + "\n\n" + msg["content"]
            else:
                merged_conv.append(msg)

    # Gemini & Anthropic require the conversation after system to start with 'user'
    if merged_conv and merged_conv[0]["role"] != "user":
        merged_conv.insert(0, {"role": "user", "content": "Please proceed with the assigned data analysis workflow."})

    final_msgs: List[Dict[str, Any]] = []
    if system_contents:
        final_msgs.append({"role": "system", "content": "\n\n".join(system_contents)})
    final_msgs.extend(merged_conv)
    return final_msgs


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitizes message history to strictly conform to OpenAI and Google Gemini API schemas.
    
    Guarantees:
    1. At most one system message at index 0.
    2. First non-system turn is a 'user' turn.
    3. Every 'assistant' message with 'tool_calls' is immediately followed by 'tool' messages
       for every single tool_call_id present in that assistant message.
    4. No orphan 'tool' messages (every 'tool' message must correspond to an active tool_call_id
       in the immediately preceding 'assistant' message).
    5. No 'assistant' tool_call turn directly after 'system' without an intervening 'user' turn.
    6. Converts any legacy 'function' role into 'tool' or 'user' format.
    7. Properly handles tool limit / partial turns without breaking API contracts.
    """
    if not messages:
        return [{"role": "user", "content": "Please proceed."}]

    system_contents: List[str] = []
    raw_conv_messages: List[Dict[str, Any]] = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            content = m.get("content", "")
            if content:
                system_contents.append(str(content))
        elif role in ("user", "assistant", "tool"):
            raw_conv_messages.append(m)
        elif role == "function":
            # Legacy function role conversion
            converted = {
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", f"call_{m.get('name', 'tool')}"),
                "name": m.get("name", "tool"),
                "content": m.get("content", ""),
            }
            raw_conv_messages.append(converted)

    if not raw_conv_messages:
        raw_conv_messages = [{"role": "user", "content": "Please proceed with the assigned task."}]

    sanitized_conv: List[Dict[str, Any]] = []

    # Gemini requires the conversation after system to start with a 'user' turn
    if raw_conv_messages[0].get("role") != "user":
        sanitized_conv.append({"role": "user", "content": "Please execute the assigned data analysis workflow."})

    i = 0
    while i < len(raw_conv_messages):
        msg = raw_conv_messages[i]
        role = msg.get("role")

        if role == "user":
            sanitized_conv.append(msg)
            i += 1

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                sanitized_conv.append(msg)
                i += 1
            else:
                # Assistant with tool_calls: MUST come after 'user' or 'tool'
                if not sanitized_conv or sanitized_conv[-1].get("role") not in ("user", "tool"):
                    sanitized_conv.append({"role": "user", "content": "Please execute the next required tool call."})

                sanitized_conv.append(msg)
                i += 1

                # Gather expected tool_call_ids
                expected_ids: Dict[str, str] = {}
                for tc in tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    fn_name = ""
                    if isinstance(tc, dict):
                        fn_name = tc.get("function", {}).get("name", "tool")
                    elif hasattr(tc, "function"):
                        fn_name = getattr(tc.function, "name", "tool")
                    if tc_id:
                        expected_ids[tc_id] = fn_name or "tool"

                # Look ahead for matching tool messages
                tool_responses: Dict[str, Dict[str, Any]] = {}
                while i < len(raw_conv_messages) and raw_conv_messages[i].get("role") == "tool":
                    t_msg = raw_conv_messages[i]
                    t_id = t_msg.get("tool_call_id")
                    if t_id and t_id in expected_ids:
                        tool_responses[t_id] = t_msg
                    i += 1

                # For every expected tool_call_id, emit tool response (or synthetic stub if missing)
                for tc_id, fn_name in expected_ids.items():
                    if tc_id in tool_responses:
                        sanitized_conv.append(tool_responses[tc_id])
                    else:
                        sanitized_conv.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": fn_name,
                            "content": json.dumps({"status": "completed", "note": f"Execution for {fn_name} recorded."}),
                        })

        elif role == "tool":
            # Orphan tool response without preceding assistant tool_calls
            t_content = msg.get("content", "")
            t_name = msg.get("name", "tool")
            sanitized_conv.append({
                "role": "user",
                "content": f"[Tool Execution Output - {t_name}]: {t_content}"
            })
            i += 1
        else:
            i += 1

    final_messages: List[Dict[str, Any]] = []
    if system_contents:
        final_messages.append({"role": "system", "content": "\n\n".join(system_contents)})

    final_messages.extend(sanitized_conv)
    return final_messages


class MessageHistoryManager:
    """
    Manages chat message history for agent turns.
    Enforces token caps, truncates execution logs, tracks dataset state,
    and compacts completed turns into action checklists.
    """
    def __init__(
        self,
        system_prompt: str,
        keep_last_tool_pairs: int = 2,
        stdout_stderr_token_limit: int = 200,
        max_tokens_before_compact: int = 3000,
    ):
        self.system_prompt = system_prompt
        self.keep_last_tool_pairs = keep_last_tool_pairs
        self.stdout_stderr_token_limit = stdout_stderr_token_limit
        self.max_tokens_before_compact = max_tokens_before_compact
        self.state = DatasetState()
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._action_checklist: List[str] = []

    def _truncate_to_tokens(self, text: str, limit: int, truncate_from_start: bool = False) -> str:
        if not text:
            return ""
        if count_tokens(text) <= limit:
            return text
        if HAS_TIKTOKEN and _ENCODING is not None:
            try:
                tokens = _ENCODING.encode(text)
                if truncate_from_start:
                    truncated = _ENCODING.decode(tokens[-limit:])
                    return f"... [truncated from start, showing last {limit} tokens]\n" + truncated
                else:
                    truncated = _ENCODING.decode(tokens[:limit])
                    return truncated + f"\n... [truncated at {limit} tokens]"
            except Exception:
                pass
        approx_chars = limit * 4
        if truncate_from_start:
            return f"... [truncated from start, showing last ~{limit} tokens]\n" + text[-approx_chars:]
        return text[:approx_chars] + f"\n... [truncated at ~{limit} tokens]"

    def _truncate_keeping_ends(self, text: str, limit: int) -> str:
        """Truncates by dropping the MIDDLE of the text, keeping head and tail.

        Execution scripts print their diagnostic conclusion (mismatch sets,
        metric summaries, final assert lines) at the END of stdout, so
        head-only truncation hides exactly the output the agent needs and
        causes it to re-run the same probe. stderr keeps tail-only truncation
        (see clean_tool_result) because tracebacks put the error last.
        """
        if not text:
            return ""
        total = count_tokens(text)
        if total <= limit:
            return text
        head_limit = max(1, int(limit * 0.4))
        tail_limit = max(1, limit - head_limit)
        if HAS_TIKTOKEN and _ENCODING is not None:
            try:
                tokens = _ENCODING.encode(text)
                head = _ENCODING.decode(tokens[:head_limit])
                tail = _ENCODING.decode(tokens[-tail_limit:])
            except Exception:
                approx_chars = limit * 4
                head = text[: int(approx_chars * 0.4)]
                tail = text[-int(approx_chars * 0.6):]
        else:
            approx_chars = limit * 4
            head = text[: int(approx_chars * 0.4)]
            tail = text[-int(approx_chars * 0.6):]
        return (
            f"{head}\n"
            f"... [middle truncated: {total} tokens -> first {head_limit} + last {tail_limit}] ...\n"
            f"{tail}"
        )

    def clean_tool_result(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        if "stdout" in tool_result:
            tool_result["stdout"] = self._truncate_keeping_ends(str(tool_result["stdout"]), self.stdout_stderr_token_limit)
        if "stderr" in tool_result:
            tool_result["stderr"] = self._truncate_to_tokens(str(tool_result["stderr"]), self.stdout_stderr_token_limit, truncate_from_start=True)
        if not tool_result.get("success", True):
            self.state.last_error = self._truncate_to_tokens(
                str(tool_result.get("error") or tool_result.get("stderr", "")), self.stdout_stderr_token_limit, truncate_from_start=True
            )
        else:
            self.state.last_error = None
        for f in tool_result.get("returned_files", []) or []:
            if f not in self.state.generated_files:
                self.state.generated_files.append(f)
        return tool_result

    def append_user(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._maybe_compact()

    def append_assistant(self, msg_dict: Dict[str, Any]):
        self.messages.append(msg_dict)
        for tc in msg_dict.get("tool_calls", []) or []:
            code = _extract_code_from_tool_call(tc)
            for action in extract_actions_from_code(code):
                if action not in self._action_checklist:
                    self._action_checklist.append(action)

    def trim_intermediate_failures(self):
        """
        Strict History Trimming: Drops intermediate tool execution failures, raw logs,
        and repetitive retry attempts from the agent's active memory once a task/tool succeeds.
        Removes both the assistant message that issued the failed tool call and the corresponding
        tool failure response, maintaining strict OpenAI and Gemini chat API message alignment.
        """
        if len(self.messages) <= 3:
            return

        # Identify all tool_call_ids that failed
        failed_tool_ids = set()
        for msg in self.messages:
            if msg.get("role") == "tool":
                try:
                    content_str = msg.get("content", "{}")
                    content_obj = json.loads(content_str) if isinstance(content_str, str) else content_str
                    if isinstance(content_obj, dict) and not content_obj.get("success", True):
                        tid = msg.get("tool_call_id")
                        if tid:
                            failed_tool_ids.add(tid)
                except Exception:
                    pass

        if not failed_tool_ids:
            return

        cleaned_messages = []
        cleaned_messages.append(self.messages[0])

        i = 1
        last_turn_start = max(1, len(self.messages) - 2)

        while i < len(self.messages):
            msg = self.messages[i]

            if i < last_turn_start:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tcs = msg.get("tool_calls", [])
                    if tcs and all(tc.get("id") in failed_tool_ids for tc in tcs):
                        i += 1
                        continue

                elif msg.get("role") == "tool":
                    if msg.get("tool_call_id") in failed_tool_ids:
                        i += 1
                        continue

            cleaned_messages.append(msg)
            i += 1

        self.messages = sanitize_messages(cleaned_messages)

    def append_tool_result(self, tool_call_id: str, name: str, tool_result: Dict[str, Any]):
        tool_result = self.clean_tool_result(tool_result)
        ok = tool_result.get("success", True)
        if self._action_checklist and not ok:
            last = self._action_checklist[-1]
            if not last.startswith(("✓", "✗")):
                self._action_checklist[-1] = f"✗ {last} (failed)"
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(tool_result, default=str),
        })
        if ok:
            self.trim_intermediate_failures()
        self._maybe_compact()

    def _total_tokens(self) -> int:
        return sum(count_message_tokens(m) for m in self.messages)

    def _maybe_compact(self):
        before_tokens = self._total_tokens()
        if before_tokens < self.max_tokens_before_compact:
            return

        # Preserve the initial user prompt so context is never severed
        initial_user_msg = None
        for m in self.messages:
            if m.get("role") == "user":
                initial_user_msg = m
                break

        keep_tail = 2 * self.keep_last_tool_pairs + 2
        if len(self.messages) <= keep_tail + 1:
            return  # nothing meaningful to compact yet

        tail = self.messages[-keep_tail:]

        checklist_block = ""
        if self._action_checklist:
            confirmed = [
                a if a.startswith(("✓", "✗")) else f"✓ {a}"
                for a in self._action_checklist
            ]
            checklist_block = "\n\n[ACTIONS COMPLETED SO FAR — do not repeat]\n" + "\n".join(confirmed)

        compacted_system = self.system_prompt + checklist_block
        reconstructed = [{"role": "system", "content": compacted_system}]
        if initial_user_msg:
            reconstructed.append(initial_user_msg)
        reconstructed.extend(tail)

        self.messages = sanitize_messages(reconstructed)
        after_tokens = self._total_tokens()

        if HAS_LOGFIRE:
            logfire.info(
                "Memory History Compacted",
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                retained_messages=len(self.messages),
                action_checklist_count=len(self._action_checklist),
                action_checklist=self._action_checklist,
            )


    def build_prompt_content(self, user_prompt: str, files: Optional[Any] = None) -> str:
        parts = [user_prompt]
        state_block = self.state.as_context_block()
        if state_block:
            parts.append(state_block)
        if files:
            names = list(files.keys()) if isinstance(files, dict) else files
            parts.append(f"Input files available: {names}")
            for n in names:
                if isinstance(n, str) and n not in self.state.files:
                    self.state.files.append(n)
        return "\n\n".join(parts)

    def prepare_user_prompt(self, user_prompt: str, files: Optional[Any] = None) -> str:
        """Formats context block and input files, appends to message history, and returns prompt content."""
        prompt_content = self.build_prompt_content(user_prompt, files=files)
        self.append_user(prompt_content)
        return prompt_content

    def append_assistant_response(self, response_msg: Any) -> Dict[str, Any]:
        """Serializes LLM response message object into history dictionary and updates action checklist."""
        msg_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": response_msg.content,
        }
        if getattr(response_msg, "tool_calls", None):
            msg_dict["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in response_msg.tool_calls
            ]
        self.append_assistant(msg_dict)
        return msg_dict

    def append_external_feedback(self, stdout: str, stderr: str, success: bool = True) -> str:
        """Formats and appends external code execution results into history."""
        output_summary = f"Execution Status: {'Success' if success else 'Failed'}\n"
        if stdout:
            output_summary += f"STDOUT:\n{stdout}\n"
        if stderr:
            output_summary += f"STDERR:\n{stderr}\n"

        feedback_message = (
            f"Code Execution Result:\n{output_summary}\n"
            "Analyze these results, organize findings, and output the next step summary & code snippet (or final insights if complete)."
        )
        self.append_user(feedback_message)
        return feedback_message

    def reset_conversation(self, keep_state: bool = True, keep_checklist: bool = True):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if not keep_state:
            self.state = DatasetState()
        if not keep_checklist:
            self._action_checklist = []

    def token_usage(self) -> Dict[str, int]:
        return {"total_tokens": self._total_tokens(), "budget": self.max_tokens_before_compact}

