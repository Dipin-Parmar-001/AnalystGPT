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

    def clean_tool_result(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        if "stdout" in tool_result:
            tool_result["stdout"] = self._truncate_to_tokens(str(tool_result["stdout"]), self.stdout_stderr_token_limit, truncate_from_start=False)
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
        self._maybe_compact()

    def _total_tokens(self) -> int:
        return sum(count_message_tokens(m) for m in self.messages)

    def _maybe_compact(self):
        before_tokens = self._total_tokens()
        if before_tokens < self.max_tokens_before_compact:
            return

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
        self.messages = [{"role": "system", "content": compacted_system}] + tail
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

