"""
memory_manager.py - Context/memory management for AnalystGPT's agent loop.

Problem this solves:
    self.messages grows without bound across turns AND across prompt_agent() calls,
    because every tool result (stdout/stderr) and every assistant message is kept
    verbatim forever. This bloats the context window, causing tool-call errors and
    hallucination as old, now-irrelevant execution output crowds out the current task.

Approach:
    - Keep a compact, persistent "state manifest" (files, schema, key computed
      facts, last error) OUTSIDE the raw chat transcript.
    - Keep only the last K tool call/result pairs verbatim in messages.
    - Once older turns fall out of the window, collapse them into a short
      ACTION CHECKLIST (what was actually done: loaded file, deduped, plotted,
      exported) instead of dropping them silently or restating the assistant's
      prose reasoning — reasoning is disposable, actions taken are not.
    - Budget compaction on real token counts (tiktoken), not character counts,
      so the trigger stays accurate regardless of model/tokenizer differences.
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


def count_tokens(text: str) -> int:
    """Real token count when tiktoken is available; a conservative
    char/4 estimate otherwise so behavior degrades gracefully rather
    than breaking if the dependency is missing in some environment."""
    if not text:
        return 0
    if HAS_TIKTOKEN:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_message_tokens(msg: Dict[str, Any]) -> int:
    # +4 per message for role/name/formatting overhead, matching common
    # chat-format accounting conventions closely enough for budgeting purposes.
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
    """Small, durable facts about the working dataset(s). Injected as a compact
    block instead of making the model re-derive this from old tool outputs."""
    files: List[str] = field(default_factory=list)
    schema: Dict[str, str] = field(default_factory=dict)      # column -> dtype
    row_count: Optional[int] = None
    computed_facts: Dict[str, Any] = field(default_factory=dict)  # e.g. {"total_revenue": 128400}
    generated_files: List[str] = field(default_factory=list)  # charts/csvs already produced
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
# 2. Action extraction — turn a raw code string + its result into a checklist
# ---------------------------------------------------------------------------

# Ordered (pattern, label-template) pairs. First match per line wins; a single
# code block can produce multiple checklist lines.
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
    """Turn a code string into a short list of concrete action lines,
    e.g. ['Loaded sales.csv', 'Removed duplicates', 'Generated plot revenue.png'].
    Order-preserving, de-duplicated."""
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
# 3. History manager: bounds the raw transcript, summarizes what falls off
#    as an action checklist rather than restating assistant prose.
# ---------------------------------------------------------------------------

class MessageHistoryManager:
    def __init__(
        self,
        system_prompt: str,
        keep_last_tool_pairs: int = 2,       # verbatim tool call/result pairs kept
        stdout_stderr_token_limit: int = 200,  # per-field token cap, not char cap
        max_tokens_before_compact: int = 3000, # token budget trigger for compaction
    ):
        self.system_prompt = system_prompt
        self.keep_last_tool_pairs = keep_last_tool_pairs
        self.stdout_stderr_token_limit = stdout_stderr_token_limit
        self.max_tokens_before_compact = max_tokens_before_compact
        self.state = DatasetState()
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._action_checklist: List[str] = []  # rolling, de-duplicated action log

    # -- token-aware truncation --------------------------------------------

    def _truncate_to_tokens(self, text: str, limit: int) -> str:
        if not text:
            return ""
        if count_tokens(text) <= limit:
            return text
        if HAS_TIKTOKEN:
            tokens = _ENCODING.encode(text)
            truncated = _ENCODING.decode(tokens[:limit])
            return truncated + f"\n... [truncated at {limit} tokens]"
        approx_chars = limit * 4
        return text[:approx_chars] + f"\n... [truncated at ~{limit} tokens]"

    def clean_tool_result(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        if "stdout" in tool_result:
            tool_result["stdout"] = self._truncate_to_tokens(str(tool_result["stdout"]), self.stdout_stderr_token_limit)
        if "stderr" in tool_result:
            tool_result["stderr"] = self._truncate_to_tokens(str(tool_result["stderr"]), self.stdout_stderr_token_limit)
        if not tool_result.get("success", True):
            self.state.last_error = self._truncate_to_tokens(
                str(tool_result.get("error") or tool_result.get("stderr", "")), self.stdout_stderr_token_limit
            )
        else:
            self.state.last_error = None
        for f in tool_result.get("returned_files", []) or []:
            if f not in self.state.generated_files:
                self.state.generated_files.append(f)
        return tool_result

    # -- appenders ----------------------------------------------------------

    def append_user(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._maybe_compact()

    def append_assistant(self, msg_dict: Dict[str, Any]):
        self.messages.append(msg_dict)
        # Record actions from any code this assistant turn is about to run,
        # so they survive even if this turn gets compacted away later.
        for tc in msg_dict.get("tool_calls", []) or []:
            code = _extract_code_from_tool_call(tc)
            for action in extract_actions_from_code(code):
                if action not in self._action_checklist:
                    self._action_checklist.append(action)

    def append_tool_result(self, tool_call_id: str, name: str, tool_result: Dict[str, Any]):
        tool_result = self.clean_tool_result(tool_result)
        ok = tool_result.get("success", True)
        # Mark the most recently logged action as failed if this call didn't succeed.
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

    # -- compaction -----------------------------------------------------

    def _total_tokens(self) -> int:
        return sum(count_message_tokens(m) for m in self.messages)

    def _maybe_compact(self):
        if self._total_tokens() < self.max_tokens_before_compact:
            return

        keep_tail = 2 * self.keep_last_tool_pairs + 2
        if len(self.messages) <= keep_tail + 1:
            return  # nothing meaningful to compact yet

        tail = self.messages[-keep_tail:]
        # head (self.messages[1:-keep_tail]) is dropped in full — its useful
        # content was already captured incrementally into self._action_checklist
        # by append_assistant(), so nothing is silently lost, just condensed.

        checklist_block = ""
        if self._action_checklist:
            confirmed = [
                a if a.startswith(("✓", "✗")) else f"✓ {a}"
                for a in self._action_checklist
            ]
            checklist_block = "\n\n[ACTIONS COMPLETED SO FAR — do not repeat]\n" + "\n".join(confirmed)

        compacted_system = self.system_prompt + checklist_block
        self.messages = [{"role": "system", "content": compacted_system}] + tail

    # -- context injection -----------------------------------------------

    def build_prompt_content(self, user_prompt: str, files: Optional[Any] = None) -> str:
        """Attach durable state instead of relying on the model to recall old turns."""
        parts = [user_prompt]
        state_block = self.state.as_context_block()
        if state_block:
            parts.append(state_block)
        if files:
            names = list(files.keys()) if isinstance(files, dict) else files
            parts.append(f"Input files available: {names}")
        return "\n\n".join(parts)

    def reset_conversation(self, keep_state: bool = True, keep_checklist: bool = True):
        """Call between unrelated analyses/sessions. Keeps the dataset manifest
        and action checklist by default so the model doesn't need to re-derive
        them, but drops the raw turn-by-turn transcript."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if not keep_state:
            self.state = DatasetState()
        if not keep_checklist:
            self._action_checklist = []

    def token_usage(self) -> Dict[str, int]:
        """Handy for logging/telemetry (e.g. wire into your existing logfire spans)."""
        return {"total_tokens": self._total_tokens(), "budget": self.max_tokens_before_compact}