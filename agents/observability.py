"""
Unified Observability Module - AnalystGPT.

Mirrors every Logfire log to a local per-run JSONL file (logs/run_<ts>_<pid>.jsonl),
tracks global latency/token counters for LLM calls, tool executions, workflow steps
and retries, and provides aggregation used by the observability dashboard API.

Design constraints:
- Logfire behavior is untouched: patching only ADDS local mirroring.
- The local file is the source of truth; one unified file per run.
- Large string payloads (code, stdout, stderr, prompts) are smart-chunked from the
  MIDDLE at 15k tokens (head + tail kept) so lines stay bounded without losing
  context from either end.
- Instrumentation can never break a workflow call: every hook is exception-guarded.
"""

import os
import sys
import json
import time
import threading
import contextvars
from contextlib import contextmanager
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Callable

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(WORKSPACE_ROOT, "logs")

# Large payloads in log records are chunked from the middle beyond this many tokens.
TOKEN_CHUNK_LIMIT = 15000

try:
    import tiktoken as _tiktoken

    _ENCODING = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None

try:
    from agents.memory_manager import count_tokens, count_message_tokens
except Exception:  # standalone import safety; fall back to local implementations
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        if _ENCODING is not None:
            try:
                return len(_ENCODING.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_message_tokens(msg: Dict[str, Any]) -> int:
        total = 4
        content = msg.get("content")
        if content:
            total += count_tokens(content if isinstance(content, str) else json.dumps(content, default=str))
        if msg.get("tool_calls"):
            total += count_tokens(json.dumps(msg["tool_calls"], default=str))
        return total

try:
    import logfire as _logfire_module
    HAS_LOGFIRE = True
except ImportError:
    _logfire_module = None
    HAS_LOGFIRE = False


# ---------------------------------------------------------------------------
# Smart chunking: drop the MIDDLE of overlong text, keep head and tail.
# ---------------------------------------------------------------------------

def _token_slice(text: str, head_tokens: int, tail_tokens: int):
    """Returns (head_text, tail_text) covering ~head_tokens from the start and
    ~tail_tokens from the end of text."""
    if _ENCODING is not None:
        try:
            toks = _ENCODING.encode(text)
            if len(toks) <= head_tokens + tail_tokens:
                return text, ""
            head = _ENCODING.decode(toks[:head_tokens])
            tail = _ENCODING.decode(toks[-tail_tokens:])
            return head, tail
        except Exception:
            pass
    total_tokens = max(1, count_tokens(text))
    chars_per_token = max(1, len(text) // total_tokens)
    head = text[: head_tokens * chars_per_token]
    tail = text[-tail_tokens * chars_per_token:] if tail_tokens * chars_per_token < len(text) else ""
    return head, tail


def smart_chunk_text(text: Any, token_limit: int = TOKEN_CHUNK_LIMIT) -> Any:
    """
    Keeps head + tail of strings longer than token_limit tokens, dropping the
    middle, with an explicit truncation marker. Non-strings pass through.
    """
    if not isinstance(text, str):
        return text
    total = count_tokens(text)
    if total <= token_limit:
        return text
    head_limit = int(token_limit * 0.4)
    tail_limit = token_limit - head_limit
    head, tail = _token_slice(text, head_limit, tail_limit)
    marker = f"\n... [middle truncated: {total} tokens -> first {head_limit} + last {tail_limit} tokens kept] ...\n"
    return f"{head}{marker}{tail}" if tail else f"{head}{marker}"


def chunk_value(value: Any, token_limit: int = TOKEN_CHUNK_LIMIT) -> Any:
    """Recursively applies smart chunking to every string in a JSON-like value."""
    if isinstance(value, str):
        return smart_chunk_text(value, token_limit)
    if isinstance(value, dict):
        return {k: chunk_value(v, token_limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [chunk_value(v, token_limit) for v in value]
    return value


def _estimate_prompt_tokens(messages: Any) -> int:
    """Token estimate of the full outgoing message list (history + prompt)."""
    if not isinstance(messages, list):
        return 0
    total = 0
    for m in messages:
        try:
            if isinstance(m, dict):
                total += count_message_tokens(m)
            else:
                content = getattr(m, "content", None)
                total += count_tokens(content) if isinstance(content, str) else 0
        except Exception:
            continue
    return total


# ---------------------------------------------------------------------------
# Task context: attributes LLM/tool events to the current agent AND task.
# ---------------------------------------------------------------------------

_task_ctx: contextvars.ContextVar = contextvars.ContextVar("obs_task_context", default=None)


def set_task_context(task_id: Any = None, agent: Optional[str] = None,
                     phase: Optional[str] = None, title: Optional[str] = None):
    """Sets the current task attribution; returns a token for reset_task_context."""
    return _task_ctx.set({
        "task_id": task_id,
        "agent": agent,
        "phase": phase,
        "title": title,
    })


def reset_task_context(token) -> None:
    try:
        _task_ctx.reset(token)
    except Exception:
        pass


@contextmanager
def task_context(task_id: Any = None, agent: Optional[str] = None,
                 phase: Optional[str] = None, title: Optional[str] = None):
    token = set_task_context(task_id=task_id, agent=agent, phase=phase, title=title)
    try:
        yield
    finally:
        reset_task_context(token)


# ---------------------------------------------------------------------------
# Latency statistics: avg / min / max / p50 / p90 / p95 / p99
# ---------------------------------------------------------------------------

def latency_stats(durations: List[float]) -> Dict[str, Any]:
    if not durations:
        return {"count": 0, "total_s": 0.0}
    s = sorted(durations)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return round(s[idx], 6)

    return {
        "count": n,
        "total_s": round(sum(s), 6),
        "avg_s": round(sum(s) / n, 6),
        "min_s": round(s[0], 6),
        "max_s": round(s[-1], 6),
        "p50_s": pct(0.50),
        "p90_s": pct(0.90),
        "p95_s": pct(0.95),
        "p99_s": pct(0.99),
    }


def _new_bucket() -> Dict[str, Any]:
    return {
        "calls": 0,
        "success": 0,
        "failed": 0,
        "durations": [],
        "tokens": {"prompt": 0, "completion": 0, "total": 0, "new": 0, "cached": 0},
    }


def _bucket_record(bucket: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "calls": bucket["calls"],
        "success": bucket["success"],
        "failed": bucket["failed"],
        "failure_rate": round(bucket["failed"] / bucket["calls"], 4) if bucket["calls"] else 0.0,
        "latency": latency_stats(bucket["durations"]),
        "tokens": dict(bucket["tokens"]),
    }
    return out


# ---------------------------------------------------------------------------
# Event aggregation shared by live snapshots and the dashboard API.
# ---------------------------------------------------------------------------

MAX_CALLS_IN_MEMORY = 300   # slimmed per-call lists kept for metrics.json
MAX_EVENTS_IN_MEMORY = 5000

_SLIM_TOOL_FIELDS = ("ts", "event", "agent", "task_id", "phase", "fn_name",
                     "tool_call_id", "success", "duration_s", "returned_files",
                     "error", "parse_error")
_SLIM_LLM_FIELDS = ("ts", "agent", "task_id", "phase", "duration_s", "success",
                    "requested_model", "served_model", "provider", "error",
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "new_prompt_tokens", "cached_tokens", "finish_reason",
                    "tool_call_count")


def aggregate_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduces a run's event list into the dashboard/metrics summary structure."""
    meta: Dict[str, Any] = {"run_id": None, "status": None, "source": None,
                            "started_at": None, "ended_at": None, "output_dir": None}
    counters: Dict[str, float] = {}
    logs_by_level: Dict[str, int] = {}
    events_by_type: Dict[str, int] = {}

    llm_overall = _new_bucket()
    llm_by_agent: Dict[str, Dict[str, Any]] = {}
    llm_by_task: Dict[str, Dict[str, Any]] = {}
    llm_by_model: Dict[str, Dict[str, Any]] = {}
    llm_calls: List[Dict[str, Any]] = []

    tools_overall = _new_bucket()
    tools_by_name: Dict[str, Dict[str, Any]] = {}
    tool_calls: List[Dict[str, Any]] = []

    tasks: Dict[Any, Dict[str, Any]] = {}
    steps: List[Dict[str, Any]] = []
    open_steps: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []
    custom_metrics: Dict[str, List[float]] = {}
    key_metrics: List[str] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        event = ev.get("event")
        events_by_type[event] = events_by_type.get(event, 0) + 1

        if event == "plan_created":
            for m in (ev.get("key_metrics") or []):
                if m and m not in key_metrics:
                    key_metrics.append(m)
            if ev.get("primary_metric") and ev.get("primary_metric") not in key_metrics:
                key_metrics.append(ev.get("primary_metric"))

        elif event == "counter":
            key = ev.get("counter") or "unknown"
            counters[key] = counters.get(key, 0) + float(ev.get("amount") or 1)
        elif event == "duration":
            custom_metrics.setdefault(ev.get("metric") or "unknown", []).append(
                float(ev.get("duration_s") or 0.0))
        elif event == "counter_duration":
            counters[ev.get("counter") or "unknown"] = counters.get(ev.get("counter") or "unknown", 0) + 1
            custom_metrics.setdefault(ev.get("metric") or "unknown", []).append(
                float(ev.get("duration_s") or 0.0))

        if event == "run_started":
            meta["run_id"] = ev.get("run_id") or meta["run_id"]
            meta["source"] = ev.get("run_source", meta["source"])
            meta["started_at"] = ev.get("ts", meta["started_at"])
            meta["output_dir"] = ev.get("output_dir", meta["output_dir"])
        elif event == "run_summary":
            meta["ended_at"] = ev.get("ts", meta["ended_at"])
            snap = ev.get("snapshot") or {}
            meta["status"] = ev.get("status") or snap.get("status") or meta["status"]
            for k, v in (snap.get("counters") or {}).items():
                counters[k] = v
        elif event == "run_finished":
            meta["ended_at"] = ev.get("ts", meta["ended_at"])
            meta["status"] = ev.get("status") or meta["status"]

        if ev.get("level") in ("error", "warn", "warning", "info", "debug"):
            logs_by_level[ev["level"]] = logs_by_level.get(ev["level"], 0) + 1
        if ev.get("level") == "error" and len(errors) < 100:
            errors.append(ev)

        if event == "llm_call":
            dur = float(ev.get("duration_s") or 0.0)
            ok = bool(ev.get("success", True))
            served = ev.get("served_model") or ev.get("requested_model") or "unknown"
            agent = ev.get("agent") or "unknown"
            task_key = str(ev.get("task_id")) if ev.get("task_id") is not None else (ev.get("phase") or "adhoc")
            usage = ev.get("usage") or {}
            tok = {
                "prompt": int(usage.get("prompt_tokens") or ev.get("prompt_tokens") or 0),
                "completion": int(usage.get("completion_tokens") or ev.get("completion_tokens") or 0),
                "total": int(usage.get("total_tokens") or ev.get("total_tokens") or 0),
                "new": int(ev.get("new_prompt_tokens") or 0),
                "cached": int(ev.get("cached_tokens")
                              or usage.get("cached_tokens")
                              or ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)),
            }
            for bucket in (llm_overall,
                           llm_by_agent.setdefault(agent, _new_bucket()),
                           llm_by_task.setdefault(task_key, _new_bucket()),
                           llm_by_model.setdefault(served, _new_bucket())):
                bucket["calls"] += 1
                bucket["success" if ok else "failed"] += 1
                bucket["durations"].append(dur)
                for k in tok:
                    bucket["tokens"][k] += tok[k]
            if len(llm_calls) < MAX_CALLS_IN_MEMORY:
                llm_calls.append({k: ev.get(k) for k in _SLIM_LLM_FIELDS if k in ev})

        elif event == "tool_executed":
            dur = float(ev.get("duration_s") or 0.0)
            ok = bool(ev.get("success", False))
            fn = ev.get("fn_name") or ev.get("tool") or "unknown"
            for bucket in (tools_overall,
                           tools_by_name.setdefault(fn, _new_bucket())):
                bucket["calls"] += 1
                bucket["success" if ok else "failed"] += 1
                bucket["durations"].append(dur)
            if len(tool_calls) < MAX_CALLS_IN_MEMORY:
                tool_calls.append({k: ev.get(k) for k in _SLIM_TOOL_FIELDS if k in ev})

        elif event == "task_retry":
            t = tasks.setdefault(ev.get("task_id"), {"task_id": ev.get("task_id"),
                                                     "title": ev.get("title"),
                                                     "agent": ev.get("agent"),
                                                     "status": None,
                                                     "duration_s": None,
                                                     "primary_metric": ev.get("primary_metric"),
                                                     "retries": 0})
            t["retries"] = max(t.get("retries", 0), int(ev.get("attempt") or 0))
            if ev.get("primary_metric"):
                t["primary_metric"] = ev.get("primary_metric")
                if ev.get("primary_metric") not in key_metrics:
                    key_metrics.append(ev.get("primary_metric"))

        elif event == "task_completed":
            t = tasks.setdefault(ev.get("task_id"), {"task_id": ev.get("task_id"),
                                                     "title": ev.get("title"),
                                                     "agent": ev.get("agent"),
                                                     "status": ev.get("status"),
                                                     "duration_s": ev.get("duration_s"),
                                                     "primary_metric": ev.get("primary_metric"),
                                                     "retries": 0})
            t.update({"title": ev.get("title", t.get("title")),
                      "agent": ev.get("agent", t.get("agent")),
                      "status": ev.get("status", t.get("status")),
                      "duration_s": ev.get("duration_s", t.get("duration_s"))})
            if ev.get("primary_metric"):
                t["primary_metric"] = ev.get("primary_metric")
                if ev.get("primary_metric") not in key_metrics:
                    key_metrics.append(ev.get("primary_metric"))
            if ev.get("retries") is not None:
                t["retries"] = int(ev.get("retries") or 0)

        elif event in ("step_start", "span_start"):
            is_span = event == "span_start"
            key = ev.get("span") if is_span else ev.get("step")
            if not key:
                continue
            entry = {"step": key,
                     "start_ts": ev.get("ts"),
                     "end_ts": None,
                     "duration_s": None,
                     "status": "running",
                     "source": "span" if is_span else "step"}
            if is_span:
                entry["task_id"] = ev.get("task_id")
                entry["agent"] = ev.get("agent")
            entry.update({k: v for k, v in ev.items()
                          if k not in ("ts", "event", "level", "run_id",
                                       "task_id", "agent", "phase", "step", "span")})
            open_steps[key] = entry
        elif event in ("step_end", "span_end"):
            is_span = event == "span_end"
            key = ev.get("span") if is_span else ev.get("step")
            if not key:
                continue
            st = open_steps.pop(key, None)
            if st is None:
                # end without a matched start (e.g. pre-run buffer flush)
                st = {"step": key, "start_ts": None, "end_ts": ev.get("ts"),
                      "duration_s": ev.get("duration_s"), "status": ev.get("status", "ok"),
                      "source": "span" if is_span else "step"}
            else:
                st["end_ts"] = ev.get("ts")
                st["duration_s"] = ev.get("duration_s", st.get("duration_s"))
                st["status"] = ev.get("status", "ok")
            steps.append(st)
            if len(steps) > 500:
                steps = steps[-500:]

    for name, st in open_steps.items():
        steps.append(st)

    return {
        "meta": meta,
        "counters": counters,
        "custom_metrics": {k: latency_stats(v) for k, v in custom_metrics.items()},
        "logs": {"by_level": logs_by_level, "by_event": events_by_type},
        "llm": {
            "overall": _bucket_record(llm_overall),
            "by_agent": {k: _bucket_record(v) for k, v in llm_by_agent.items()},
            "by_task": {k: _bucket_record(v) for k, v in llm_by_task.items()},
            "by_model": {k: _bucket_record(v) for k, v in llm_by_model.items()},
            "calls": llm_calls,
        },
        "tools": {
            "overall": _bucket_record(tools_overall),
            "by_tool": {k: _bucket_record(v) for k, v in tools_by_name.items()},
            "calls": tool_calls,
        },
        "tasks": list(tasks.values()),
        "key_metrics": key_metrics,
        "steps": steps,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# RunRecorder: per-run JSONL writer + in-memory aggregates.
# ---------------------------------------------------------------------------

class RunRecorder:
    """
    Owns the active run: appends events to one JSONL file, keeps slim in-memory
    events for final metrics, and exposes counter/timer APIs.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._fh = None
        self.active = False
        self.run_id: Optional[str] = None
        self.log_path: Optional[str] = None
        self.logs_dir: str = LOGS_DIR
        self.output_dir: Optional[str] = None
        self.started_at: Optional[float] = None
        self.status: str = "running"
        self._events: deque = deque(maxlen=MAX_EVENTS_IN_MEMORY)
        self._prerun_buffer: deque = deque(maxlen=500)
        self._prev_prompt_tokens: Dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    def is_active(self) -> bool:
        return self.active

    def ensure_run(self, source: str = "workflow", output_dir: Optional[str] = None,
                   logs_dir: Optional[str] = None) -> str:
        """Starts a run if none is active; returns the run_id."""
        with self._lock:
            if self.active:
                return self.run_id
            return self._start_run(source=source, output_dir=output_dir, logs_dir=logs_dir)

    def _start_run(self, source: str = "workflow", output_dir: Optional[str] = None,
                   logs_dir: Optional[str] = None) -> str:
        if logs_dir:
            self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}_{os.getpid()}"
        self.log_path = os.path.join(self.logs_dir, f"{self.run_id}.jsonl")
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self.started_at = time.time()
        self.status = "running"
        self.output_dir = output_dir
        self._events.clear()
        self._prev_prompt_tokens = {}
        self.active = True
        # Flush events that were logged before the run started (agent setup etc.)
        prerun = list(self._prerun_buffer)
        self._prerun_buffer.clear()
        for ev in prerun:
            ev["run_id"] = self.run_id
            self._append(ev)
        self.log_event("run_started", run_source=source, output_dir=output_dir, pid=os.getpid())
        return self.run_id

    def finalize(self, status: str = "completed", **attrs) -> Optional[Dict[str, Any]]:
        """Writes the run summary event + run_metrics.json, then closes the file."""
        with self._lock:
            if not self.active:
                return None
            self.status = status
            snapshot = self.snapshot(status=status)
            try:
                self.log_event("run_summary", level="info", status=status, snapshot=snapshot, **attrs)
            except Exception:
                pass
            if self.output_dir:
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    with open(os.path.join(self.output_dir, "run_metrics.json"), "w", encoding="utf-8") as mf:
                        json.dump(snapshot, mf, indent=2, default=str)
                except Exception:
                    pass
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            self.active = False
            return snapshot

    # -- core event writing --------------------------------------------------

    def log_event(self, event: str, level: str = "info", **attrs) -> None:
        """Appends one JSON line to the run file. Safe to call from anywhere."""
        try:
            record: Dict[str, Any] = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "event": event,
                "level": level,
            }
            ctx = _task_ctx.get() or {}
            for key in ("task_id", "agent", "phase", "title"):
                if ctx.get(key) is not None:
                    record[key] = ctx[key]
            for k, v in attrs.items():
                record[k] = chunk_value(v)

            with self._lock:
                if self.active:
                    record["run_id"] = self.run_id
                    self._append(record)
                else:
                    self._prerun_buffer.append(record)
        except Exception:
            pass

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
            self._events.append(record)
        except Exception:
            pass

    # -- counters & timers ---------------------------------------------------

    def increment(self, key: str, amount: float = 1) -> None:
        self.log_event("counter", counter=key, amount=amount)

    def add_duration(self, key: str, seconds: float) -> None:
        self.log_event("duration", metric=key, duration_s=round(float(seconds), 6))

    def record_counter_and_duration(self, counter_key: str, duration_key: str,
                                    seconds: float) -> None:
        """Single event carrying both a counter bump and a duration sample."""
        self.log_event("counter_duration", counter=counter_key, metric=duration_key,
                       duration_s=round(float(seconds), 6))

    def step_start(self, step: str, **attrs) -> None:
        self.log_event("step_start", step=step, **attrs)

    def step_end(self, step: str, status: str = "ok", **attrs) -> None:
        self.log_event("step_end", step=step, status=status, **attrs)

    @contextmanager
    def step_timer(self, step: str, **attrs):
        """Context manager recording a named step with duration for the timeline."""
        self.step_start(step, **attrs)
        t0 = time.perf_counter()
        status = "ok"
        try:
            yield self
        except BaseException as err:
            status = f"error: {type(err).__name__}: {err}"
            raise
        finally:
            self.step_end(step, status=status, duration_s=round(time.perf_counter() - t0, 6))

    # -- structured recorders ------------------------------------------------

    def record_llm_call(self, agent_name: str, duration_s: float, success: bool,
                        response: Any = None, requested_model: Optional[str] = None,
                        messages: Any = None, error: Optional[str] = None,
                        finish_reason: Optional[str] = None,
                        tool_call_count: int = 0) -> None:
        """Records one LLM call with latency, served model, and token metrics."""
        ctx = _task_ctx.get() or {}
        served_model, provider = None, None
        usage: Dict[str, Any] = {}
        if response is not None:
            served_model = getattr(response, "model", None)
            provider = getattr(response, "provider", None)
            u = getattr(response, "usage", None)
            if u is not None:
                try:
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", None) or 0,
                        "completion_tokens": getattr(u, "completion_tokens", None) or 0,
                        "total_tokens": getattr(u, "total_tokens", None) or 0,
                    }
                    ptd = getattr(u, "prompt_tokens_details", None)
                    if ptd is not None:
                        cached = getattr(ptd, "cached_tokens", None)
                        if cached:
                            usage["cached_tokens"] = cached
                except Exception:
                    usage = {}

        # Cachable (new-only) tokens: current outgoing message tokens minus what
        # the same agent+task already had on its previous call. A shrinking count
        # means compaction happened; the baseline resets and nothing new is counted.
        prompt_total_est = _estimate_prompt_tokens(messages)
        cache_key = f"{agent_name}|{ctx.get('task_id')}|{ctx.get('phase')}"
        new_tokens = 0
        with self._lock:
            prev = self._prev_prompt_tokens.get(cache_key)
            if prev is None:
                new_tokens = prompt_total_est
            else:
                new_tokens = max(0, prompt_total_est - prev)
            self._prev_prompt_tokens[cache_key] = prompt_total_est

        self.log_event(
            "llm_call",
            level="info" if success else "error",
            agent=agent_name,
            duration_s=round(duration_s, 6),
            success=success,
            requested_model=requested_model,
            served_model=served_model,
            provider=provider,
            usage=usage,
            # Flattened for slim call lists / the dashboard's recent-calls table
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cached_tokens=usage.get("cached_tokens"),
            new_prompt_tokens=new_tokens,
            prompt_tokens_estimated=prompt_total_est,
            error=error,
            finish_reason=finish_reason,
            tool_call_count=tool_call_count,
        )

    def record_tool_call(self, fn_name: str, tool_call_id: Any, success: bool,
                         duration_s: float, code: Optional[str] = None,
                         stdout: Optional[str] = None, stderr: Optional[str] = None,
                         returned_files: Optional[List[str]] = None,
                         error: Optional[str] = None,
                         parse_error: Optional[str] = None) -> None:
        """Records one tool execution with FULL payloads (smart-chunked at 15k tokens)."""
        self.log_event(
            "tool_executed",
            level="info" if success else "error",
            fn_name=fn_name,
            tool_call_id=str(tool_call_id) if tool_call_id is not None else None,
            success=success,
            duration_s=round(float(duration_s or 0.0), 6),
            code=code,
            stdout=stdout,
            stderr=stderr,
            returned_files=returned_files or [],
            error=error,
            parse_error=parse_error,
        )

    # -- snapshot ------------------------------------------------------------

    def snapshot(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Aggregates the in-memory events of the active (or last) run."""
        agg = aggregate_events(list(self._events))
        agg["meta"]["status"] = status or self.status
        agg["meta"]["run_id"] = self.run_id
        agg["meta"]["log_file"] = os.path.basename(self.log_path) if self.log_path else None
        if self.started_at:
            agg["meta"]["elapsed_s"] = round(time.time() - self.started_at, 3)
        return agg

    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)


_recorder = RunRecorder()


def get_recorder() -> RunRecorder:
    """Returns the process-wide run recorder singleton."""
    return _recorder


# ---------------------------------------------------------------------------
# Logfire mirroring: patch module functions to ALSO write the local JSONL.
# ---------------------------------------------------------------------------

_logfire_patched = False


def _wrap_log_method(target: Any, name: str, level: str, recorder: RunRecorder) -> None:
    orig = getattr(target, name)

    def wrapper(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            msg = args[0] if args else kwargs.get("message", "")
            extra_attrs = dict(kwargs)
            if len(args) > 1:
                # logfire templating: positional args beyond the message fill {}
                # placeholders; mirror them so the rendered values are not lost.
                extra_attrs["args"] = [str(a) for a in args[1:]]
            recorder.log_event("logfire", level=level, message=str(msg) if msg is not None else "",
                               **{k: v for k, v in extra_attrs.items()
                                  if k not in ("_tags", "_span_id", "_attributes")})
        except Exception:
            pass
        return result

    setattr(target, name, wrapper)


class _MirroredSpan:
    """Wraps a logfire span context manager, mirroring start/end/duration locally."""

    def __init__(self, span_ctx: Any, name: Any, attrs: Dict[str, Any], recorder: RunRecorder):
        self._span_ctx = span_ctx
        self._name = name
        self._attrs = attrs
        self._recorder = recorder
        self._t0: Optional[float] = None

    def __enter__(self):
        entered = self._span_ctx.__enter__()
        self._t0 = time.perf_counter()
        try:
            self._recorder.log_event("span_start", span=self._name, **self._attrs)
        except Exception:
            pass
        return entered

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        try:
            self._recorder.log_event(
                "span_end",
                level="error" if exc_type else "info",
                span=self._name,
                duration_s=round(duration, 6),
                error=f"{exc_type.__name__}: {exc_val}" if exc_type else None,
                **self._attrs,
            )
        except Exception:
            pass
        return self._span_ctx.__exit__(exc_type, exc_val, exc_tb)


def _wrap_span_method(target: Any, recorder: RunRecorder) -> None:
    orig = getattr(target, "span")

    def span_wrapper(*args, **kwargs):
        name = args[0] if args else kwargs.get("span_name", "")
        attrs = {k: v for k, v in kwargs.items() if k != "span_name"}
        span_ctx = orig(*args, **kwargs)
        return _MirroredSpan(span_ctx, name, attrs, recorder)

    setattr(target, "span", span_wrapper)


def patch_logfire(target: Any = None, recorder: Optional[RunRecorder] = None) -> bool:
    """
    Wraps logfire.info/warn/error/debug/trace and logfire.span so every log and
    span is mirrored to the local run JSONL. Logfire's own behavior is unchanged.
    Idempotent. `target` is injectable for tests.
    """
    global _logfire_patched
    lf = target if target is not None else _logfire_module
    if lf is None:
        return False
    rec = recorder or _recorder
    if target is None and _logfire_patched:
        return True
    for name, level in (("info", "info"), ("warn", "warn"), ("warning", "warn"),
                        ("error", "error"), ("debug", "debug"), ("trace", "debug")):
        if hasattr(lf, name):
            try:
                _wrap_log_method(lf, name, level, rec)
            except Exception:
                pass
    if hasattr(lf, "span"):
        try:
            _wrap_span_method(lf, rec)
        except Exception:
            pass
    _logfire_patched = True
    return True


# ---------------------------------------------------------------------------
# OpenAI client instrumentation: per-call latency, served model, tokens.
# ---------------------------------------------------------------------------

def _attach_header_hook(client: Any, captured: Dict[str, Any]) -> None:
    """Best-effort capture of Portkey response headers (provider, retries, cache)."""
    try:
        httpx_client = getattr(client, "_client", None)
        if httpx_client is None:
            return

        def _on_response(response):
            try:
                headers = getattr(response, "headers", None) or {}
                captured["headers"] = {
                    "portkey_provider": headers.get("x-portkey-provider"),
                    "portkey_retry_attempt_count": headers.get("x-portkey-retry-attempt-count"),
                    "portkey_cache_status": headers.get("x-portkey-cache-status"),
                }
            except Exception:
                pass

        hooks = getattr(httpx_client, "event_hooks", None)
        if hooks is None:
            httpx_client.event_hooks = {"response": [_on_response]}
        else:
            hooks.setdefault("response", []).append(_on_response)
    except Exception:
        pass


def instrument_openai_client(client: Any, agent_name: str,
                             recorder: Optional[RunRecorder] = None) -> Any:
    """
    Wraps client.chat.completions.create to record an llm_call event per call:
    wall-time latency, served model from the response body (NOT the requested
    name), provider, token usage (total + new-only estimate), success/failure.
    Never raises.
    """
    rec = recorder or _recorder
    try:
        if getattr(client, "_obs_instrumented", False):
            return client
        completions = client.chat.completions
        if getattr(completions, "_obs_instrumented", False):
            # Shared completions object already wrapped (e.g. two agents on one client)
            client._obs_instrumented = True
            return client
        orig_create = completions.create
        captured: Dict[str, Any] = {"headers": None}
        _attach_header_hook(client, captured)

        def create(*args, **kwargs):
            start = time.perf_counter()
            messages = kwargs.get("messages")
            if messages is None and len(args) >= 2:
                messages = args[1]
            try:
                response = orig_create(*args, **kwargs)
                duration = time.perf_counter() - start
                try:
                    finish_reason = None
                    tool_call_count = 0
                    choices = getattr(response, "choices", None)
                    if choices:
                        finish_reason = getattr(choices[0], "finish_reason", None)
                        msg = getattr(choices[0], "message", None)
                        tool_call_count = len(getattr(msg, "tool_calls", None) or [])
                    rec.record_llm_call(
                        agent_name=agent_name,
                        duration_s=duration,
                        success=True,
                        response=response,
                        requested_model=kwargs.get("model") or (args[0] if args else None),
                        messages=messages,
                        finish_reason=finish_reason,
                        tool_call_count=tool_call_count,
                    )
                    hdrs = captured.get("headers") or {}
                    if hdrs:
                        rec.log_event("llm_transport", agent=agent_name, **hdrs)
                except Exception:
                    pass
                return response
            except Exception as err:
                duration = time.perf_counter() - start
                try:
                    rec.record_llm_call(
                        agent_name=agent_name,
                        duration_s=duration,
                        success=False,
                        requested_model=kwargs.get("model"),
                        messages=messages,
                        error=f"{type(err).__name__}: {err}",
                    )
                except Exception:
                    pass
                raise

        completions.create = create
        completions._obs_instrumented = True
        client._obs_instrumented = True
    except Exception:
        pass
    return client


__all__ = [
    "RunRecorder", "get_recorder", "ensure_run", "finalize_run",
    "smart_chunk_text", "chunk_value", "latency_stats", "aggregate_events",
    "set_task_context", "reset_task_context", "task_context",
    "patch_logfire", "instrument_openai_client",
    "LOGS_DIR", "TOKEN_CHUNK_LIMIT",
]


def ensure_run(source: str = "workflow", output_dir: Optional[str] = None) -> str:
    return _recorder.ensure_run(source=source, output_dir=output_dir)


def finalize_run(status: str = "completed", **attrs) -> Optional[Dict[str, Any]]:
    return _recorder.finalize(status=status, **attrs)
