"""
Tests for the unified observability module (agents/observability.py):
- smart middle-out chunking at the 15k token limit
- per-run JSONL logging (one file per run, pre-run buffer flush)
- latency percentile stats and event aggregation
- logfire mirroring patch (against a dummy target, no real Logfire calls)
- OpenAI client instrumentation (served model, token usage, new-only tokens)
"""

import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.observability import (
    RunRecorder,
    aggregate_events,
    chunk_value,
    instrument_openai_client,
    latency_stats,
    patch_logfire,
    smart_chunk_text,
    TOKEN_CHUNK_LIMIT,
)


@pytest.fixture()
def recorder(tmp_path):
    rec = RunRecorder()
    rec.ensure_run(source="test", output_dir=str(tmp_path / "out"), logs_dir=str(tmp_path / "logs"))
    yield rec
    rec.finalize(status="completed")


# ---------------------------------------------------------------------------
# Smart chunking
# ---------------------------------------------------------------------------

def test_short_text_unchanged():
    text = "print('hello world')"
    assert smart_chunk_text(text) == text


def test_long_text_middle_truncated():
    # ~40k tokens of text; only head and tail must survive.
    head_marker = "HEAD_START_MARKER"
    tail_marker = "TAIL_END_MARKER"
    text = head_marker + " word " * 30000 + tail_marker
    result = smart_chunk_text(text, token_limit=1000)
    assert result.count("[middle truncated") == 1
    assert result.startswith(head_marker)
    assert result.endswith(tail_marker)
    # Result is bounded near the token limit (plus marker overhead)
    from agents.observability import count_tokens
    assert count_tokens(result) < 1000 + 200


def test_chunk_value_recurses_into_dicts_and_lists():
    big = "x " * 30000
    value = {"code": big, "files": [{"stdout": "small"}, {"stderr": big}]}
    out = chunk_value(value, token_limit=500)
    assert "[middle truncated" in out["code"]
    assert "[middle truncated" in out["files"][1]["stderr"]
    assert out["files"][0]["stdout"] == "small"
    # input not mutated
    assert value["code"] == big


# ---------------------------------------------------------------------------
# RunRecorder JSONL logging
# ---------------------------------------------------------------------------

def test_one_file_per_run_with_all_events(recorder, tmp_path):
    logs_dir = recorder.logs_dir
    files = [f for f in os.listdir(logs_dir) if f.endswith(".jsonl")]
    assert len(files) == 1

    recorder.log_event("tool_executed", fn_name="execute_python", success=True, stdout="hi")
    recorder.log_event("llm_call", agent="DataAnalystAgent", duration_s=1.5, success=True)
    recorder.finalize(status="completed")

    path = os.path.join(logs_dir, files[0])
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]

    events = [ln["event"] for ln in lines]
    assert events[0] == "run_started"
    assert "tool_executed" in events
    assert "llm_call" in events
    assert events[-1] == "run_summary"
    assert all("run_id" in ln for ln in lines)
    # run_id is the file stem
    assert files[0][: len("run_")] == "run_"


def test_pre_run_events_are_flushed_into_the_run(tmp_path):
    rec = RunRecorder()
    # log before the run starts -> buffered, not written anywhere
    rec.log_event("logfire", level="info", message="agent setup log")
    rec.ensure_run(source="test", logs_dir=str(tmp_path / "logs"))
    path = os.path.join(rec.logs_dir, f"{rec.run_id}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        events = [json.loads(ln)["event"] for ln in f if ln.strip()]
    assert "logfire" in events
    rec.finalize(status="completed")


def test_finalize_writes_run_metrics_json(recorder, tmp_path):
    recorder.log_event("llm_call", agent="A", duration_s=2.0, success=True)
    snap = recorder.finalize(status="completed")
    metrics_path = os.path.join(tmp_path / "out", "run_metrics.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["llm"]["overall"]["latency"]["count"] == 1
    assert snap["meta"]["status"] == "completed"


def test_huge_payloads_are_chunked_in_the_file(recorder):
    recorder.log_event("tool_executed", fn_name="execute_python", success=True,
                       stdout="data " * 30000)
    path = os.path.join(recorder.logs_dir, f"{recorder.run_id}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    ev = [ln for ln in lines if ln["event"] == "tool_executed"][0]
    assert "[middle truncated" in ev["stdout"]
    assert len(ev["stdout"]) < TOKEN_CHUNK_LIMIT * 6  # chars ~= tokens*4 + marker


# ---------------------------------------------------------------------------
# Latency stats & aggregation
# ---------------------------------------------------------------------------

def test_latency_stats_percentiles():
    durations = [float(i) for i in range(1, 101)]  # 1..100
    s = latency_stats(durations)
    assert s["count"] == 100
    assert s["min_s"] == 1.0
    assert s["max_s"] == 100.0
    assert 50.0 <= s["p50_s"] <= 51.0
    assert 99.0 <= s["p99_s"] <= 100.0
    assert abs(s["avg_s"] - 50.5) < 0.01


def test_aggregate_events_llm_tools_tasks_steps():
    events = [
        {"ts": "t0", "event": "run_started", "run_id": "r1", "run_source": "test"},
        {"ts": "t1", "event": "step_start", "step": "plan_creation"},
        {"ts": "t2", "event": "step_end", "step": "plan_creation", "duration_s": 1.0, "status": "ok"},
        {"ts": "t3", "event": "span_start", "span": "Task #1: clean"},
        {"ts": "t4", "event": "span_end", "span": "Task #1: clean", "duration_s": 5.0},
        {"ts": "t5", "event": "llm_call", "agent": "DataAnalystAgent", "task_id": 1,
         "duration_s": 2.0, "success": True, "served_model": "m1", "provider": "openrouter",
         "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                   "prompt_tokens_details": {"cached_tokens": 10}},
         "new_prompt_tokens": 80},
        {"ts": "t6", "event": "llm_call", "agent": "MLAgent", "task_id": 2,
         "duration_s": 3.0, "success": False, "served_model": "m2",
         "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
         "new_prompt_tokens": 10, "error": "boom"},
        {"ts": "t7", "event": "tool_executed", "fn_name": "execute_python", "success": True,
         "duration_s": 4.0, "stdout": "ok", "returned_files": ["a.csv"]},
        {"ts": "t8", "event": "tool_executed", "fn_name": "execute_python", "success": False,
         "duration_s": 1.0, "stderr": "trace"},
        {"ts": "t9", "event": "task_retry", "task_id": 2, "title": "ml", "agent": "MLAgent", "attempt": 1},
        {"ts": "t10", "event": "task_completed", "task_id": 2, "title": "ml", "agent": "MLAgent",
         "status": "completed", "duration_s": 6.0, "retries": 1},
        {"ts": "t11", "event": "counter", "counter": "retries.validation", "amount": 1},
        {"ts": "t12", "event": "duration", "metric": "plan_creation_time_s", "duration_s": 1.2},
    ]
    agg = aggregate_events(events)

    assert agg["meta"]["run_id"] == "r1"
    assert agg["counters"]["retries.validation"] == 1
    assert agg["custom_metrics"]["plan_creation_time_s"]["count"] == 1

    llm = agg["llm"]["overall"]
    assert llm["calls"] == 2
    assert llm["success"] == 1 and llm["failed"] == 1
    assert abs(llm["failure_rate"] - 0.5) < 1e-6
    assert llm["latency"]["total_s"] == 5.0
    assert llm["tokens"]["total"] == 160
    assert llm["tokens"]["new"] == 90
    assert llm["tokens"]["cached"] == 10

    assert agg["llm"]["by_agent"]["DataAnalystAgent"]["calls"] == 1
    assert agg["llm"]["by_agent"]["MLAgent"]["failed"] == 1
    assert agg["llm"]["by_task"]["1"]["calls"] == 1
    assert set(agg["llm"]["by_model"].keys()) == {"m1", "m2"}

    tools = agg["tools"]["overall"]
    assert tools["calls"] == 2
    assert tools["success"] == 1 and tools["failed"] == 1
    assert agg["tools"]["by_tool"]["execute_python"]["latency"]["total_s"] == 5.0

    tasks = {t["task_id"]: t for t in agg["tasks"]}
    assert tasks[2]["status"] == "completed"
    assert tasks[2]["retries"] == 1
    assert tasks[2]["duration_s"] == 6.0

    step_names = {s["step"] for s in agg["steps"]}
    assert "plan_creation" in step_names
    assert "Task #1: clean" in step_names  # span folded into timeline


# ---------------------------------------------------------------------------
# Logfire mirroring patch (dummy target, no real Logfire)
# ---------------------------------------------------------------------------

def test_patch_logfire_mirrors_logs_and_spans(tmp_path):
    rec = RunRecorder()
    rec.ensure_run(source="test", logs_dir=str(tmp_path / "logs"))

    dummy = types.SimpleNamespace()
    calls = []

    def fake_info(msg, **attrs):
        calls.append(("info", msg, attrs))
        return "logged"

    def fake_span(name, **attrs):
        return _FakeSpanCtx(name, attrs)

    class _FakeSpanCtx:
        def __init__(self, name, attrs):
            self.name, self.attrs = name, attrs

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            return False

    dummy.info = fake_info
    dummy.warn = fake_info
    dummy.error = fake_info
    dummy.span = fake_span

    assert patch_logfire(target=dummy, recorder=rec) is True

    dummy.info("hello world", key="value")
    assert calls and calls[0][1] == "hello world"  # original still called

    with dummy.span("MySpan", n=1):
        rec.log_event("inside_span")

    rec.finalize(status="completed")
    path = os.path.join(rec.logs_dir, f"{rec.run_id}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        events = [json.loads(ln) for ln in f if ln.strip()]
    names = [e["event"] for e in events]

    assert "logfire" in names
    mirrored = [e for e in events if e["event"] == "logfire"][0]
    assert mirrored["message"] == "hello world"
    assert mirrored["key"] == "value"

    assert "span_start" in names and "span_end" in names
    span_end = [e for e in events if e["event"] == "span_end"][0]
    assert span_end["span"] == "MySpan"
    assert span_end["duration_s"] >= 0


# ---------------------------------------------------------------------------
# OpenAI client instrumentation
# ---------------------------------------------------------------------------

def _fake_response(model="inclusionai/ling-3.0-flash-fin:free", provider="openrouter",
                   prompt=100, completion=50, cached=5):
    return SimpleNamespace(
        model=model,
        provider=provider,
        choices=[SimpleNamespace(finish_reason="stop",
                                 message=SimpleNamespace(tool_calls=None, content="hi"))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                resp = outer.responses.pop(0)
                if isinstance(resp, Exception):
                    raise resp
                return resp

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_instrument_openai_client_records_served_model_and_tokens():
    import tempfile
    rec = RunRecorder()
    rec.ensure_run(source="test", logs_dir=tempfile.mkdtemp())

    client = _FakeClient([_fake_response(), _fake_response(prompt=260, completion=80)])
    instrument_openai_client(client, agent_name="PlannerAgent", recorder=rec)

    msgs1 = [{"role": "system", "content": "sys " * 50}, {"role": "user", "content": "q " * 50}]
    client.chat.completions.create(model="requested-model", messages=msgs1)
    msgs2 = msgs1 + [{"role": "assistant", "content": "a " * 50}]
    client.chat.completions.create(model="requested-model", messages=msgs2)

    events = [e for e in rec.events() if e["event"] == "llm_call"]
    assert len(events) == 2

    first, second = events
    assert first["success"] is True
    assert first["requested_model"] == "requested-model"
    assert first["served_model"] == "inclusionai/ling-3.0-flash-fin:free"
    assert first["provider"] == "openrouter"
    assert first["usage"]["prompt_tokens"] == 100
    assert first["usage"]["completion_tokens"] == 50
    assert first["usage"]["cached_tokens"] == 5
    assert first["new_prompt_tokens"] > 0
    # Second call re-sends the same history plus a short new message:
    # new-only tokens must be much smaller than the full prompt estimate.
    assert second["new_prompt_tokens"] < first["new_prompt_tokens"]
    assert second["duration_s"] >= 0


def test_instrument_openai_client_records_failures():
    import tempfile
    rec = RunRecorder()
    rec.ensure_run(source="test", logs_dir=tempfile.mkdtemp())
    client = _FakeClient([RuntimeError("gateway down")])
    instrument_openai_client(client, agent_name="MLAgent", recorder=rec)

    with pytest.raises(RuntimeError):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])

    events = [e for e in rec.events() if e["event"] == "llm_call"]
    assert len(events) == 1
    assert events[0]["success"] is False
    assert "gateway down" in events[0]["error"]
    assert events[0]["level"] == "error"


def test_aggregation_reflected_in_snapshot(recorder):
    recorder.log_event("llm_call", agent="A", duration_s=1.0, success=True,
                       usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                       new_prompt_tokens=15)
    snap = recorder.snapshot()
    assert snap["llm"]["overall"]["calls"] == 1
    assert snap["llm"]["overall"]["tokens"]["total"] == 15
    assert snap["llm"]["overall"]["tokens"]["new"] == 15
