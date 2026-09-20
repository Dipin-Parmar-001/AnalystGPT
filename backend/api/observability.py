"""
Observability API - AnalystGPT.

Serves the unified per-run JSONL logs (logs/run_*.jsonl) to the standalone
observability dashboard:
- GET /api/observability/runs                  -> list of runs
- GET /api/observability/runs/{run_id}         -> aggregated metrics for one run
- GET /api/observability/runs/{run_id}/events  -> paginated raw events (live tail)
- GET /api/observability/active                -> the in-memory active run, if any
"""

import os
import json
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query

from agents.observability import (
    LOGS_DIR,
    aggregate_events,
    get_recorder,
)

router = APIRouter(prefix="/api/observability", tags=["Observability"])


def _read_last_line(path: str, max_bytes: int = 65536) -> Optional[Dict[str, Any]]:
    """Reads the last JSON line of a file without loading the whole file."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return None
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read()
        lines = [ln for ln in data.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        for ln in reversed(lines):
            try:
                return json.loads(ln)
            except Exception:
                continue
    except Exception:
        return None
    return None


def _read_first_line(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        return json.loads(line)
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _iter_run_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


@router.get("/runs")
async def list_runs() -> Dict[str, Any]:
    """Lists every run JSONL file, newest first, plus the in-memory active run."""
    runs: List[Dict[str, Any]] = []
    if os.path.isdir(LOGS_DIR):
        for fname in sorted(os.listdir(LOGS_DIR), reverse=True):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(LOGS_DIR, fname)
            try:
                stat = os.stat(path)
            except Exception:
                continue
            last = _read_last_line(path) or {}
            first = _read_first_line(path) or {}
            runs.append({
                "run_id": (first.get("run_id") or fname[:-6]),
                "file": fname,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
                "started_at": first.get("ts"),
                "last_event": last.get("event"),
                "last_ts": last.get("ts"),
                "status": last.get("status") or ("running" if last.get("event") != "run_summary" else "completed"),
            })
    active = get_recorder()
    active_id = active.run_id if active.is_active() else None
    return {"active_run_id": active_id, "logs_dir": LOGS_DIR, "runs": runs}


@router.get("/active")
async def active_run() -> Dict[str, Any]:
    """Aggregated view of the in-memory active run (fast live polling)."""
    rec = get_recorder()
    if not rec.is_active():
        return {"active": False, "run_id": None}
    return {"active": True, "run_id": rec.run_id, "summary": rec.snapshot()}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    """Aggregated metrics for one run, reduced from its JSONL file."""
    if run_id != os.path.basename(run_id) or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run id")
    path = os.path.join(LOGS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    events = _iter_run_events(path)
    summary = aggregate_events(events)
    summary["meta"]["log_file"] = f"{run_id}.jsonl"
    summary["event_count"] = len(events)
    return summary


@router.get("/runs/{run_id}/events")
async def get_run_events(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    level: Optional[str] = Query(None),
    event: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Paginated raw events for one run, with level/event/text filters."""
    if run_id != os.path.basename(run_id) or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run id")
    path = os.path.join(LOGS_DIR, f"{run_id}.jsonl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    events = _iter_run_events(path)

    def _matches(ev: Dict[str, Any]) -> bool:
        if level and ev.get("level") != level:
            return False
        if event and ev.get("event") != event:
            return False
        if q:
            try:
                haystack = json.dumps(ev, ensure_ascii=False, default=str).lower()
                if q.lower() not in haystack:
                    return False
            except Exception:
                return False
        return True

    filtered = [ev for ev in events if _matches(ev)]
    total = len(filtered)
    page = filtered[offset: offset + limit]
    return {
        "run_id": run_id,
        "total": total,
        "total_unfiltered": len(events),
        "offset": offset,
        "limit": limit,
        "events": page,
    }
