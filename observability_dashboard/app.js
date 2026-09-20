/* AnalystGPT Observability Dashboard - vanilla JS, no external deps. */
"use strict";

const API = "";  // same origin
let currentRunId = null;
let currentSummary = null;
let liveTimer = null;
let eventsOffset = 0;

// ---------- helpers ----------

const $ = (id) => document.getElementById(id);

function fmtSecs(s) {
  if (s === null || s === undefined) return "—";
  const n = Number(s);
  if (!isFinite(n)) return "—";
  if (n >= 60) return `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`;
  if (n >= 10) return `${n.toFixed(1)}s`;
  if (n >= 0.001) return `${(n * 1000).toFixed(0)}ms`;
  return `${(n * 1000).toFixed(2)}ms`;
}

function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString();
}

function fmtTs(ts) {
  if (!ts) return "—";
  return String(ts).replace("T", " ").slice(0, 23);
}

function pct(n) {
  if (n === null || n === undefined) return "—";
  return `${(Number(n) * 100).toFixed(1)}%`;
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function latencyCells(stats) {
  if (!stats || !stats.count) return `<td class="num">0</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td>`;
  return `<td class="num">${stats.count}</td>
      <td class="num">${fmtSecs(stats.avg_s)}</td>
      <td class="num">${fmtSecs(stats.p50_s)}</td>
      <td class="num">${fmtSecs(stats.p90_s)}</td>
      <td class="num">${fmtSecs(stats.p95_s)}</td>
      <td class="num">${fmtSecs(stats.p99_s)}</td>`;
}

const LATENCY_THS = `<tr><th>Name</th><th class="num">Calls</th><th class="num">Avg</th><th class="num">P50</th><th class="num">P90</th><th class="num">P95</th><th class="num">P99</th></tr>`;

// ---------- run list ----------

async function loadRuns() {
  const res = await fetch(`${API}/api/observability/runs`);
  const data = await res.json();
  const sel = $("run-select");
  const prev = currentRunId;
  sel.innerHTML = "";
  if (!data.runs || data.runs.length === 0) {
    sel.innerHTML = `<option value="">No runs found in logs/</option>`;
    return;
  }
  for (const run of data.runs) {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    const flag = run.run_id === data.active_run_id ? "● " : "";
    opt.textContent = `${flag}${run.run_id} · ${(run.size_bytes / 1024).toFixed(0)}KB · ${run.status}`;
    sel.appendChild(opt);
  }
  if (prev && [...sel.options].some(o => o.value === prev)) {
    sel.value = prev;
  } else if (data.active_run_id) {
    sel.value = data.active_run_id;
  }
  if (sel.value !== prev || !currentSummary) {
    currentRunId = sel.value;
    loadRun(currentRunId);
    loadEvents(true);
  }
}

// ---------- run summary ----------

async function loadRun(runId) {
  if (!runId) return;
  try {
    const res = await fetch(`${API}/api/observability/runs/${runId}`);
    if (!res.ok) { renderEmpty(`Run ${runId} not found`); return; }
    currentSummary = await res.json();
    renderAll(currentSummary);
  } catch (err) {
    renderEmpty(`Failed to load run: ${err}`);
  }
}

function renderEmpty(msg) {
  $("kpi-grid").innerHTML = `<div class="empty">${esc(msg)}</div>`;
}

function renderAll(s) {
  renderStatus(s);
  renderKpis(s);
  renderTimeline(s);
  renderLlmTables(s);
  renderTokens(s);
  renderTools(s);
  renderTasks(s);
  renderToolCalls(s);
  renderLlmCalls(s);
  renderErrors(s);
}

function renderStatus(s) {
  const strip = $("status-strip");
  strip.classList.remove("hidden");
  const status = s.meta?.status || "unknown";
  const badge = $("run-status-badge");
  badge.textContent = status;
  badge.className = `badge ${status}`;
  const meta = $("run-meta");
  meta.textContent = [
    `run_id: ${s.meta?.run_id || "—"}`,
    `started: ${fmtTs(s.meta?.started_at)}`,
    s.meta?.ended_at ? `ended: ${fmtTs(s.meta.ended_at)}` : null,
    s.meta?.elapsed_s != null ? `elapsed: ${fmtSecs(s.meta.elapsed_s)}` : null,
    `events: ${s.event_count ?? "—"}`,
    s.meta?.output_dir ? `out: ${s.meta.output_dir}` : null,
  ].filter(Boolean).join("   ·   ");
}

function renderKpis(s) {
  const llm = s.llm?.overall || {};
  const tools = s.tools?.overall || {};
  const steps = s.steps || [];
  const wf = steps.find(x => x.step === "total_workflow");
  const plan = s.custom_metrics?.plan_creation_time_s;
  const counters = s.counters || {};
  const tasks = s.tasks || [];
  const tok = llm.tokens || {};

  const failedTasks = tasks.filter(t => t.status === "failed" || t.status === "validation_failed").length;
  const completedTasks = tasks.filter(t => t.status === "completed").length;

  const kpis = [
    { label: "Workflow latency", value: fmtSecs(wf?.duration_s ?? s.meta?.elapsed_s), sub: wf ? "step total" : "run elapsed", cls: "" },
    { label: "Plan creation", value: fmtSecs(plan?.total_s), sub: plan ? `${plan.count} plan call(s), avg ${fmtSecs(plan.avg_s)}` : "—", cls: "" },
    { label: "LLM total latency", value: fmtSecs(llm.latency?.total_s), sub: `${fmtNum(llm.latency?.count || 0)} calls · avg ${fmtSecs(llm.latency?.avg_s)} · p99 ${fmtSecs(llm.latency?.p99_s)}`, cls: "" },
    { label: "LLM failure rate", value: pct(llm.failure_rate), sub: `${fmtNum(llm.failed || 0)} failed / ${fmtNum(llm.calls || 0)}`, cls: llm.failure_rate > 0 ? "bad" : "good" },
    { label: "Tool exec time (total)", value: fmtSecs(tools.latency?.total_s), sub: `${fmtNum(tools.latency?.count || 0)} calls · avg ${fmtSecs(tools.latency?.avg_s)} · p99 ${fmtSecs(tools.latency?.p99_s)}`, cls: "" },
    { label: "Tool failure rate", value: pct(tools.failure_rate), sub: `${fmtNum(tools.failed || 0)} failed / ${fmtNum(tools.calls || 0)}`, cls: tools.failure_rate > 0 ? "bad" : "good" },
    { label: "Retries", value: fmtNum((counters["retries.validation"] || 0)), sub: `retry latency avg ${fmtSecs(s.custom_metrics?.["retries.latency_s"]?.avg_s)} · ${fmtNum(s.custom_metrics?.["retries.latency_s"]?.count || 0)} retry attempt(s)`, cls: (counters["retries.validation"] || 0) > 0 ? "warn" : "" },
    { label: "Tasks", value: `${completedTasks}/${tasks.length}`, sub: failedTasks ? `${failedTasks} failed` : "all passed", cls: failedTasks ? "bad" : "good" },
    { label: "Tokens (total)", value: fmtNum(tok.total), sub: `prompt ${fmtNum(tok.prompt)} · completion ${fmtNum(tok.completion)}`, cls: "" },
    { label: "Tokens (new-only)", value: fmtNum(tok.new), sub: `provider-cached ${fmtNum(tok.cached)}`, cls: "" },
    { label: "Repair attempts", value: fmtNum((counters["repair.attempts"] || 0)), sub: "post-report repair loop", cls: "" },
    { label: "Warnings / Errors", value: `${(s.logs?.by_level?.warn || 0) + (s.logs?.by_level?.warning || 0)} / ${s.logs?.by_level?.error || 0}`, sub: `total events: ${Object.values(s.logs?.by_event || {}).reduce((a, b) => a + b, 0)}`, cls: s.logs?.by_level?.error ? "bad" : "" },
  ];

  $("kpi-grid").innerHTML = kpis.map(k => `
    <div class="kpi">
      <div class="label">${esc(k.label)}</div>
      <div class="value ${k.cls}">${esc(k.value)}</div>
      <div class="sub">${esc(k.sub)}</div>
    </div>`).join("");
}

function renderTimeline(s) {
  const includeSpans = $("show-spans").checked;
  const steps = (s.steps || []).filter(x => includeSpans || x.source !== "span");
  const el = $("timeline");
  if (!steps.length) { el.innerHTML = `<div class="empty">No steps recorded.</div>`; return; }

  const t0s = steps.map(x => x.start_ts).filter(Boolean);
  const t1s = steps.map(x => x.end_ts || x.start_ts).filter(Boolean);
  if (!t0s.length) { el.innerHTML = `<div class="empty">No timing data.</div>`; return; }
  const min = Math.min(...t0s.map(t => new Date(t).getTime()));
  const max = Math.max(...t1s.map(t => new Date(t).getTime()));
  const span = Math.max(1, max - min);

  const sorted = [...steps].sort((a, b) =>
    (a.start_ts ? new Date(a.start_ts).getTime() : 0) - (b.start_ts ? new Date(b.start_ts).getTime() : 0));

  el.innerHTML = sorted.slice(0, 300).map(x => {
    const start = x.start_ts ? new Date(x.start_ts).getTime() : min;
    const end = x.end_ts ? new Date(x.end_ts).getTime() : (x.duration_s ? start + x.duration_s * 1000 : max);
    const left = ((start - min) / span) * 100;
    const width = Math.max(0.15, ((end - start) / span) * 100);
    const cls = (x.status && String(x.status).startsWith("error")) ? "error" : (x.source || "step");
    const dur = x.duration_s != null ? fmtSecs(x.duration_s) : (x.status === "running" ? "running…" : "—");
    return `<div class="tl-row">
      <div class="tl-label" title="${esc(x.step)}">${esc(x.step)}<span class="src">${esc(x.source || "step")}</span></div>
      <div class="tl-track"><div class="tl-bar ${cls}" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%" title="${esc(x.step)}: ${esc(dur)}"></div></div>
      <div class="tl-dur">${esc(dur)}</div>
    </div>`;
  }).join("");
}

function bucketTable(buckets, nameKey) {
  const entries = Object.entries(buckets || {});
  if (!entries.length) return `<div class="empty">No data.</div>`;
  const rows = entries.sort((a, b) => (b[1].latency?.total_s || 0) - (a[1].latency?.total_s || 0));
  return `<table>
    <thead>${LATENCY_THS.replace("<th>Name</th>", `<th>${nameKey}</th>`)}</thead>
    <tbody>
      ${rows.map(([name, b]) => `<tr>
        <td>${esc(name)}</td>${latencyCells(b.latency)}
      </tr>`).join("")}
    </tbody>
  </table>`;
}

function renderLlmTables(s) {
  $("llm-by-agent").innerHTML = bucketTable(s.llm?.by_agent, "Agent");
  $("llm-by-task").innerHTML = bucketTable(s.llm?.by_task, "Task");
  const modelEntries = Object.entries(s.llm?.by_model || {});
  if (!modelEntries.length) {
    $("llm-by-model").innerHTML = `<div class="empty">No model data.</div>`;
  } else {
    $("llm-by-model").innerHTML = `<table>
      <thead><tr><th>Served model</th><th>Provider</th><th class="num">Calls</th><th class="num">Avg</th><th class="num">P50</th><th class="num">P99</th><th class="num">Total tokens</th></tr></thead>
      <tbody>
        ${modelEntries.sort((a, b) => b[1].calls - a[1].calls).map(([model, b]) => `<tr>
          <td>${esc(model)}</td>
          <td>${esc(b.provider || "—")}</td>
          <td class="num">${fmtNum(b.calls)}</td>
          <td class="num">${fmtSecs(b.latency?.avg_s)}</td>
          <td class="num">${fmtSecs(b.latency?.p50_s)}</td>
          <td class="num">${fmtSecs(b.latency?.p99_s)}</td>
          <td class="num">${fmtNum(b.tokens?.total)}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
  }
}

function renderTokens(s) {
  const perAgent = Object.entries(s.llm?.by_agent || {});
  if (!perAgent.length) { $("tokens").innerHTML = `<div class="empty">No token data.</div>`; return; }
  $("tokens").innerHTML = `<table>
    <thead><tr><th>Agent</th><th class="num">Total</th><th class="num">Prompt</th><th class="num">Completion</th><th class="num">New-only (cachable)</th><th class="num">Provider-cached</th></tr></thead>
    <tbody>
      ${perAgent.map(([agent, b]) => `<tr>
        <td>${esc(agent)}</td>
        <td class="num">${fmtNum(b.tokens?.total)}</td>
        <td class="num">${fmtNum(b.tokens?.prompt)}</td>
        <td class="num">${fmtNum(b.tokens?.completion)}</td>
        <td class="num">${fmtNum(b.tokens?.new)}</td>
        <td class="num">${fmtNum(b.tokens?.cached)}</td>
      </tr>`).join("")}
    </tbody>
  </table>
  <h3 class="subhead">By task / phase</h3>
  <table>
    <thead><tr><th>Task</th><th class="num">Total</th><th class="num">New-only</th><th class="num">Provider-cached</th><th class="num">Calls</th></tr></thead>
    <tbody>
      ${Object.entries(s.llm?.by_task || {}).map(([task, b]) => `<tr>
        <td>${esc(task)}</td>
        <td class="num">${fmtNum(b.tokens?.total)}</td>
        <td class="num">${fmtNum(b.tokens?.new)}</td>
        <td class="num">${fmtNum(b.tokens?.cached)}</td>
        <td class="num">${fmtNum(b.calls)}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;
}

function renderTools(s) {
  const overall = s.tools?.overall || {};
  const rows = overall.latency?.count ? `
    <table>
      <thead>${LATENCY_THS.replace("<th>Name</th>", "<th>Overall</th>")}</thead>
      <tbody><tr><td>all tools</td>${latencyCells(overall.latency)}</tr></tbody>
    </table>` : `<div class="empty">No tool executions.</div>`;
  $("tools-summary").innerHTML = rows;

  const byTool = Object.entries(s.tools?.by_tool || {});
  if (!byTool.length) { $("tools-by-tool").innerHTML = `<div class="empty">No per-tool data.</div>`; return; }
  $("tools-by-tool").innerHTML = `<table>
    <thead><tr><th>Tool</th><th class="num">Calls</th><th class="num">Success</th><th class="num">Failed</th><th class="num">Failure rate</th><th class="num">Avg</th><th class="num">P50</th><th class="num">P95</th><th class="num">P99</th><th class="num">Total</th></tr></thead>
    <tbody>
      ${byTool.sort((a, b) => (b[1].latency?.total_s || 0) - (a[1].latency?.total_s || 0)).map(([fn, b]) => `
        <tr>
          <td>${esc(fn)}</td>
          <td class="num">${fmtNum(b.calls)}</td>
          <td class="num">${fmtNum(b.success)}</td>
          <td class="num">${fmtNum(b.failed)}</td>
          <td class="num">${pct(b.failure_rate)}</td>
          <td class="num">${fmtSecs(b.latency?.avg_s)}</td>
          <td class="num">${fmtSecs(b.latency?.p50_s)}</td>
          <td class="num">${fmtSecs(b.latency?.p95_s)}</td>
          <td class="num">${fmtSecs(b.latency?.p99_s)}</td>
          <td class="num">${fmtSecs(b.latency?.total_s)}</td>
        </tr>`).join("")}
    </tbody>
  </table>`;
}

function renderTasks(s) {
  const tasks = s.tasks || [];
  if (!tasks.length) { $("tasks-table").innerHTML = `<div class="empty">No tasks recorded.</div>`; return; }
  $("tasks-table").innerHTML = `<table>
    <thead><tr><th>#</th><th>Title</th><th>Agent</th><th>Status</th><th class="num">Duration</th><th class="num">Retries</th></tr></thead>
    <tbody>
      ${tasks.map(t => `<tr>
        <td>${esc(t.task_id ?? "—")}</td>
        <td>${esc(t.title || "—")}</td>
        <td>${esc(t.agent || "—")}</td>
        <td><span class="pill ${t.status === "completed" ? "ok" : (String(t.status).includes("fail") ? "bad" : "warn")}">${esc(t.status || "—")}</span></td>
        <td class="num">${fmtSecs(t.duration_s)}</td>
        <td class="num">${fmtNum(t.retries || 0)}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;
}

function renderToolCalls(s) {
  const calls = (s.tools?.calls || []).slice().reverse();
  if (!calls.length) { $("tool-calls").innerHTML = `<div class="empty">No tool executions recorded.</div>`; return; }
  $("tool-calls").innerHTML = calls.map((c, i) => `
    <div class="tool-call" data-idx="${i}">
      <div class="tool-call-head">
        <span class="fn">${esc(c.fn_name || "—")}</span>
        <span class="muted">${esc(c.error || (c.returned_files || []).join(", ") || "")}</span>
        <span class="num">${fmtSecs(c.duration_s)}</span>
        <span><span class="pill ${c.success ? "ok" : "bad"}">${c.success ? "OK" : "FAILED"}</span></span>
      </div>
      <div class="tool-call-body">
        ${c.task_id != null || c.agent ? `<h4>Context</h4><div class="muted">agent: ${esc(c.agent || "—")} · task: ${esc(c.task_id ?? "—")} · phase: ${esc(c.phase || "—")} · ts: ${esc(c.ts || "—")}</div>` : ""}
        ${c.parse_error ? `<h4>Parse error</h4><pre>${esc(c.parse_error)}</pre>` : ""}
        ${c.code ? `<h4>Code</h4><pre>${esc(c.code)}</pre>` : ""}
        ${c.stdout ? `<h4>Stdout</h4><pre>${esc(c.stdout)}</pre>` : ""}
        ${c.stderr ? `<h4>Stderr</h4><pre>${esc(c.stderr)}</pre>` : ""}
        ${(c.returned_files || []).length ? `<h4>Returned files</h4><div>${c.returned_files.map(f => `<span class="pill">${esc(f)}</span>`).join(" ")}</div>` : ""}
      </div>
    </div>`).join("");

  $("tool-calls").querySelectorAll(".tool-call-head").forEach(head => {
    head.addEventListener("click", () => head.parentElement.classList.toggle("open"));
  });
}

function renderLlmCalls(s) {
  const calls = (s.llm?.calls || []).slice().reverse();
  if (!calls.length) { $("llm-calls").innerHTML = `<div class="empty">No LLM calls recorded.</div>`; return; }
  $("llm-calls").innerHTML = `<table>
    <thead><tr><th>Time</th><th>Agent</th><th>Task</th><th>Served model</th><th class="num">Latency</th><th class="num">Prompt</th><th class="num">New</th><th class="num">Completion</th><th class="num">Total</th><th>OK</th><th class="num">Tool calls</th></tr></thead>
    <tbody>
      ${calls.slice(0, 150).map(c => `<tr>
        <td>${esc(fmtTs(c.ts))}</td>
        <td>${esc(c.agent || "—")}</td>
        <td>${esc(c.task_id ?? c.phase ?? "—")}</td>
        <td>${esc(c.served_model || c.requested_model || "—")}</td>
        <td class="num">${fmtSecs(c.duration_s)}</td>
        <td class="num">${fmtNum(c.prompt_tokens)}</td>
        <td class="num">${fmtNum(c.new_prompt_tokens)}</td>
        <td class="num">${fmtNum(c.completion_tokens)}</td>
        <td class="num">${fmtNum(c.total_tokens)}</td>
        <td><span class="pill ${c.success ? "ok" : "bad"}">${c.success ? "✓" : "✗"}</span></td>
        <td class="num">${fmtNum(c.tool_call_count || 0)}</td>
      </tr>`).join("")}
    </tbody>
  </table>`;
}

function renderErrors(s) {
  const errors = (s.errors || []).filter(e => e.level === "error").slice(-50).reverse();
  if (!errors.length) { $("errors").innerHTML = `<div class="empty">No errors recorded 🎉</div>`; return; }
  $("errors").innerHTML = errors.map(e => `
    <div class="event-row">
      <div class="event-row-head">
        <span class="event-ts">${esc(fmtTs(e.ts))}</span>
        <span class="event-name">${esc(e.event)}</span>
        <span class="event-ctx">${esc(e.agent ? `${e.agent} · ` : "")}${esc(e.task_id != null ? `task ${e.task_id} · ` : "")}${esc(e.message || e.error || e.span || "")}</span>
      </div>
    </div>`).join("");
}

// ---------- events viewer ----------

async function loadEvents(reset = false) {
  if (!currentRunId) return;
  if (reset) eventsOffset = 0;
  const params = new URLSearchParams({
    offset: String(eventsOffset),
    limit: "200",
  });
  const level = $("filter-level").value;
  const evName = $("filter-event").value.trim();
  const q = $("filter-q").value.trim();
  if (level) params.set("level", level);
  if (evName) params.set("event", evName);
  if (q) params.set("q", q);

  try {
    const res = await fetch(`${API}/api/observability/runs/${currentRunId}/events?${params}`);
    if (!res.ok) return;
    const data = await res.json();
    const el = $("events");
    if (reset) el.innerHTML = "";
    if (!data.events.length && eventsOffset === 0) {
      el.innerHTML = `<div class="empty">No events match the filters.</div>`;
    } else {
      el.insertAdjacentHTML("beforeend", data.events.map(eventRowHtml).join(""));
      el.querySelectorAll(".event-row-head").forEach(head => {
        if (!head.dataset.bound) {
          head.dataset.bound = "1";
          head.addEventListener("click", () => head.parentElement.classList.toggle("open"));
        }
      });
    }
    eventsOffset += data.events.length;
    $("events-more").style.display = eventsOffset < data.total ? "" : "none";
  } catch (err) {
    console.error("events load failed", err);
  }
}

function eventRowHtml(ev) {
  const ctxBits = [ev.agent, ev.task_id != null ? `task ${ev.task_id}` : null, ev.phase]
    .filter(Boolean).join(" · ");
  let preview = ev.message || ev.span || ev.step || ev.counter || ev.metric || "";
  if (ev.event === "llm_call") preview = `${ev.served_model || ev.requested_model || ""} ${ev.duration_s ?? ""}s`;
  if (ev.event === "tool_executed") preview = `${ev.fn_name || ""} ${ev.duration_s ?? ""}s ${ev.success ? "OK" : "FAILED"}`;
  if (ev.event === "step_end" || ev.event === "span_end") preview = `${ev.step || ev.span || ""} ${ev.duration_s ?? ""}s`;
  return `<div class="event-row">
    <div class="event-row-head">
      <span class="event-ts">${esc(fmtTs(ev.ts))}</span>
      <span class="pill ${ev.level === "error" ? "bad" : ev.level === "warn" ? "warn" : ""}">${esc(ev.level || "")}</span>
      <span class="event-name">${esc(ev.event)}</span>
      <span class="event-ctx">${esc(ctxBits)}</span>
      <span class="event-ctx">${esc(String(preview).slice(0, 140))}</span>
    </div>
    <div class="event-body"><pre>${esc(JSON.stringify(ev, null, 2))}</pre></div>
  </div>`;
}

// ---------- live polling ----------

function setLive(on) {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  if (on) {
    liveTimer = setInterval(async () => {
      await loadRuns();  // refreshes selector; reloads current run data
      loadEvents(false).then(() => {});
    }, 3000);
  }
}

// ---------- boot ----------

$("run-select").addEventListener("change", (e) => {
  currentRunId = e.target.value;
  currentSummary = null;
  loadRun(currentRunId);
  loadEvents(true);
});
$("refresh-btn").addEventListener("click", async () => {
  await loadRuns();
  loadRun(currentRunId);
  loadEvents(true);
});
$("live-toggle").addEventListener("change", (e) => setLive(e.target.checked));
$("show-spans").addEventListener("change", () => { if (currentSummary) renderTimeline(currentSummary); });
$("filter-apply").addEventListener("click", () => loadEvents(true));
$("events-more").addEventListener("click", () => loadEvents(false));

loadRuns();
setLive($("live-toggle").checked);
