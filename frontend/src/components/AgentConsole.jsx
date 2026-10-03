import React, { useState, useEffect, useRef } from 'react';
import {
  Terminal, Bot, ChevronDown, ChevronRight, CheckCircle2, Code2, ShieldCheck, ShieldAlert,
  Loader2, Play, Wrench, XCircle, AlertTriangle, Filter, ArrowDownToLine
} from 'lucide-react';

const AGENT_COLORS = {
  planner: { dot: 'bg-agent-planner', text: 'text-agent-planner', chip: 'bg-agent-planner/10 text-agent-planner border-agent-planner/30' },
  analyst: { dot: 'bg-agent-analyst', text: 'text-agent-analyst', chip: 'bg-agent-analyst/10 text-agent-analyst border-agent-analyst/30' },
  ml: { dot: 'bg-agent-ml', text: 'text-agent-ml', chip: 'bg-agent-ml/10 text-agent-ml border-agent-ml/30' },
  reporter: { dot: 'bg-agent-reporter', text: 'text-agent-reporter', chip: 'bg-agent-reporter/10 text-agent-reporter border-agent-reporter/30' },
};

function agentMeta(name) {
  const n = (name || '').toLowerCase();
  if (n.includes('planner')) return AGENT_COLORS.planner;
  if (n.includes('analyst') || n.includes('data')) return AGENT_COLORS.analyst;
  if (n.includes('ml') || n.includes('machine')) return AGENT_COLORS.ml;
  if (n.includes('reporter') || n.includes('report')) return AGENT_COLORS.reporter;
  return { dot: 'bg-ink-faint', text: 'text-walnut-mute', chip: 'bg-walnut-800 text-walnut-text border-walnut-700' };
}

export default function AgentConsole({ events, isRunning }) {
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterAgent, setFilterAgent] = useState('all');
  const [expandedTurns, setExpandedTurns] = useState({});
  const consoleBottomRef = useRef(null);

  useEffect(() => {
    if (autoScroll && consoleBottomRef.current) {
      consoleBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScroll]);

  const toggleTurn = (key) => {
    setExpandedTurns((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const filteredEvents = events.filter((ev) => {
    if (filterAgent === 'all') return true;
    const agent = (ev.data?.agent || ev.agent || '').toLowerCase();
    return agent.includes(filterAgent.toLowerCase());
  });

  return (
    <div className="rounded-xl border border-edge bg-card shadow-card overflow-hidden">
      {/* Chrome */}
      <div className="px-4 py-3 border-b border-edge-soft flex items-center justify-between bg-card">
        <div className="flex items-center gap-2.5">
          <Terminal className="w-4 h-4 text-ink-faint" />
          <span className="text-[13px] font-semibold text-ink">Live Console</span>
          {isRunning && (
            <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-accent-deep">
              <Loader2 className="w-3 h-3 animate-spin" />
              Streaming
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-[12px]">
            <Filter className="w-3.5 h-3.5 text-ink-faint" />
            <select
              value={filterAgent}
              onChange={(e) => setFilterAgent(e.target.value)}
              className="h-7 px-2 rounded-md border border-edge bg-paper text-[12px] text-ink-soft focus:outline-none focus:border-accent cursor-pointer"
            >
              <option value="all">All agents</option>
              <option value="Planner">Planner</option>
              <option value="Analyst">Analyst</option>
              <option value="ML">ML Agent</option>
              <option value="Reporter">Reporter</option>
            </select>
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md border text-[11px] font-medium transition-colors ${
              autoScroll
                ? 'bg-accent-wash text-accent-deep border-accent-edge'
                : 'bg-paper text-ink-faint border-edge hover:text-ink-soft'
            }`}
          >
            <ArrowDownToLine className="w-3 h-3" />
            Auto-scroll {autoScroll ? 'on' : 'off'}
          </button>
        </div>
      </div>

      {/* Stream body — the one dark surface */}
      <div className="bg-walnut-900 scrollbar-walnut overflow-y-auto h-[calc(100vh-260px)] min-h-[420px] max-h-[720px] p-4 space-y-2.5 font-code text-[12px]">
        {filteredEvents.length === 0 ? (
          <div className="h-full min-h-[380px] flex flex-col items-center justify-center text-walnut-mute space-y-2">
            <Bot className="w-7 h-7 opacity-40" strokeWidth={1.5} />
            <p className="font-sans text-[13px]">Live telemetry and agent interactions stream here in real time.</p>
            <p className="font-sans text-[12px] opacity-70">Start a run from Setup &amp; Run to begin.</p>
          </div>
        ) : (
          filteredEvents.map((ev, index) => {
            const eventType = ev.event;
            const data = ev.data || ev;
            const timestamp = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '';
            const key = `event-${index}`;
            const am = agentMeta(data.agent);

            // LLM thought
            if (eventType === 'llm_thought') {
              return (
                <div key={key} className="rounded-lg bg-walnut-800/80 border border-walnut-700 p-3 animate-fade-up">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border font-sans text-[11px] font-medium ${am.chip}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${am.dot}`} />
                      {data.agent || 'Agent'} · thought
                    </span>
                    <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                  </div>
                  {data.reasoning && (
                    <div className="mb-2">
                      <div className="text-[10px] uppercase tracking-wide text-accent mb-1">Model reasoning</div>
                      <p className="text-walnut-text/90 whitespace-pre-wrap font-sans text-[13px] leading-relaxed">
                        {data.reasoning}
                      </p>
                    </div>
                  )}
                  {data.thought && (
                    <div>
                      {data.reasoning && <div className="text-[10px] uppercase tracking-wide text-walnut-mute mb-1">Response</div>}
                      <p className="text-walnut-text/90 whitespace-pre-wrap font-sans text-[13px] leading-relaxed">
                        {data.thought}
                      </p>
                    </div>
                  )}
                </div>
              );
            }

            // Tool executing
            if (eventType === 'tool_executing') {
              let parsedCode = '';
              try {
                const args = typeof data.raw_args === 'string' ? JSON.parse(data.raw_args) : data.raw_args;
                parsedCode = args?.code || '';
              } catch {
                parsedCode = data.raw_args;
              }
              const isExp = expandedTurns[key];

              return (
                <div key={key} className="rounded-lg bg-walnut-950/60 border border-walnut-700 space-y-1 animate-fade-up">
                  <button
                    onClick={() => toggleTurn(key)}
                    className="w-full flex items-center justify-between px-3 py-2 text-left"
                  >
                    <span className="flex items-center gap-2 font-sans text-[12px] font-medium text-walnut-text">
                      {isExp ? <ChevronDown className="w-3.5 h-3.5 text-walnut-mute" /> : <ChevronRight className="w-3.5 h-3.5 text-walnut-mute" />}
                      <Code2 className="w-3.5 h-3.5 text-accent" />
                      {data.agent || 'Agent'} — executing <span className="text-walnut-mute">{data.fn_name || 'execute_python'}</span>
                    </span>
                    <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                  </button>
                  {parsedCode && (
                    <div className={`${isExp ? 'block' : 'hidden'}`}>
                      <pre className="px-3 pb-3 text-[11px] leading-relaxed text-walnut-text/80 overflow-x-auto">
                        <code>{parsedCode}</code>
                      </pre>
                    </div>
                  )}
                </div>
              );
            }

            // Tool completed
            if (eventType === 'tool_completed') {
              const res = data.result || {};
              const stdout = res.stdout || res.output || '';
              const stderr = res.stderr || '';
              const files = res.returned_files || [];
              const isExp = expandedTurns[key];

              return (
                <div key={key} className="rounded-lg bg-walnut-800/50 border border-walnut-700/70 space-y-1 animate-fade-up">
                  <button
                    onClick={() => toggleTurn(key)}
                    className="w-full flex items-center justify-between px-3 py-2 text-left"
                  >
                    <span className="flex items-center gap-2 font-sans text-[12px] font-medium text-walnut-text">
                      {isExp ? <ChevronDown className="w-3.5 h-3.5 text-walnut-mute" /> : <ChevronRight className="w-3.5 h-3.5 text-walnut-mute" />}
                      {res.success ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-accent" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-danger" />
                      )}
                      {data.agent || 'Agent'} — {res.success ? 'tool finished' : 'tool failed'}
                    </span>
                    <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                  </button>

                  {(stdout || stderr || files.length > 0) && (
                    <div className={`${isExp ? 'block' : 'hidden'} px-3 pb-3 space-y-1.5`}>
                      {stdout && (
                        <pre className="bg-walnut-950/70 p-2.5 rounded text-[11px] text-walnut-text/75 overflow-x-auto">
                          <code>{stdout}</code>
                        </pre>
                      )}
                      {stderr && (
                        <pre className="bg-danger/10 border border-danger/30 p-2.5 rounded text-[11px] text-danger/90 overflow-x-auto">
                          <code>{stderr}</code>
                        </pre>
                      )}
                      {files.length > 0 && (
                        <p className="font-sans text-[11px] text-walnut-mute">
                          Saved: {files.join(', ')}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            }

            // Task start / completed
            if (eventType === 'task_execution_start') {
              return (
                <div key={key} className="flex items-center justify-between px-3 py-2 rounded-lg bg-walnut-800/70 border border-walnut-700 animate-fade-up">
                  <span className="font-sans text-[12px] font-medium text-walnut-text flex items-center gap-2">
                    <Play className="w-3.5 h-3.5 text-accent" />
                    Task #{data.task_id} started — {data.title}
                  </span>
                  <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                </div>
              );
            }

            if (eventType === 'task_execution_completed') {
              return (
                <div key={key} className="flex items-center justify-between px-3 py-2 rounded-lg bg-walnut-800/40 border border-walnut-700/60 animate-fade-up">
                  <span className="font-sans text-[12px] text-walnut-text/85 flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-accent" />
                    Task #{data.task_id} {data.status === 'completed' ? 'completed' : `ended (${data.status || 'done'})`} — {data.title}
                  </span>
                  <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                </div>
              );
            }

            // Verification & repair
            if (eventType === 'final_verification_started') {
              return (
                <div key={key} className="flex items-center justify-between px-3 py-2 rounded-lg bg-accent/10 border border-accent/30 animate-fade-up">
                  <span className="font-sans text-[12px] font-medium text-accent flex items-center gap-2">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    Final verification — numeric gate, consistency checks, LLM judge
                  </span>
                  <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                </div>
              );
            }

            if (eventType === 'report_repair_started') {
              return (
                <div key={key} className="px-3 py-2.5 rounded-lg bg-caution/10 border border-caution/30 space-y-0.5 animate-fade-up">
                  <div className="flex items-center justify-between">
                    <span className="font-sans text-[12px] font-semibold text-caution flex items-center gap-2">
                      <Wrench className="w-3.5 h-3.5" />
                      Report repair pass {data.attempt}/{data.max_attempts}
                    </span>
                    <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                  </div>
                  <p className="font-sans text-[11.5px] text-walnut-text/75">
                    {data.critical_count} critical finding(s) remain — all findings combined into one
                    correction prompt (repair agent gets 5 sandbox tool calls).
                  </p>
                </div>
              );
            }

            if (eventType === 'report_repair_completed') {
              return (
                <div key={key} className="flex items-center justify-between px-3 py-2 rounded-lg bg-accent/10 border border-accent/30 animate-fade-up">
                  <span className="font-sans text-[12px] text-accent flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    Repair pass {data.attempt} complete — re-verifying with all methods
                  </span>
                  <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                </div>
              );
            }

            // Workflow cancelled / failed
            if (eventType === 'workflow_cancelled') {
              return (
                <div key={key} className="px-3 py-2.5 rounded-lg bg-danger/10 border border-danger/40 space-y-0.5 animate-fade-up">
                  <div className="flex items-center justify-between">
                    <span className="font-sans text-[12px] font-semibold text-danger">Workflow terminated by user</span>
                    <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                  </div>
                  <p className="font-sans text-[11.5px] text-walnut-text/75">{data.message || 'Execution stopped immediately.'}</p>
                </div>
              );
            }

            if (eventType === 'workflow_failed') {
              return (
                <div key={key} className="px-3 py-2.5 rounded-lg bg-caution/10 border border-caution/40 space-y-0.5 animate-fade-up">
                  <div className="flex items-center justify-between">
                    <span className="font-sans text-[12px] font-semibold text-caution flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> Workflow error
                    </span>
                    <span className="text-walnut-mute text-[10px]">{timestamp}</span>
                  </div>
                  <p className="font-sans text-[11.5px] text-walnut-text/75">{data.error || data.message}</p>
                </div>
              );
            }

            // Generic fallback
            return (
              <div key={key} className="flex items-center justify-between px-3 py-1.5 text-walnut-mute animate-fade-up">
                <span className="font-sans text-[11.5px]">
                  <span className="text-walnut-text/70 font-medium">{eventType}</span>
                  {' — '}
                  {typeof data === 'string' ? data : data.message || JSON.stringify(data).slice(0, 110)}
                </span>
                <span className="text-[10px]">{timestamp}</span>
              </div>
            );
          })
        )}
        <div ref={consoleBottomRef} />
      </div>
    </div>
  );
}
