import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Bot, ChevronDown, ChevronRight, Play, CheckCircle2, Code2, AlertCircle, Sparkles, Filter, ShieldCheck, ShieldAlert } from 'lucide-react';

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

  const getAgentColor = (agent) => {
    const a = (agent || '').toLowerCase();
    if (a.includes('planner')) return 'text-indigo-400 border-indigo-800 bg-indigo-950/60';
    if (a.includes('analyst')) return 'text-emerald-400 border-emerald-800 bg-emerald-950/60';
    if (a.includes('ml')) return 'text-fuchsia-400 border-fuchsia-800 bg-fuchsia-950/60';
    if (a.includes('reporter')) return 'text-amber-400 border-amber-800 bg-amber-950/60';
    return 'text-sky-400 border-sky-800 bg-sky-950/60';
  };

  const filteredEvents = events.filter((ev) => {
    if (filterAgent === 'all') return true;
    const agent = (ev.data?.agent || ev.agent || '').toLowerCase();
    return agent.includes(filterAgent.toLowerCase());
  });

  return (
    <div className="border border-surface-border rounded-xl bg-surface-200/90 flex flex-col h-[520px] overflow-hidden shadow-xl">
      {/* Console Header */}
      <div className="px-4 py-2.5 bg-surface-100/90 border-b border-surface-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-brand-secondary" />
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
            Agent Live Stream
          </span>
          {isRunning && (
            <span className="flex items-center gap-1 text-[11px] text-indigo-400 font-mono">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-ping"></span>
              Streaming
            </span>
          )}
        </div>

        {/* Console Controls */}
        <div className="flex items-center gap-2">
          {/* Agent Filter */}
          <div className="flex items-center gap-1 text-[11px] bg-surface-200 border border-surface-border rounded-md px-2 py-0.5 text-slate-300">
            <Filter className="w-3 h-3 text-slate-500" />
            <select
              value={filterAgent}
              onChange={(e) => setFilterAgent(e.target.value)}
              className="bg-transparent border-none text-slate-300 text-[11px] focus:outline-none cursor-pointer"
            >
              <option value="all">All Agents</option>
              <option value="Planner">Planner</option>
              <option value="Analyst">Analyst</option>
              <option value="ML">ML Agent</option>
              <option value="Reporter">Reporter</option>
            </select>
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
              autoScroll
                ? 'bg-brand-primary/20 text-indigo-300 border-brand-primary/40'
                : 'bg-surface-200 text-slate-400 border-surface-border'
            }`}
          >
            Auto-Scroll: {autoScroll ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Events Stream Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-xs">
        {filteredEvents.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 italic space-y-1">
            <Bot className="w-6 h-6 opacity-40" />
            <p>Live telemetry and agent interactions will stream here in real-time.</p>
          </div>
        ) : (
          filteredEvents.map((ev, index) => {
            const eventType = ev.event;
            const data = ev.data || ev;
            const timestamp = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '';
            const key = `event-${index}`;

            // 1. LLM Thought Event
            if (eventType === 'llm_thought') {
              return (
                <div key={key} className="p-3 rounded-lg bg-surface-100/70 border border-surface-border">
                  <div className="flex items-center justify-between text-[11px] mb-1.5">
                    <span className={`px-2 py-0.5 rounded border font-semibold ${getAgentColor(data.agent)}`}>
                      {data.agent || 'Agent'} Thought (Turn {data.turn || 1})
                    </span>
                    <span className="text-slate-500 text-[10px]">{timestamp}</span>
                  </div>
                  <p className="text-slate-300 whitespace-pre-wrap font-sans text-xs leading-relaxed">
                    {data.thought}
                  </p>
                </div>
              );
            }

            // 2. Tool Executing
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
                <div key={key} className="p-3 rounded-lg bg-indigo-950/20 border border-indigo-900/50 space-y-1.5">
                  <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => toggleTurn(key)}
                  >
                    <div className="flex items-center gap-2">
                      {isExp ? <ChevronDown className="w-3.5 h-3.5 text-indigo-400" /> : <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />}
                      <Code2 className="w-3.5 h-3.5 text-indigo-400" />
                      <span className="text-xs font-semibold text-indigo-300">
                        {data.agent}: Executing {data.fn_name || 'Python Script'}
                      </span>
                    </div>
                    <span className="text-slate-500 text-[10px]">{timestamp}</span>
                  </div>

                  {parsedCode && (
                    <div className={isExp ? 'block' : 'line-clamp-3'}>
                      <pre className="bg-black/50 p-2 rounded text-[11px] text-emerald-300 overflow-x-auto border border-indigo-950">
                        <code>{parsedCode}</code>
                      </pre>
                    </div>
                  )}
                </div>
              );
            }

            // 3. Tool Completed
            if (eventType === 'tool_completed') {
              const res = data.result || {};
              const stdout = res.stdout || res.output || '';
              const stderr = res.stderr || '';
              const files = res.returned_files || [];
              const isExp = expandedTurns[key];

              return (
                <div key={key} className="p-3 rounded-lg bg-surface-100/50 border border-surface-border space-y-1.5">
                  <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => toggleTurn(key)}
                  >
                    <div className="flex items-center gap-2">
                      {isExp ? <ChevronDown className="w-3.5 h-3.5 text-emerald-400" /> : <ChevronRight className="w-3.5 h-3.5 text-emerald-400" />}
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-xs font-medium text-slate-300">
                        {data.agent}: Tool Output {res.success ? '(Success)' : '(Finished)'}
                      </span>
                    </div>
                    <span className="text-slate-500 text-[10px]">{timestamp}</span>
                  </div>

                  {(stdout || stderr || files.length > 0) && (
                    <div className={`space-y-1 ${isExp ? 'block' : 'line-clamp-3'}`}>
                      {stdout && (
                        <pre className="bg-black/40 p-2 rounded text-[11px] text-slate-300 overflow-x-auto">
                          <code>{stdout}</code>
                        </pre>
                      )}
                      {stderr && (
                        <pre className="bg-red-950/30 border border-red-900/40 p-2 rounded text-[11px] text-red-300 overflow-x-auto">
                          <code>{stderr}</code>
                        </pre>
                      )}
                      {files.length > 0 && (
                        <p className="text-[11px] text-brand-secondary font-mono">
                          Saved Artifacts: {files.join(', ')}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            }

            // 4. Task Execution Start / Completed
            if (eventType === 'task_execution_start') {
              return (
                <div key={key} className="p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-700/60 flex items-center justify-between">
                  <span className="text-xs font-semibold text-indigo-300 flex items-center gap-1.5">
                    <Play className="w-3.5 h-3.5 text-indigo-400" />
                    Starting Task #{data.task_id}: {data.title} ({data.assigned_agent})
                  </span>
                  <span className="text-slate-500 text-[10px]">{timestamp}</span>
                </div>
              );
            }

            if (eventType === 'task_execution_completed') {
              return (
                <div key={key} className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-800/50 flex items-center justify-between">
                  <span className="text-xs font-medium text-emerald-300 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    Completed Task #{data.task_id}: {data.title}
                  </span>
                  <span className="text-slate-500 text-[10px]">{timestamp}</span>
                </div>
              );
            }

            // 5. Workflow Cancelled Event
            if (eventType === 'workflow_cancelled') {
              return (
                <div key={key} className="p-3 rounded-lg bg-red-950/60 border border-red-700 text-red-300 space-y-1">
                  <div className="flex items-center justify-between font-bold">
                    <span>🛑 Workflow Execution Halted by User Request</span>
                    <span className="text-[10px] text-red-400 font-normal">{timestamp}</span>
                  </div>
                  <p className="text-xs text-red-200">{data.message || 'Execution stopped immediately.'}</p>
                </div>
              );
            }

            // 6. Workflow Failed Event
            if (eventType === 'workflow_failed') {
              return (
                <div key={key} className="p-3 rounded-lg bg-amber-950/60 border border-amber-700 text-amber-300 space-y-1">
                  <div className="flex items-center justify-between font-bold">
                    <span>⚠️ Workflow Execution Error</span>
                    <span className="text-[10px] text-amber-400 font-normal">{timestamp}</span>
                  </div>
                  <p className="text-xs text-amber-200">{data.error || data.message}</p>
                </div>
              );
            }

            // 7. Final Verification & Repair Events
            if (eventType === 'final_verification_started') {
              return (
                <div key={key} className="p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-700/60 flex items-center justify-between">
                  <span className="text-xs font-semibold text-indigo-300 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                    Final Verification: numeric gate + consistency checker + LLM judge
                  </span>
                  <span className="text-slate-500 text-[10px]">{timestamp}</span>
                </div>
              );
            }

            if (eventType === 'report_repair_started') {
              return (
                <div key={key} className="p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-300 space-y-1">
                  <div className="flex items-center justify-between font-bold text-xs">
                    <span className="flex items-center gap-1.5">
                      <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                      Report Repair Pass {data.attempt}/{data.max_attempts}
                    </span>
                    <span className="text-[10px] text-amber-400 font-normal">{timestamp}</span>
                  </div>
                  <p className="text-xs text-amber-200">
                    {data.critical_count} critical finding(s) survived. Combining all verification findings into one
                    correction prompt (repair agent gets 5 sandbox tool calls).
                  </p>
                </div>
              );
            }

            if (eventType === 'report_repair_completed') {
              return (
                <div key={key} className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-800/50 flex items-center justify-between">
                  <span className="text-xs font-medium text-emerald-300 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    Repair Pass {data.attempt} complete — re-verifying with all methods
                  </span>
                  <span className="text-slate-500 text-[10px]">{timestamp}</span>
                </div>
              );
            }

            // Fallback generic event log
            return (
              <div key={key} className="p-2 rounded bg-surface-100/40 border border-surface-border text-slate-400 flex items-center justify-between text-[11px]">
                <span>
                  <strong className="text-slate-300">{eventType}: </strong>
                  {typeof data === 'string' ? data : data.message || JSON.stringify(data).slice(0, 120)}
                </span>
                <span className="text-slate-500 text-[10px]">{timestamp}</span>
              </div>
            );
          })
        )}
        <div ref={consoleBottomRef} />
      </div>
    </div>
  );
}
