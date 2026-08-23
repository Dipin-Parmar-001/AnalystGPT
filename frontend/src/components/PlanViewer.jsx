import React, { useState } from 'react';
import {
  ListChecks,
  CheckCircle2,
  Clock,
  PlayCircle,
  XCircle,
  AlertTriangle,
  Bot,
  BrainCircuit,
  BarChart3,
  Cpu,
  FileText,
  Edit3,
  Send
} from 'lucide-react';

export default function PlanViewer({
  plan,
  isRunning,
  onExecutePlan,
  onModifyPlan,
  disabled
}) {
  const [isModifying, setIsModifying] = useState(false);
  const [modPrompt, setModPrompt] = useState('');
  const [submittingMod, setSubmittingMod] = useState(false);

  if (!plan || !plan.tasks || plan.tasks.length === 0) {
    return (
      <div className="p-6 rounded-xl border border-surface-border bg-surface-100/40 text-center">
        <BrainCircuit className="w-8 h-8 text-slate-500 mx-auto mb-2 opacity-60" />
        <p className="text-sm text-slate-400 font-medium">No Execution Plan Active</p>
        <p className="text-xs text-slate-500 mt-1">
          Enter an analysis prompt above and generate a plan to view decomposed multi-agent tasks.
        </p>
      </div>
    );
  }

  const getAgentBadge = (agentName) => {
    const norm = (agentName || '').toLowerCase();
    if (norm.includes('analyst') || norm.includes('data')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">
          <BarChart3 className="w-3 h-3" />
          DataAnalyst
        </span>
      );
    } else if (norm.includes('ml') || norm.includes('machine') || norm.includes('model')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-fuchsia-950/80 text-fuchsia-300 border border-fuchsia-800/60">
          <Cpu className="w-3 h-3" />
          MLAgent
        </span>
      );
    } else if (norm.includes('reporter') || norm.includes('report')) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-950/80 text-amber-300 border border-amber-800/60">
          <FileText className="w-3 h-3" />
          ReporterAgent
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-950/80 text-indigo-300 border border-indigo-800/60">
        <Bot className="w-3 h-3" />
        {agentName || 'Planner'}
      </span>
    );
  };

  const getTaskStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Completed
          </span>
        );
      case 'in_progress':
        return (
          <span className="inline-flex items-center gap-1 text-xs text-indigo-400 font-medium animate-pulse">
            <PlayCircle className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
            In Progress
          </span>
        );
      case 'cancelled':
        return (
          <span className="inline-flex items-center gap-1 text-xs text-red-400 font-medium">
            <XCircle className="w-3.5 h-3.5" />
            Cancelled
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1 text-xs text-amber-400 font-medium">
            <AlertTriangle className="w-3.5 h-3.5" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-xs text-slate-500 font-medium">
            <Clock className="w-3.5 h-3.5" />
            Pending
          </span>
        );
    }
  };

  const handleModifySubmit = async (e) => {
    e.preventDefault();
    if (!modPrompt.trim() || submittingMod) return;
    setSubmittingMod(true);
    try {
      await onModifyPlan(modPrompt);
      setModPrompt('');
      setIsModifying(false);
    } finally {
      setSubmittingMod(false);
    }
  };

  return (
    <div className="border border-surface-border rounded-xl bg-surface-100/60 p-4 space-y-4">
      {/* Header with plan info & modification toggle */}
      <div className="flex items-center justify-between border-b border-surface-border/80 pb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-950/80 border border-indigo-700/60 flex items-center justify-center">
            <ListChecks className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              Execution Plan
              <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-surface-200 text-slate-400 border border-surface-border">
                #{plan.plan_id || 'active'}
              </span>
            </h3>
            <p className="text-[11px] text-slate-400">
              {plan.tasks.length} Decomposed Steps • Revision {plan.revision_count || 0}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsModifying(!isModifying)}
            disabled={isRunning || disabled}
            className="px-2.5 py-1 rounded-lg text-xs font-medium bg-surface-200 hover:bg-surface-hover border border-surface-border text-slate-300 flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <Edit3 className="w-3.5 h-3.5 text-brand-secondary" />
            {isModifying ? 'Close' : 'Modify Plan'}
          </button>

          {!isRunning && (
            <button
              onClick={onExecutePlan}
              disabled={disabled}
              className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-brand-primary to-indigo-600 hover:from-brand-primaryHover hover:to-indigo-500 text-white shadow-md shadow-brand-primary/20 flex items-center gap-1.5 transition-all disabled:opacity-50"
            >
              <PlayCircle className="w-3.5 h-3.5" />
              Execute Workflow
            </button>
          )}
        </div>
      </div>

      {/* Optional Modification Box */}
      {isModifying && (
        <form onSubmit={handleModifySubmit} className="p-3 rounded-lg bg-surface-200 border border-indigo-900/60 space-y-2">
          <label className="text-xs font-medium text-indigo-300">
            Tell PlannerAgent how to adapt this plan:
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={modPrompt}
              onChange={(e) => setModPrompt(e.target.value)}
              placeholder="e.g. Include a confusion matrix plot and focus only on Random Forest..."
              className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-surface-300 border border-surface-border text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-primary"
            />
            <button
              type="submit"
              disabled={submittingMod || !modPrompt.trim()}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-brand-primary hover:bg-brand-primaryHover text-white flex items-center gap-1 disabled:opacity-50"
            >
              {submittingMod ? 'Updating...' : <Send className="w-3 h-3" />}
            </button>
          </div>
        </form>
      )}

      {/* Plan Notes if any */}
      {plan.notes && (
        <div className="p-2.5 rounded-lg bg-surface-200/60 border border-surface-border text-xs text-slate-300 italic">
          <span className="font-semibold text-slate-400 not-italic">Planner Strategy: </span>
          {plan.notes}
        </div>
      )}

      {/* Task List Cards */}
      <div className="space-y-2.5">
        {plan.tasks.map((task, idx) => (
          <div
            key={task.task_id || idx}
            className={`p-3 rounded-lg border transition-all ${
              task.status === 'in_progress'
                ? 'bg-indigo-950/40 border-indigo-600/80 shadow-md shadow-indigo-900/20'
                : task.status === 'completed'
                ? 'bg-surface-200/70 border-emerald-900/40'
                : task.status === 'cancelled'
                ? 'bg-red-950/20 border-red-900/40 opacity-75'
                : task.status === 'failed'
                ? 'bg-amber-950/30 border-amber-800/50'
                : 'bg-surface-200/40 border-surface-border'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-start gap-2.5">
                <div className="w-5 h-5 rounded-full bg-surface-300 border border-surface-border flex items-center justify-center text-[10px] font-mono font-bold text-slate-400 shrink-0 mt-0.5">
                  {task.task_id || idx + 1}
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-slate-100 flex items-center gap-2">
                    {task.title}
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                    {task.description}
                  </p>
                  {task.expected_output && (
                    <p className="text-[10px] text-slate-500 font-mono mt-1">
                      Deliverable: {task.expected_output}
                    </p>
                  )}
                </div>
              </div>

              <div className="flex flex-col items-end gap-1.5 shrink-0">
                {getAgentBadge(task.assigned_agent)}
                {getTaskStatusBadge(task.status)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
