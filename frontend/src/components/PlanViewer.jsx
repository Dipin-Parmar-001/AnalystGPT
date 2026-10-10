import React, { useState } from 'react';
import {
  ListChecks, CheckCircle2, Clock, Loader2, XCircle, AlertTriangle,
  BrainCircuit, BarChart3, Cpu, FileText, Edit3, Send, Bot, Play
} from 'lucide-react';

const AGENT_BADGES = {
  analyst: { icon: BarChart3, label: 'DataAnalyst', cls: 'bg-agent-analyst/10 text-agent-analyst border-agent-analyst/25' },
  ml: { icon: Cpu, label: 'MLAgent', cls: 'bg-agent-ml/10 text-agent-ml border-agent-ml/25' },
  reporter: { icon: FileText, label: 'ReporterAgent', cls: 'bg-agent-reporter/10 text-agent-reporter border-agent-reporter/25' },
  planner: { icon: Bot, label: 'Planner', cls: 'bg-agent-planner/10 text-agent-planner border-agent-planner/25' },
};

function getAgentBadge(agentName) {
  const n = (agentName || '').toLowerCase();
  let m = AGENT_BADGES.planner;
  if (n.includes('analyst') || n.includes('data')) m = AGENT_BADGES.analyst;
  else if (n.includes('ml') || n.includes('machine') || n.includes('model')) m = AGENT_BADGES.ml;
  else if (n.includes('reporter') || n.includes('report')) m = AGENT_BADGES.reporter;
  const Icon = m.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium ${m.cls}`}>
      <Icon className="w-3 h-3" />
      {m.label === 'Planner' ? (agentName || 'Planner') : m.label}
    </span>
  );
}

function TaskStatusBadge({ status }) {
  switch (status) {
    case 'completed':
      return (
        <span className="inline-flex items-center gap-1 text-[12px] text-accent-deep font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" /> Completed
        </span>
      );
    case 'in_progress':
      return (
        <span className="inline-flex items-center gap-1 text-[12px] text-agent-planner font-medium">
          <Loader2 className="w-3.5 h-3.5 animate-spin" /> In progress
        </span>
      );
    case 'cancelled':
      return (
        <span className="inline-flex items-center gap-1 text-[12px] text-ink-faint font-medium">
          <XCircle className="w-3.5 h-3.5" /> Cancelled
        </span>
      );
    case 'failed':
      return (
        <span className="inline-flex items-center gap-1 text-[12px] text-danger font-medium">
          <AlertTriangle className="w-3.5 h-3.5" /> Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 text-[12px] text-ink-faint font-medium">
          <Clock className="w-3.5 h-3.5" /> Pending
        </span>
      );
  }
}

export default function PlanViewer({ plan, isRunning, onExecutePlan, onModifyPlan, disabled }) {
  const [isModifying, setIsModifying] = useState(false);
  const [modPrompt, setModPrompt] = useState('');
  const [submittingMod, setSubmittingMod] = useState(false);

  if (!plan || !plan.tasks || plan.tasks.length === 0) {
    return (
      <div className="p-10 text-center border border-dashed border-edge-strong rounded-xl bg-card/60">
        <BrainCircuit className="w-8 h-8 text-ink-faint/50 mx-auto mb-3" strokeWidth={1.6} />
        <p className="text-[14px] font-medium text-ink-soft">No execution plan yet</p>
        <p className="text-[13px] text-ink-faint mt-1">
          Describe your analysis in Setup &amp; Run and generate a plan to see the decomposed tasks here.
        </p>
      </div>
    );
  }

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

  const completed = plan.tasks.filter((t) => t.status === 'completed').length;

  return (
    <div className="bg-card border border-edge rounded-xl shadow-card overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-edge-soft flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-agent-planner/10 border border-agent-planner/25 flex items-center justify-center">
            <ListChecks className="w-[18px] h-[18px] text-agent-planner" />
          </div>
          <div>
            <h3 className="text-[14px] font-semibold text-ink flex items-center gap-2">
              Execution plan
              <span className="text-[11px] font-code text-ink-faint bg-paper border border-edge rounded px-1.5 py-0.5">
                #{plan.plan_id || 'active'}
              </span>
            </h3>
            <p className="text-[12px] text-ink-faint mt-0.5">
              {plan.tasks.length} tasks · {completed} completed · revision {plan.revision_count || 0}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsModifying(!isModifying)}
            disabled={isRunning || disabled}
            className="inline-flex items-center gap-1.5 h-8 px-3 rounded-lg border border-edge-strong bg-card hover:bg-paper text-[12px] font-medium text-ink transition-colors disabled:opacity-40"
          >
            <Edit3 className="w-3.5 h-3.5 text-ink-faint" />
            {isModifying ? 'Close' : 'Modify plan'}
          </button>

          {!isRunning && (
            <button
              onClick={onExecutePlan}
              disabled={disabled}
              className="inline-flex items-center gap-1.5 h-8 px-4 rounded-lg bg-accent hover:bg-accent-deep text-white text-[12px] font-semibold transition-colors disabled:opacity-40"
            >
              <Play className="w-3.5 h-3.5" />
              Execute workflow
            </button>
          )}
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* Modify box */}
        {isModifying && (
          <form onSubmit={handleModifySubmit} className="p-3.5 rounded-lg bg-paper border border-edge space-y-2">
            <label className="text-[12px] font-medium text-ink-soft">
              Tell the PlannerAgent how to adapt this plan:
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={modPrompt}
                onChange={(e) => setModPrompt(e.target.value)}
                placeholder="e.g. Include a confusion matrix plot and focus only on Random Forest…"
                className="flex-1 h-9 px-3 text-[13px] rounded-lg border border-edge bg-card text-ink placeholder-ink-faint focus:outline-none focus:border-accent"
              />
              <button
                type="submit"
                disabled={submittingMod || !modPrompt.trim()}
                className="inline-flex items-center gap-1.5 h-9 px-3.5 rounded-lg bg-accent hover:bg-accent-deep text-white text-[12px] font-medium transition-colors disabled:opacity-40"
              >
                {submittingMod ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                Apply
              </button>
            </div>
          </form>
        )}

        {/* Planner strategy notes */}
        {plan.notes && (
          <div className="p-3.5 rounded-lg bg-paper border border-edge-soft text-[12.5px] text-ink-soft leading-relaxed">
            <span className="font-semibold text-ink">Planner strategy: </span>
            {plan.notes}
          </div>
        )}

        {/* Task list */}
        <ol className="relative space-y-3">
          {plan.tasks.map((task, idx) => {
            const inProgress = task.status === 'in_progress';
            return (
              <li
                key={task.task_id || idx}
                className={`flex gap-3.5 p-4 rounded-lg border transition-colors ${inProgress
                    ? 'bg-accent-wash/60 border-accent-edge'
                    : task.status === 'failed'
                      ? 'bg-caution-wash/50 border-caution-edge'
                      : task.status === 'cancelled'
                        ? 'bg-paper border-edge-soft opacity-70'
                        : 'bg-card border-edge hover:border-edge-strong'
                  }`}
              >
                {/* Step indicator */}
                <span
                  className={`w-6 h-6 rounded-full border flex items-center justify-center text-[11px] font-semibold shrink-0 mt-0.5 ${task.status === 'completed'
                      ? 'bg-accent border-accent text-white'
                      : inProgress
                        ? 'bg-accent-wash border-accent text-accent-deep'
                        : 'bg-paper border-edge-strong text-ink-faint'
                    }`}
                >
                  {task.status === 'completed' ? <CheckCircle2 className="w-3.5 h-3.5" /> : task.task_id || idx + 1}
                </span>

                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-3">
                    <h4 className="text-[13px] font-semibold text-ink leading-snug">{task.title}</h4>
                    <div className="flex flex-col items-end gap-1 shrink-0">
                      {getAgentBadge(task.assigned_agent)}
                      <TaskStatusBadge status={task.status} />
                    </div>
                  </div>
                  <p className="text-[12.5px] text-ink-soft mt-1 leading-relaxed">{task.description}</p>
                  {task.expected_output && (
                    <p className="text-[11.5px] text-ink-faint mt-1.5">
                      Deliverable: <span className="font-code text-[11px]">{task.expected_output}</span>
                    </p>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
