import React from 'react';
import { CheckCircle2, AlertTriangle, ShieldAlert, Flame, Loader2, PauseCircle } from 'lucide-react';

const BANNERS = {
  pausing: {
    cls: 'bg-amber-500/10 border-amber-500/30',
    icon: Loader2,
    iconCls: 'bg-amber-500/15 text-amber-500 border-amber-500/30 animate-spin',
    titleCls: 'text-amber-500',
    textCls: 'text-ink-soft',
    chip: { label: 'Pausing...', cls: 'bg-amber-500 text-white' },
    fallback: 'Workflow is pausing. Halting the active step and saving checkpoint...',
  },
  paused: {
    cls: 'bg-amber-500/10 border-amber-500/30',
    icon: PauseCircle,
    iconCls: 'bg-amber-500/15 text-amber-500 border-amber-500/30',
    titleCls: 'text-amber-500',
    textCls: 'text-ink-soft',
    chip: { label: 'Paused', cls: 'bg-amber-500 text-white' },
    fallback: 'Workflow is paused. You can resume anytime or save to disk to continue tomorrow or after restart.',
  },
  completed: {
    cls: 'bg-accent-wash border-accent-edge',
    icon: CheckCircle2,
    iconCls: 'bg-accent-wash text-accent border-accent-edge',
    titleCls: 'text-accent-deep',
    textCls: 'text-ink-soft',
    chip: { label: 'Verified', cls: 'bg-accent text-white' },
    fallback: 'All tasks finished, the report passed verification, and the executive deliverable is ready.',
  },
  completed_with_warnings: {
    cls: 'bg-caution-wash border-caution-edge',
    icon: ShieldAlert,
    iconCls: 'bg-caution-wash text-caution border-caution-edge',
    titleCls: 'text-caution',
    textCls: 'text-ink-soft',
    chip: { label: 'Review findings', cls: 'bg-caution text-white' },
    fallback: 'The report is delivered, but minor consistency warnings were detected. Open Verification for the details.',
  },
  'failed-consistency': {
    cls: 'bg-danger-wash border-danger-edge',
    icon: ShieldAlert,
    iconCls: 'bg-danger-wash text-danger border-danger-edge',
    titleCls: 'text-danger',
    textCls: 'text-ink-soft',
    chip: { label: 'Not cleared', cls: 'bg-danger text-white' },
    fallback: 'Critical consistency findings survived verification and the automated repair passes. The report is NOT cleared for delivery — open Verification for the findings.',
  },
  cancelled: {
    cls: 'bg-card border-edge',
    icon: Flame,
    iconCls: 'bg-danger-wash text-danger border-danger-edge',
    titleCls: 'text-ink',
    textCls: 'text-ink-soft',
    chip: { label: 'Terminated', cls: 'bg-danger text-white' },
    fallback: 'Workflow was intentionally terminated by user confirmation. No further agent calls were made.',
  },
  failed: {
    cls: 'bg-caution-wash border-caution-edge',
    icon: AlertTriangle,
    iconCls: 'bg-caution-wash text-caution border-caution-edge',
    titleCls: 'text-caution',
    textCls: 'text-ink-soft',
    chip: { label: 'Error', cls: 'bg-caution text-white' },
    fallback: 'An error occurred during task execution. Pipeline stopped cleanly and state was preserved.',
  },
};

export default function StatusAlertBanner({ status, message, error, cancelledStep }) {
  if (!status || status === 'idle' || status === 'planning' || status === 'plan_ready' || status === 'executing') return null;

  const b = BANNERS[status];
  if (!b) return null;

  const Icon = b.icon;

  return (
    <div className={`p-4 rounded-xl border ${b.cls} flex items-start gap-3.5 animate-fade-up`}>
      <div className={`w-9 h-9 rounded-lg border flex items-center justify-center shrink-0 ${b.iconCls}`}>
        <Icon className="w-[18px] h-[18px]" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2.5 flex-wrap">
          <h4 className={`text-[14px] font-semibold ${b.titleCls}`}>
            {status === 'paused' && 'Workflow paused'}
            {status === 'completed' && 'Workflow completed'}
            {status === 'completed_with_warnings' && 'Completed with warnings'}
            {status === 'failed-consistency' && 'Run not cleared for delivery'}
            {status === 'cancelled' && 'Workflow terminated'}
            {status === 'failed' && 'Execution error'}
          </h4>
          <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full text-white ${b.chip.cls}`}>
            {b.chip.label}
          </span>
        </div>
        <p className={`text-[13px] mt-1 leading-relaxed ${b.textCls}`}>
          {message || b.fallback}
        </p>

        {cancelledStep && status === 'cancelled' && (
          <p className="text-[12px] text-ink-faint mt-1.5">
            Stopped at step: <span className="text-ink-soft font-medium">{cancelledStep}</span>
          </p>
        )}
        {error && status === 'failed' && (
          <pre className="mt-2.5 p-3 rounded-lg bg-card border border-caution-edge text-[11px] font-code text-danger overflow-x-auto whitespace-pre-wrap">
            {error}
          </pre>
        )}
      </div>
    </div>
  );
}
