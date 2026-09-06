import React from 'react';
import { CheckCircle2, XCircle, AlertTriangle, Info, ShieldCheck, ShieldAlert, Flame } from 'lucide-react';

export default function StatusAlertBanner({ status, message, error, cancelledStep }) {
  if (!status || status === 'idle') return null;

  if (status === 'completed') {
    return (
      <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-700/60 shadow-lg glow-success flex items-start gap-3 animate-in fade-in">
        <div className="w-8 h-8 rounded-lg bg-emerald-900/60 border border-emerald-600 flex items-center justify-center shrink-0 mt-0.5">
          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
        </div>
        <div className="flex-1">
          <h4 className="text-sm font-bold text-emerald-300 flex items-center gap-2">
            Workflow Execution Successfully Completed
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-900/60 text-emerald-200 border border-emerald-700">
              Verified
            </span>
          </h4>
          <p className="text-xs text-emerald-200/90 mt-1 leading-relaxed">
            {message || 'All tasks finished and final executive synthesis report generated.'}
          </p>
        </div>
      </div>
    );
  }

  if (status === 'completed_with_warnings') {
    return (
      <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-700/70 shadow-lg glow-warning flex items-start gap-3 animate-in fade-in">
        <div className="w-8 h-8 rounded-lg bg-amber-900/60 border border-amber-600 flex items-center justify-center shrink-0 mt-0.5">
          <ShieldAlert className="w-5 h-5 text-amber-400" />
        </div>
        <div className="flex-1">
          <h4 className="text-sm font-bold text-amber-300 flex items-center gap-2">
            Workflow Completed With Warnings
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-amber-900/60 text-amber-200 border border-amber-700">
              Review Findings
            </span>
          </h4>
          <p className="text-xs text-amber-200/90 mt-1 leading-relaxed">
            {message || 'The report is delivered, but minor consistency warnings were detected. Review the verification findings below.'}
          </p>
        </div>
      </div>
    );
  }

  if (status === 'failed-consistency') {
    return (
      <div className="p-4 rounded-xl bg-red-950/50 border border-red-700/80 shadow-lg glow-danger flex items-start gap-3 animate-in fade-in">
        <div className="w-8 h-8 rounded-lg bg-red-900/60 border border-red-600 flex items-center justify-center shrink-0 mt-0.5">
          <ShieldAlert className="w-5 h-5 text-red-400" />
        </div>
        <div className="flex-1">
          <h4 className="text-sm font-bold text-red-300 flex items-center gap-2">
            Run Not Cleared For Delivery
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-red-900/60 text-red-200 border border-red-700">
              Failed Consistency
            </span>
          </h4>
          <p className="text-xs text-red-200/90 mt-1 leading-relaxed">
            {message || 'Critical consistency findings survived verification and the automated repair passes. The report is NOT cleared for delivery — review the verification findings.'}
          </p>
        </div>
      </div>
    );
  }

  if (status === 'cancelled') {
    return (
      <div className="p-4 rounded-xl bg-red-950/50 border border-red-700/80 shadow-lg glow-danger flex items-start gap-3 animate-in fade-in">
        <div className="w-8 h-8 rounded-lg bg-red-900/60 border border-red-600 flex items-center justify-center shrink-0 mt-0.5">
          <Flame className="w-5 h-5 text-red-400" />
        </div>
        <div className="flex-1">
          <h4 className="text-sm font-bold text-red-300 flex items-center gap-2">
            Workflow Execution Stopped by User Request
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-red-900/60 text-red-200 border border-red-700">
              Terminated
            </span>
          </h4>
          <p className="text-xs text-red-200/90 mt-1 leading-relaxed">
            {message || 'Workflow was intentionally terminated by user confirmation. No unwanted subsequent calls were made.'}
          </p>
          {cancelledStep && (
            <p className="text-[11px] text-red-300 font-mono mt-1">
              Stopped at step: <span className="underline">{cancelledStep}</span>
            </p>
          )}
        </div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="p-4 rounded-xl bg-amber-950/50 border border-amber-700/80 shadow-lg glow-warning flex items-start gap-3 animate-in fade-in">
        <div className="w-8 h-8 rounded-lg bg-amber-900/60 border border-amber-600 flex items-center justify-center shrink-0 mt-0.5">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
        </div>
        <div className="flex-1">
          <h4 className="text-sm font-bold text-amber-300 flex items-center gap-2">
            Workflow Halted Due to Execution Error
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-amber-900/60 text-amber-200 border border-amber-700">
              Diagnostic State
            </span>
          </h4>
          <p className="text-xs text-amber-200/90 mt-1 leading-relaxed">
            {message || 'An error occurred during task execution. Pipeline stopped cleanly and state was preserved.'}
          </p>
          {error && (
            <div className="mt-2 p-2.5 rounded-lg bg-black/40 border border-amber-900/60 font-mono text-[11px] text-amber-200 overflow-x-auto">
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}
