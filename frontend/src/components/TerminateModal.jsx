import React from 'react';
import { AlertOctagon, ShieldAlert, Check, X, Flame } from 'lucide-react';

export default function TerminateModal({ isOpen, onClose, onConfirm, terminating }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface-100 border border-red-800/80 rounded-2xl max-w-md w-full p-6 shadow-2xl glow-danger relative overflow-hidden">
        {/* Top warning accent bar */}
        <div className="absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-red-600 via-rose-500 to-amber-500"></div>

        {/* Icon & Title */}
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-red-950/80 border border-red-700 flex items-center justify-center shrink-0">
            <AlertOctagon className="w-6 h-6 text-red-400 animate-pulse" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">
              Terminate Multi-Agent Workflow?
            </h3>
            <p className="text-xs text-red-300/90 font-mono mt-0.5">
              Confirm Immediate Cancellation
            </p>
          </div>
        </div>

        {/* Details & Impact Explanation */}
        <div className="mt-4 p-3.5 rounded-xl bg-surface-200/80 border border-surface-border text-xs text-slate-300 space-y-2">
          <p className="font-semibold text-slate-200 flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-red-400" />
            What happens when confirmed:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-400 text-[11px]">
            <li>Current agent code execution & sandbox processes stop immediately.</li>
            <li>All remaining planned tasks and report synthesis are aborted.</li>
            <li>No additional tokens or LLM queries will be consumed.</li>
            <li>Artifacts generated before termination are preserved.</li>
          </ul>
        </div>

        {/* Buttons: Cancel vs Confirm Termination */}
        <div className="mt-6 flex items-center justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={terminating}
            className="px-4 py-2 rounded-xl text-xs font-medium bg-surface-200 hover:bg-surface-hover border border-surface-border text-slate-300 transition-colors"
          >
            Keep Running
          </button>

          <button
            type="button"
            onClick={onConfirm}
            disabled={terminating}
            className="px-4 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-red-600 to-rose-700 hover:from-red-500 hover:to-rose-600 text-white shadow-lg shadow-red-900/40 flex items-center gap-1.5 transition-all disabled:opacity-50"
          >
            {terminating ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Halting Workflow...
              </>
            ) : (
              <>
                <Flame className="w-3.5 h-3.5" />
                Confirm Termination
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
