import React from 'react';
import { AlertOctagon, ShieldAlert, Loader2, X, Flame } from 'lucide-react';

export default function TerminateModal({ isOpen, onClose, onConfirm, terminating }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/60 animate-fade-up">
      <div className="bg-card border border-danger-edge rounded-xl max-w-md w-full shadow-pop relative overflow-hidden">
        <button
          onClick={onClose}
          className="absolute top-3.5 right-3.5 p-1.5 rounded-md text-ink-faint hover:text-ink hover:bg-paper transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="p-6">
          {/* Title */}
          <div className="flex items-start gap-4">
            <div className="w-11 h-11 rounded-lg bg-danger-wash border border-danger-edge flex items-center justify-center shrink-0">
              <AlertOctagon className="w-5 h-5 text-danger" />
            </div>
            <div>
              <h3 className="text-[16px] font-semibold text-ink">Terminate the workflow?</h3>
              <p className="text-[12.5px] text-ink-soft mt-0.5">
                This immediately halts every agent and sandbox process.
              </p>
            </div>
          </div>

          {/* Impact */}
          <div className="mt-5 p-4 rounded-lg bg-paper border border-edge text-[13px] text-ink-soft">
            <p className="font-medium text-ink flex items-center gap-1.5 pb-2">
              <ShieldAlert className="w-4 h-4 text-danger" />
              What happens when you confirm
            </p>
            <ul className="list-disc pl-5 space-y-1 text-[12.5px]">
              <li>Current agent code execution and sandbox processes stop immediately.</li>
              <li>All remaining tasks and report synthesis are aborted.</li>
              <li>No further tokens or LLM queries are consumed.</li>
              <li>Artifacts generated before termination are preserved.</li>
            </ul>
          </div>

          {/* Actions */}
          <div className="mt-6 flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={onClose}
              disabled={terminating}
              className="h-9 px-4 rounded-lg border border-edge-strong bg-card hover:bg-paper text-[13px] font-medium text-ink transition-colors disabled:opacity-50"
            >
              Keep running
            </button>

            <button
              type="button"
              onClick={onConfirm}
              disabled={terminating}
              className="inline-flex items-center gap-2 h-9 px-4 rounded-lg bg-danger hover:bg-danger-deep text-white text-[13px] font-semibold transition-colors disabled:opacity-60"
            >
              {terminating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Halting…
                </>
              ) : (
                <>
                  <Flame className="w-4 h-4" />
                  Confirm termination
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
