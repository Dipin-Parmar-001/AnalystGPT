import React from 'react';
import { RefreshCw, Wifi, WifiOff, Flame, CheckCircle2, ShieldAlert, Play, ListChecks, XCircle, AlertTriangle, Loader2 } from 'lucide-react';

const RUN_STATE_META = {
  idle: { label: 'Idle', cls: 'bg-paper text-ink-faint border-edge', icon: null, spin: false },
  planning: { label: 'Planning', cls: 'bg-paper text-agent-planner border-edge', icon: Loader2, spin: true },
  plan_ready: { label: 'Plan ready', cls: 'bg-accent-wash text-accent-deep border-accent-edge', icon: ListChecks, spin: false },
  executing: { label: 'Executing', cls: 'bg-accent-wash text-accent-deep border-accent-edge', icon: Loader2, spin: true },
  completed: { label: 'Verified', cls: 'bg-accent-wash text-accent-deep border-accent-edge', icon: CheckCircle2, spin: false },
  completed_with_warnings: { label: 'Delivered with warnings', cls: 'bg-caution-wash text-caution border-caution-edge', icon: ShieldAlert, spin: false },
  'failed-consistency': { label: 'Not cleared', cls: 'bg-danger-wash text-danger border-danger-edge', icon: ShieldAlert, spin: false },
  cancelled: { label: 'Terminated', cls: 'bg-paper text-ink-faint border-edge', icon: XCircle, spin: false },
  failed: { label: 'Execution error', cls: 'bg-caution-wash text-caution border-caution-edge', icon: AlertTriangle, spin: false },
};

export default function Topbar({ viewTitle, viewDesc, currentStatus, wsConnected, isRunning, onReset, onOpenTerminate }) {
  const meta = RUN_STATE_META[currentStatus] || RUN_STATE_META.idle;
  const StatusIcon = meta.icon;

  return (
    <header className="sticky top-0 z-30 bg-paper/90 backdrop-blur border-b border-edge">
      <div className="h-16 px-6 lg:px-8 flex items-center justify-between gap-4">
        {/* View title */}
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold text-ink leading-tight truncate">{viewTitle}</h2>
          <p className="text-[12px] text-ink-faint leading-tight truncate">{viewDesc}</p>
        </div>

        {/* Right controls */}
        <div className="flex items-center gap-2.5 shrink-0">
          {/* Run status pill */}
          <span className={`inline-flex items-center gap-1.5 h-8 px-3 rounded-full border text-[12px] font-medium ${meta.cls}`}>
            {StatusIcon && (
              <StatusIcon className={`w-3.5 h-3.5 ${meta.spin ? 'animate-spin' : ''}`} />
            )}
            {meta.label}
          </span>

          {/* Connection health */}
          <span
            title={wsConnected ? 'Live WebSocket sync with the API' : 'Reconnecting to the API…'}
            className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-full border border-edge bg-card text-[12px]"
          >
            {wsConnected ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-accent animate-blink" />
                <span className="text-ink-soft hidden sm:inline">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-caution" />
                <span className="text-ink-faint hidden sm:inline">Offline</span>
              </>
            )}
          </span>

          {/* Terminate (armed while running) */}
          {isRunning && (
            <button
              onClick={onOpenTerminate}
              className="inline-flex items-center gap-1.5 h-8 px-3.5 rounded-lg bg-danger hover:bg-danger-deep text-white text-[12px] font-semibold transition-colors"
            >
              <Flame className="w-3.5 h-3.5" />
              Terminate
            </button>
          )}

          {/* Reset */}
          <button
            onClick={onReset}
            disabled={isRunning}
            title="Reset sandbox and workflow state"
            className="p-2 rounded-lg border border-edge bg-card text-ink-soft hover:text-ink hover:border-edge-strong transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
