import React from 'react';
import { Bot, Activity, Wifi, WifiOff, RefreshCw, Layers, ShieldCheck, Zap } from 'lucide-react';

export default function Navbar({ wsConnected, isRunning, currentStatus, onReset }) {
  const getStatusBadge = () => {
    switch (currentStatus) {
      case 'executing':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-950/80 text-indigo-400 border border-indigo-700/60 animate-pulse-subtle">
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-ping"></span>
            Executing Pipeline
          </span>
        );
      case 'planning':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-purple-950/80 text-purple-300 border border-purple-700/60">
            <span className="w-2 h-2 rounded-full bg-purple-400"></span>
            Planning Tasks
          </span>
        );
      case 'plan_ready':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-sky-950/80 text-sky-300 border border-sky-700/60">
            <span className="w-2 h-2 rounded-full bg-sky-400"></span>
            Plan Ready
          </span>
        );
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-700/60">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            Workflow Completed
          </span>
        );
      case 'cancelled':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-red-950/80 text-red-300 border border-red-700/60">
            <span className="w-2 h-2 rounded-full bg-red-400"></span>
            Workflow Terminated
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-700/60">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            Execution Error
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700">
            <span className="w-2 h-2 rounded-full bg-slate-500"></span>
            System Idle
          </span>
        );
    }
  };

  return (
    <header className="sticky top-0 z-40 border-b border-surface-border glass-panel">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Left Branding */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-primary to-brand-secondary flex items-center justify-center shadow-lg shadow-brand-primary/20">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-100 to-slate-400 tracking-tight">
                AnalystGPT
              </h1>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-brand-primary/20 text-indigo-300 border border-brand-primary/30 font-bold tracking-wider">
                v2.0 Multi-Agent
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Collaborative Data Science & ML Engine
            </p>
          </div>
        </div>

        {/* Center: Agents Active Indicators */}
        <div className="hidden md:flex items-center gap-2 text-xs font-mono">
          <span className="px-2.5 py-1 rounded-md bg-indigo-900/40 text-indigo-300 border border-indigo-800/50 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
            Planner
          </span>
          <span className="px-2.5 py-1 rounded-md bg-emerald-900/40 text-emerald-300 border border-emerald-800/50 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            Analyst
          </span>
          <span className="px-2.5 py-1 rounded-md bg-fuchsia-900/40 text-fuchsia-300 border border-fuchsia-800/50 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400"></span>
            ML Modeler
          </span>
          <span className="px-2.5 py-1 rounded-md bg-amber-900/40 text-amber-300 border border-amber-800/50 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
            Reporter
          </span>
        </div>

        {/* Right Status & Actions */}
        <div className="flex items-center gap-3">
          {/* Workflow Status */}
          {getStatusBadge()}

          {/* WebSocket Health */}
          <div className="flex items-center text-xs text-slate-400 gap-1.5 px-2.5 py-1 rounded-lg bg-surface-100 border border-surface-border">
            {wsConnected ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                <span className="hidden sm:inline text-emerald-400 font-mono">Live Sync</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-amber-400" />
                <span className="hidden sm:inline text-amber-400 font-mono">Connecting...</span>
              </>
            )}
          </div>

          {/* Reset button */}
          <button
            onClick={onReset}
            disabled={isRunning}
            title="Reset sandbox & workflow state"
            className="p-2 rounded-lg bg-surface-100 hover:bg-surface-hover border border-surface-border text-slate-400 hover:text-slate-200 transition-all disabled:opacity-40"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

      </div>
    </header>
  );
}
