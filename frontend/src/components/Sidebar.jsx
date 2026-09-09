import React from 'react';
import {
  LayoutDashboard, ListChecks, Terminal, ShieldCheck, FileText, Package, Bot
} from 'lucide-react';

const NAV_GROUPS = [
  {
    label: 'Workspace',
    items: [
      { id: 'workspace', label: 'Setup & Run', icon: LayoutDashboard, desc: 'Query, datasets, and run controls' },
    ],
  },
  {
    label: 'Run',
    items: [
      { id: 'plan', label: 'Execution Plan', icon: ListChecks, desc: 'Decomposed agent tasks' },
      { id: 'console', label: 'Live Console', icon: Terminal, desc: 'Agent thoughts and sandbox execution' },
    ],
  },
  {
    label: 'Results',
    items: [
      { id: 'findings', label: 'Verification', icon: ShieldCheck, desc: 'Consistency findings and repair passes' },
      { id: 'report', label: 'Executive Report', icon: FileText, desc: 'Final synthesized deliverable' },
      { id: 'artifacts', label: 'Artifacts', icon: Package, desc: 'Charts, data files, and models' },
    ],
  },
];

function NavBadge({ tone, children }) {
  const tones = {
    danger: 'bg-danger-wash text-danger border-danger-edge',
    neutral: 'bg-paper text-ink-soft border-edge',
    accent: 'bg-accent-wash text-accent border-accent-edge',
  };
  return (
    <span className={`min-w-[20px] h-5 px-1.5 inline-flex items-center justify-center rounded-full border text-[11px] font-semibold tabular-nums ${tones[tone] || tones.neutral}`}>
      {children}
    </span>
  );
}

export default function Sidebar({ activeView, onNavigate, badges, runStatus }) {
  return (
    <aside className="w-60 shrink-0 h-screen sticky top-0 flex flex-col border-r border-edge bg-card">
      {/* Brand */}
      <div className="px-5 pt-5 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-accent flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" strokeWidth={2.2} />
          </div>
          <div>
            <h1 className="text-[15px] font-semibold text-ink leading-tight tracking-tight">AnalystGPT</h1>
            <p className="text-[11px] text-ink-faint leading-tight">Multi-Agent Analytics Studio</p>
          </div>
        </div>
      </div>

      <div className="mx-5 border-t border-edge-soft" />

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
        {NAV_GROUPS.map((group) => (
          <div key={group.label}>
            <p className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
              {group.label}
            </p>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const isActive = activeView === item.id;
                const badge = badges ? badges[item.id] : null;
                return (
                  <button
                    key={item.id}
                    onClick={() => onNavigate(item.id)}
                    className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-colors group ${
                      isActive
                        ? 'bg-accent-wash text-accent-deep'
                        : 'text-ink-soft hover:bg-paper hover:text-ink'
                    }`}
                  >
                    <item.icon
                      className={`w-4 h-4 shrink-0 ${isActive ? 'text-accent' : 'text-ink-faint group-hover:text-ink-soft'}`}
                      strokeWidth={2}
                    />
                    <span className={`flex-1 text-[13px] font-medium ${isActive ? 'font-semibold' : ''}`}>
                      {item.label}
                    </span>
                    {badge?.count > 0 && (
                      <NavBadge tone={badge.tone}>{badge.count}</NavBadge>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer note */}
      <div className="px-5 py-4 border-t border-edge-soft">
        <p className="text-[11px] text-ink-faint leading-relaxed">
          Every run is verified by the numeric gate, cross-artifact checks, and the
          consistency judge before delivery.
        </p>
      </div>
    </aside>
  );
}
