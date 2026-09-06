import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, ChevronDown, ChevronRight, AlertOctagon, AlertTriangle } from 'lucide-react';

const CODE_LABELS = {
  shape_mismatch: 'Shape mismatch',
  phantom_columns: 'Phantom columns',
  non_engineered_column_claimed: 'Column mislabeled as engineered',
  imputation_claim_without_missing: 'Imputation claim without missing data',
  unreported_remaining_missing: 'Unreported missing values',
  stale_champion_pointer: 'Stale champion pointer',
  registry_missing_artifact: 'Registry entry missing artifact',
  champion_registry_disagreement: 'Champion/registry disagreement',
  annotated_model_label: 'Mislabeled model',
  duplicate_summaries: 'Duplicate summaries',
  report_references_missing_artifacts: 'Report references missing artifacts',
  unverified_numeric_claim: 'Unverified numeric claim',
  check_error: 'Checker error',
  judge_contradicted_claim: 'Contradicted claim (judge)',
  judge_unverifiable_claim: 'Unverifiable claim (judge)',
};

const SEVERITY_STYLE = {
  critical: {
    icon: <AlertOctagon className="w-4 h-4 text-red-400 shrink-0" />,
    border: 'border-red-800/70',
    bg: 'bg-red-950/30',
    title: 'text-red-300',
  },
  minor: {
    icon: <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />,
    border: 'border-amber-800/60',
    bg: 'bg-amber-950/20',
    title: 'text-amber-300',
  },
};

function FindingCard({ finding, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen);
  const sev = finding.severity === 'critical' ? 'critical' : 'minor';
  const style = SEVERITY_STYLE[sev];
  const label = CODE_LABELS[finding.code] || finding.code || 'finding';

  return (
    <div className={`rounded-xl border ${style.border} ${style.bg} overflow-hidden`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-white/[0.03] transition-colors"
      >
        {style.icon}
        <span className={`text-xs font-bold uppercase tracking-wider font-mono ${style.title}`}>
          [{sev}]
        </span>
        <span className="text-xs font-semibold text-slate-200 flex-1 truncate">
          {label}: {finding.message?.slice(0, 140)}
        </span>
        {open ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
      </button>
      {open && (
        <div className="px-4 pb-3 pt-0 space-y-2">
          <p className="text-xs text-slate-300 leading-relaxed">{finding.message}</p>
          {finding.grounding && (
            <p className="text-[11px] text-slate-400 font-mono">Grounding: {finding.grounding}</p>
          )}
          {finding.artifacts?.length > 0 && (
            <p className="text-[11px] text-slate-500 font-mono">
              Artifacts: {finding.artifacts.map((a, i) => (
                <span key={i} className="inline-block mr-2 text-slate-400">`{a}`</span>
              ))}
            </p>
          )}
          {finding.source && (
            <p className="text-[10px] text-slate-600 font-mono uppercase">source: {finding.source}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function FindingsPanel({ runStatus, severityCounts, findings, repairAttemptsUsed }) {
  if (!runStatus || runStatus === 'idle') return null;
  if (runStatus === 'completed' && (!findings || findings.length === 0)) {
    return (
      <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-800/60 flex items-center gap-3">
        <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
        <div>
          <h4 className="text-xs font-bold text-emerald-300 font-mono uppercase tracking-wider">
            Verification Passed
          </h4>
          <p className="text-xs text-emerald-200/80 mt-0.5">
            All gates passed: numeric verification, cross-artifact consistency, and the LLM consistency judge found no issues.
          </p>
        </div>
      </div>
    );
  }

  const findingsList = findings || [];
  if (findingsList.length === 0 && runStatus !== 'failed-consistency' && runStatus !== 'completed_with_warnings') {
    return null;
  }

  const critical = findingsList.filter((f) => f.severity === 'critical');
  const minor = findingsList.filter((f) => f.severity !== 'critical');

  const statusBanner = () => {
    if (runStatus === 'failed-consistency') {
      return {
        cls: 'bg-red-950/50 border-red-700/80',
        icon: <ShieldAlert className="w-5 h-5 text-red-400 shrink-0" />,
        title: 'Run NOT cleared for delivery (failed-consistency)',
        text:
          'Critical consistency findings survived verification and the automated repair passes. The report below is NOT cleared for delivery — treat every figure as unverified until the findings are resolved.',
        titleCls: 'text-red-300',
      };
    }
    if (runStatus === 'completed_with_warnings') {
      return {
        cls: 'bg-amber-950/40 border-amber-700/70',
        icon: <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0" />,
        title: 'Delivered with warnings (completed_with_warnings)',
        text: 'Minor consistency warnings were detected. The report is usable, but review the findings below.',
        titleCls: 'text-amber-300',
      };
    }
    return null;
  };

  const banner = statusBanner();

  return (
    <div className="border border-surface-border rounded-xl bg-surface-100/70 shadow-xl overflow-hidden">
      <div className="px-4 py-3 bg-surface-200/90 border-b border-surface-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-brand-secondary" />
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
            Verification Findings
          </h3>
        </div>
        <div className="flex items-center gap-2 text-[11px] font-mono">
          <span className="px-2 py-0.5 rounded-full bg-red-950/60 border border-red-800 text-red-300">
            {critical.length} critical
          </span>
          <span className="px-2 py-0.5 rounded-full bg-amber-950/50 border border-amber-800 text-amber-300">
            {minor.length} minor
          </span>
          {typeof repairAttemptsUsed === 'number' && repairAttemptsUsed > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-surface-300/60 border border-surface-border text-slate-300">
              {repairAttemptsUsed} repair {repairAttemptsUsed === 1 ? 'pass' : 'passes'}
            </span>
          )}
        </div>
      </div>

      <div className="p-4 space-y-3">
        {banner && (
          <div className={`p-3 rounded-lg border ${banner.cls} flex items-start gap-3`}>
            {banner.icon}
            <div>
              <h4 className={`text-xs font-bold ${banner.titleCls}`}>{banner.title}</h4>
              <p className="text-xs text-slate-300/90 mt-0.5 leading-relaxed">{banner.text}</p>
            </div>
          </div>
        )}

        {findingsList.length === 0 && (
          <p className="text-xs text-slate-500 font-mono">No findings recorded for this run.</p>
        )}

        {critical.map((f, i) => (
          <FindingCard key={`c-${i}`} finding={f} defaultOpen />
        ))}
        {minor.map((f, i) => (
          <FindingCard key={`m-${i}`} finding={f} defaultOpen={false} />
        ))}
      </div>
    </div>
  );
}
