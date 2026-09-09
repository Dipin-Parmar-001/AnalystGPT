import React, { useState } from 'react';
import { ShieldCheck, ChevronDown, ChevronRight, AlertOctagon, AlertTriangle, Wrench } from 'lucide-react';

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
    icon: AlertOctagon,
    chip: 'bg-danger-wash text-danger border-danger-edge',
    card: 'border-danger-edge hover:border-danger/50',
    label: 'text-danger',
  },
  minor: {
    icon: AlertTriangle,
    chip: 'bg-caution-wash text-caution border-caution-edge',
    card: 'border-caution-edge hover:border-caution/50',
    label: 'text-caution',
  },
};

function FindingCard({ finding, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen);
  const sev = finding.severity === 'critical' ? 'critical' : 'minor';
  const style = SEVERITY_STYLE[sev];
  const Icon = style.icon;
  const label = CODE_LABELS[finding.code] || finding.code || 'finding';

  return (
    <div className={`rounded-lg border bg-card shadow-card transition-colors ${style.card}`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-3 flex items-center gap-3 text-left"
      >
        <Icon className={`w-4 h-4 shrink-0 ${sev === 'critical' ? 'text-danger' : 'text-caution'}`} />
        <span className={`text-[11px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded border ${style.chip}`}>
          {sev}
        </span>
        <span className="text-[13px] font-medium text-ink flex-1 truncate">
          {label}
          <span className="text-ink-faint font-normal"> — {finding.message?.slice(0, 110)}</span>
        </span>
        {open ? <ChevronDown className="w-4 h-4 text-ink-faint" /> : <ChevronRight className="w-4 h-4 text-ink-faint" />}
      </button>
      {open && (
        <div className="px-4 pb-4 pt-0 space-y-2.5 border-t border-edge-soft pt-3 mx-4 mb-1 pl-4">
          <p className="text-[13px] text-ink-soft leading-relaxed">{finding.message}</p>
          {finding.grounding && (
            <p className="text-[12px] text-ink-faint">
              <span className="font-medium text-ink-soft">Grounding:</span> {finding.grounding}
            </p>
          )}
          {finding.artifacts?.length > 0 && (
            <p className="text-[12px] text-ink-faint">
              <span className="font-medium text-ink-soft">Artifacts:</span>{' '}
              {finding.artifacts.map((a, i) => (
                <code key={i} className="font-code text-[11px] bg-paper border border-edge rounded px-1 py-0.5 mr-1.5">{a}</code>
              ))}
            </p>
          )}
          {finding.source && (
            <p className="text-[11px] text-ink-faint">Source: {finding.source}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function FindingsPanel({ runStatus, severityCounts, findings, repairAttemptsUsed }) {
  if (!runStatus || runStatus === 'idle') {
    return (
      <div className="p-10 text-center border border-dashed border-edge-strong rounded-xl bg-card/60">
        <ShieldCheck className="w-8 h-8 text-ink-faint/50 mx-auto mb-3" strokeWidth={1.6} />
        <p className="text-[14px] font-medium text-ink-soft">No verification results yet</p>
        <p className="text-[13px] text-ink-faint mt-1">
          Findings appear here after a run finishes its verification passes.
        </p>
      </div>
    );
  }

  const findingsList = findings || [];
  const critical = findingsList.filter((f) => f.severity === 'critical');
  const minor = findingsList.filter((f) => f.severity !== 'critical');

  // Fully clean run
  if (runStatus === 'completed' && findingsList.length === 0) {
    return (
      <div className="p-6 rounded-xl bg-accent-wash border border-accent-edge flex items-start gap-4">
        <div className="w-10 h-10 rounded-lg bg-accent text-white flex items-center justify-center shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <h4 className="text-[15px] font-semibold text-accent-deep">Report cleared for delivery</h4>
          <p className="text-[13px] text-ink-soft mt-1 leading-relaxed">
            All verification gates passed — the numeric check, cross-artifact consistency, and the LLM
            consistency judge found no issues.
          </p>
        </div>
      </div>
    );
  }

  const banner = () => {
    if (runStatus === 'failed-consistency') {
      return (
        <div className="p-4 rounded-lg bg-danger-wash border border-danger-edge">
          <h4 className="text-[14px] font-semibold text-danger">Not cleared for delivery</h4>
          <p className="text-[13px] text-ink-soft mt-1 leading-relaxed">
            Critical findings survived verification and {repairAttemptsUsed || 'the automated'} repair
            {repairAttemptsUsed === 1 ? ' pass' : ' passes'}. The report is shown for diagnostics only —
            treat every figure as unverified until these findings are resolved.
          </p>
        </div>
      );
    }
    if (runStatus === 'completed_with_warnings') {
      return (
        <div className="p-4 rounded-lg bg-caution-wash border border-caution-edge">
          <h4 className="text-[14px] font-semibold text-caution">Delivered with warnings</h4>
          <p className="text-[13px] text-ink-soft mt-1 leading-relaxed">
            The report is usable, but review the {minor.length} minor finding{minor.length === 1 ? '' : 's'} below.
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-[15px] font-semibold text-ink">Verification findings</h3>
          <p className="text-[12px] text-ink-faint mt-0.5">
            Detected by the numeric gate, cross-artifact consistency checks, and the consistency judge.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {typeof repairAttemptsUsed === 'number' && repairAttemptsUsed > 0 && (
            <span className="inline-flex items-center gap-1 h-7 px-2.5 rounded-full border border-edge bg-card text-[11px] font-medium text-ink-soft">
              <Wrench className="w-3 h-3" />
              {repairAttemptsUsed} repair {repairAttemptsUsed === 1 ? 'pass' : 'passes'}
            </span>
          )}
          <span className="inline-flex items-center h-7 px-2.5 rounded-full border border-danger-edge bg-danger-wash text-[11px] font-semibold text-danger">
            {critical.length} critical
          </span>
          <span className="inline-flex items-center h-7 px-2.5 rounded-full border border-caution-edge bg-caution-wash text-[11px] font-semibold text-caution">
            {minor.length} minor
          </span>
        </div>
      </div>

      {banner()}

      {findingsList.length === 0 ? (
        <div className="p-8 text-center border border-dashed border-edge-strong rounded-xl bg-card/60">
          <p className="text-[13px] text-ink-faint">No findings recorded for this run.</p>
        </div>
      ) : (
        <div className="space-y-2.5">
          {critical.map((f, i) => (
            <FindingCard key={`c-${i}`} finding={f} defaultOpen />
          ))}
          {minor.map((f, i) => (
            <FindingCard key={`m-${i}`} finding={f} defaultOpen={false} />
          ))}
        </div>
      )}
    </div>
  );
}
