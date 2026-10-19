import React from 'react';
import {
  Sparkles, Play, BrainCircuit, ListChecks, CheckCircle2, Loader2, Database
} from 'lucide-react';
import DatasetUploader from '../components/DatasetUploader';

function SectionCard({ title, step, children, aside }) {
  return (
    <section className="bg-card border border-edge rounded-xl shadow-card">
      <header className="flex items-center justify-between px-5 py-3.5 border-b border-edge-soft">
        <div className="flex items-center gap-3">
          {step && (
            <span className="w-6 h-6 rounded-full bg-accent-wash border border-accent-edge text-accent-deep text-[12px] font-semibold flex items-center justify-center">
              {step}
            </span>
          )}
          <h3 className="text-[14px] font-semibold text-ink">{title}</h3>
        </div>
        {aside}
      </header>
      <div className="p-5">{children}</div>
    </section>
  );
}

function StatRow({ label, value, tone }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-edge-soft last:border-0">
      <span className="text-[13px] text-ink-soft">{label}</span>
      <span className={`text-[13px] font-semibold ${tone || 'text-ink'}`}>{value}</span>
    </div>
  );
}

export default function WorkspaceView({
  query, setQuery,
  onGeneratePlan, onExecuteWorkflow,
  isRunning, hasPlan,
  suggestedQueries,
  datasets, selectedFiles, onToggleSelect, onUploadSuccess, onPreviewDataset,
  plan, artifacts, maxToolCalls, setMaxToolCalls, runStatus,
}) {
  const completedTasks = plan ? (plan.tasks || []).filter((t) => t.status === 'completed').length : 0;

  return (
    <div className="max-w-5xl mx-auto animate-fade-up">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left: query + controls */}
        <div className="lg:col-span-7 space-y-5">
          <SectionCard title="Analysis query" step="1" aside={<BrainCircuit className="w-4 h-4 text-ink-faint" />}>
            <form onSubmit={onGeneratePlan} className="space-y-3">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={4}
                disabled={isRunning}
                placeholder="Describe what you want to learn from the data — e.g. clean the churn dataset, explore the drivers of churn, train and compare classification models, and summarise the findings."
                className="w-full px-3.5 py-3 text-[13px] leading-relaxed rounded-lg border border-edge bg-paper/60 text-ink placeholder-ink-faint focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent-ring/40 transition-shadow resize-y disabled:opacity-60"
              />

              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-ink-faint">Suggestions:</span>
                {suggestedQueries.map((sq, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setQuery(sq)}
                    disabled={isRunning}
                    className="max-w-full truncate text-[11px] px-2.5 py-1 rounded-full border border-edge bg-paper hover:border-edge-strong hover:text-ink text-ink-soft transition-colors disabled:opacity-50"
                  >
                    {sq.length > 64 ? sq.slice(0, 64) + '…' : sq}
                  </button>
                ))}
              </div>
            </form>
          </SectionCard>

          <SectionCard
            title="Run controls"
            step="2"
            aside={
              <label className="flex items-center gap-2 text-[12px] text-ink-soft">
                Tool calls per task
                <select
                  value={maxToolCalls}
                  onChange={(e) => setMaxToolCalls(Number(e.target.value))}
                  disabled={isRunning}
                  className="h-7 px-2 rounded-md border border-edge bg-paper text-[12px] text-ink focus:outline-none focus:border-accent"
                >
                  {[5, 7, 10, 12, 15].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
            }
          >
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={onGeneratePlan}
                disabled={isRunning || !query.trim()}
                className="inline-flex items-center gap-2 h-10 px-4 rounded-lg border border-edge-strong bg-card hover:bg-paper text-ink text-[13px] font-medium transition-colors disabled:opacity-40"
              >
                <Sparkles className="w-4 h-4 text-agent-planner" />
                Generate plan
              </button>

              <button
                type="button"
                onClick={onExecuteWorkflow}
                disabled={isRunning || (!query.trim() && !hasPlan)}
                className="inline-flex items-center gap-2 h-10 px-5 rounded-lg bg-accent hover:bg-accent-deep text-white text-[13px] font-semibold transition-colors disabled:opacity-40 shadow-card"
              >
                {isRunning ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isRunning ? 'Running…' : 'Run workflow'}
              </button>

              {!hasPlan && (
                <p className="text-[12px] text-ink-faint">
                  Running without a plan auto-generates one first.
                </p>
              )}
            </div>
          </SectionCard>

          <SectionCard title="Run summary" step="3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8">
              <div>
                <StatRow label="Plan" value={plan ? `#${plan.plan_id}` : '—'} />
                <StatRow label="Tasks completed" value={`${completedTasks} / ${plan ? plan.tasks.length : 0}`} />
                <StatRow label="Datasets selected" value={selectedFiles.length} />
                {(() => {
                  const taskMetrics = (plan?.tasks || []).map((t) => t.primary_metric).filter(Boolean);
                  const metrics = Array.from(new Set([...(plan?.key_metrics || []), ...taskMetrics]));
                  let val = 'NULL';
                  let tone = 'text-ink-faint font-code';

                  if (metrics.length > 0) {
                    val = metrics.join(', ');
                    tone = 'text-accent-deep font-code font-semibold';
                  } else if (plan?.effective_metric) {
                    val = `${plan.effective_metric} (will use)`;
                    tone = 'text-accent-deep font-code font-semibold';
                  } else if ((plan?.tasks || []).some(t => t.task_type?.includes('model'))) {
                    val = (plan.dataset_type === 'regression' ? 'rmse' : 'roc_auc') + ' (will use)';
                    tone = 'text-accent-deep font-code font-semibold';
                  }

                  return (
                    <StatRow
                      label="Key Metric"
                      value={val}
                      tone={tone}
                    />
                  );
                })()}
              </div>
              <div>
                <StatRow label="Artifacts produced" value={artifacts.length} />
                <StatRow
                  label="Run state"
                  value={runStatus === 'executing' ? 'Executing…' : runStatus === 'completed' ? 'Verified' : runStatus || 'Idle'}
                  tone={runStatus === 'completed' ? 'text-accent-deep' : 'text-ink'}
                />
                <StatRow label="Verification" value="Numeric gate · Consistency · Judge" tone="text-[11px] font-medium text-ink-soft" />
              </div>
            </div>
          </SectionCard>
        </div>

        {/* Right: datasets */}
        <div className="lg:col-span-5 space-y-5">
          <SectionCard
            title="Datasets"
            step="1b"
            aside={
              <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-soft">
                <Database className="w-3.5 h-3.5 text-ink-faint" />
                {selectedFiles.length} selected
              </span>
            }
          >
            <DatasetUploader
              datasets={datasets}
              selectedFiles={selectedFiles}
              onToggleSelect={onToggleSelect}
              onUploadSuccess={onUploadSuccess}
              onPreviewDataset={onPreviewDataset}
              disabled={isRunning}
            />
          </SectionCard>

          {plan && (
            <div className="bg-card border border-edge rounded-xl shadow-card p-5">
              <div className="flex items-center gap-2 pb-3 border-b border-edge-soft">
                <ListChecks className="w-4 h-4 text-ink-faint" />
                <h4 className="text-[13px] font-semibold text-ink">Active plan preview</h4>
              </div>
              <ul className="pt-3 space-y-2">
                {plan.tasks.slice(0, 5).map((t) => {
                  const hasModel = t.task_type?.includes('model') || t.task_type === 'hyperparameter_tuning';
                  return (
                    <li key={t.task_id} className="flex items-start justify-between gap-2 text-[12px]">
                      <div className="flex items-start gap-2 min-w-0">
                        {t.status === 'completed' ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-accent mt-0.5 shrink-0" />
                        ) : (
                          <span className="w-3.5 h-3.5 mt-0.5 rounded-full border border-edge-strong shrink-0" />
                        )}
                        <span className="text-ink-soft leading-snug truncate">{t.title}</span>
                      </div>
                      {t.primary_metric ? (
                        <span className="shrink-0 text-[10.5px] font-code px-1.5 py-0.5 rounded bg-accent-wash text-accent-deep border border-accent-edge font-medium">
                          🎯 {t.primary_metric}
                        </span>
                      ) : (t.effective_metric || hasModel) ? (
                        <span className="shrink-0 text-[10.5px] font-code px-1.5 py-0.5 rounded bg-accent-wash/60 text-accent-deep border border-accent-edge font-medium" title="Will use default metric">
                          🎯 {t.effective_metric || 'roc_auc'} <span className="opacity-75 text-[9px] font-sans">(will use)</span>
                        </span>
                      ) : null}
                    </li>
                  );
                })}
                {plan.tasks.length > 5 && (
                  <li className="text-[12px] text-ink-faint pl-5">
                    +{plan.tasks.length - 5} more in Execution Plan
                  </li>
                )}
              </ul>
              <p className="text-[11px] text-ink-faint mt-3">
                Open <span className="text-ink-soft font-medium">Execution Plan</span> in the sidebar to review or modify.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
