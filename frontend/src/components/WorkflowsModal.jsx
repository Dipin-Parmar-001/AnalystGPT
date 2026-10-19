import React, { useState, useEffect } from 'react';
import { FolderArchive, Play, Trash2, X, RefreshCw, Clock, FileCode, CheckCircle2, PauseCircle, Loader2 } from 'lucide-react';

export default function WorkflowsModal({ isOpen, onClose, onLoadWorkflow, onResumeWorkflow }) {
  const [workflows, setWorkflows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [actionLoadingId, setActionLoadingId] = useState(null);

  const fetchWorkflows = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/workflows');
      if (res.ok) {
        const data = await res.json();
        setWorkflows(data.workflows || []);
      }
    } catch (err) {
      console.error('Failed to fetch saved workflows:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchWorkflows();
    }
  }, [isOpen]);

  const handleDelete = async (workflowId, e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete saved workflow "${workflowId}" from disk?`)) return;

    try {
      const res = await fetch(`/api/workflows/${workflowId}`, { method: 'DELETE' });
      if (res.ok) {
        setWorkflows((prev) => prev.filter((w) => w.workflow_id !== workflowId));
      }
    } catch (err) {
      console.error('Failed to delete workflow:', err);
    }
  };

  const handleLoad = async (workflowId) => {
    setActionLoadingId(workflowId);
    try {
      if (onLoadWorkflow) {
        await onLoadWorkflow(workflowId);
        onClose();
      }
    } finally {
      setActionLoadingId(null);
    }
  };

  const handleResume = async (workflowId) => {
    setActionLoadingId(workflowId);
    try {
      if (onResumeWorkflow) {
        await onResumeWorkflow(workflowId);
        onClose();
      }
    } finally {
      setActionLoadingId(null);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/60 backdrop-blur-sm animate-fade-up">
      <div className="bg-card border border-edge rounded-xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-pop relative overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-edge flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent-wash border border-accent-edge flex items-center justify-center shrink-0">
              <FolderArchive className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h3 className="text-[16px] font-semibold text-ink">Saved Workflows</h3>
              <p className="text-[12px] text-ink-faint">
                Persisted in the <code className="text-accent text-[11px] font-mono">workflows/</code> directory for cross-session resumption
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchWorkflows}
              title="Refresh saved list"
              className="p-1.5 rounded-md text-ink-soft hover:text-ink hover:bg-paper transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md text-ink-faint hover:text-ink hover:bg-paper transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content list */}
        <div className="p-5 overflow-y-auto flex-1 space-y-3.5">
          {loading && workflows.length === 0 ? (
            <div className="py-12 flex flex-col items-center justify-center text-ink-faint gap-2">
              <Loader2 className="w-6 h-6 animate-spin text-accent" />
              <p className="text-[13px]">Scanning workflows/ directory...</p>
            </div>
          ) : workflows.length === 0 ? (
            <div className="py-12 flex flex-col items-center justify-center text-center px-4">
              <div className="w-12 h-12 rounded-full bg-paper border border-edge flex items-center justify-center mb-3">
                <FolderArchive className="w-6 h-6 text-ink-faint" />
              </div>
              <p className="text-[14px] font-medium text-ink">No saved workflows found</p>
              <p className="text-[12px] text-ink-soft max-w-sm mt-1">
                When you pause a running workflow, use the <strong className="text-ink">"Save to Disk"</strong> option in the top bar to save its state so you can resume anytime after server restart.
              </p>
            </div>
          ) : (
            workflows.map((wf) => {
              const isActing = actionLoadingId === wf.workflow_id;
              const formattedDate = wf.saved_at
                ? new Date(wf.saved_at).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit'
                })
                : 'Unknown date';

              return (
                <div
                  key={wf.workflow_id}
                  className="p-4 rounded-xl border border-edge bg-paper/60 hover:bg-paper hover:border-edge-strong transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3.5"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h4 className="text-[13.5px] font-semibold text-ink truncate max-w-md">
                        {wf.name || wf.workflow_id}
                      </h4>
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-500 border border-amber-500/30">
                        <PauseCircle className="w-3 h-3" />
                        Paused at Task #{wf.paused_task_id || 1}
                      </span>
                    </div>

                    <p className="text-[12px] text-ink-soft truncate mt-1">
                      {wf.user_query || 'No query recorded'}
                    </p>

                    <div className="flex items-center gap-3.5 mt-2 text-[11px] text-ink-faint flex-wrap">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formattedDate}
                      </span>
                      <span>
                        Tasks: <strong className="text-ink">{wf.completed_tasks_count || 0}/{wf.total_tasks || 0}</strong> completed
                      </span>
                      {wf.artifacts_count > 0 && (
                        <span>
                          Artifacts: <strong className="text-ink">{wf.artifacts_count}</strong>
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={() => handleResume(wf.workflow_id)}
                      disabled={isActing}
                      className="inline-flex items-center gap-1.5 h-8 px-3 rounded-lg bg-accent hover:bg-accent-deep text-white text-[12px] font-semibold transition-colors shadow-sm disabled:opacity-50"
                    >
                      {isActing ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Play className="w-3.5 h-3.5 fill-current" />
                      )}
                      Resume
                    </button>

                    <button
                      type="button"
                      onClick={() => handleLoad(wf.workflow_id)}
                      disabled={isActing}
                      className="inline-flex items-center gap-1 h-8 px-2.5 rounded-lg border border-edge bg-card hover:bg-paper text-ink text-[12px] font-medium transition-colors disabled:opacity-50"
                    >
                      Load Plan
                    </button>

                    <button
                      type="button"
                      onClick={(e) => handleDelete(wf.workflow_id, e)}
                      disabled={isActing}
                      title="Delete workflow file"
                      className="p-1.5 rounded-lg border border-edge bg-card text-ink-faint hover:text-danger hover:border-danger-edge transition-colors disabled:opacity-50"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-edge bg-paper/40 flex items-center justify-between text-[12px] text-ink-soft">
          <span>{workflows.length} saved workflow{workflows.length === 1 ? '' : 's'} on disk</span>
          <button
            onClick={onClose}
            className="h-8 px-3.5 rounded-lg border border-edge bg-card hover:bg-paper text-ink text-[12px] font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
