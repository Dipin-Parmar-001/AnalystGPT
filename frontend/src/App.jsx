import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import DatasetUploader from './components/DatasetUploader';
import PlanViewer from './components/PlanViewer';
import AgentConsole from './components/AgentConsole';
import TerminateModal from './components/TerminateModal';
import StatusAlertBanner from './components/StatusAlertBanner';
import ReportViewer from './components/ReportViewer';
import ArtifactsGallery from './components/ArtifactsGallery';
import DatasetPreviewModal from './components/DatasetPreviewModal';
import FindingsPanel from './components/FindingsPanel';

import {
  Play,
  Sparkles,
  Flame,
  FileText,
  Terminal,
  Box,
  Layers,
  CheckCircle2,
  RefreshCw,
  Send,
  AlertOctagon,
  BrainCircuit,
  Settings2,
  HelpCircle
} from 'lucide-react';

const SUGGESTED_QUERIES = [
  "Perform complete EDA, correlation analysis, and train standard classification/regression models.",
  "Inspect dataset for anomalies, generate statistical visualizations, and compare machine learning models.",
  "Run feature importance ranking, evaluate baseline vs best ML model, and generate an executive report."
];

export default function App() {
  // Navigation & View
  const [activeTab, setActiveTab] = useState('plan_console'); // 'plan_console' | 'report_artifacts'

  // Datasets
  const [datasets, setDatasets] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previewingDataset, setPreviewingDataset] = useState(null);

  // Workflow & State
  const [query, setQuery] = useState('');
  const [plan, setPlan] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('idle'); // 'idle', 'planning', 'plan_ready', 'executing', 'completed', 'cancelled', 'failed'
  const [statusMessage, setStatusMessage] = useState('Ready');
  const [lastError, setLastError] = useState('');
  const [cancelledStep, setCancelledStep] = useState('');
  const [events, setEvents] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [finalReport, setFinalReport] = useState('');
  const [maxToolCalls, setMaxToolCalls] = useState(10);

  // Verification findings (severity-based failure policy)
  const [runStatus, setRunStatus] = useState(null); // null | completed | completed_with_warnings | failed-consistency | failed
  const [severityCounts, setSeverityCounts] = useState(null);
  const [findings, setFindings] = useState([]);
  const [repairAttemptsUsed, setRepairAttemptsUsed] = useState(0);

  // Termination Confirmation Modal
  const [terminateModalOpen, setTerminateModalOpen] = useState(false);
  const [isTerminating, setIsTerminating] = useState(false);

  // WebSocket Connection
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  // Fetch initial datasets and artifacts
  useEffect(() => {
    loadDatasets();
    loadArtifacts();
    loadReport();
    checkWorkflowStatus();
    setupWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, []);

  const setupWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/workflow`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          handleWebSocketMessage(payload);
        } catch (e) {
          console.error("WS Parse error", e);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        // Attempt reconnection after 3 seconds
        reconnectTimerRef.current = setTimeout(() => {
          setupWebSocket();
        }, 3000);
      };

      ws.onerror = () => {
        setWsConnected(false);
      };
    } catch (e) {
      console.warn("WebSocket init error", e);
    }
  };

  const handleWebSocketMessage = (payload) => {
    const { event, data, timestamp } = payload;

    setEvents((prev) => [...prev, payload]);

    if (event === 'plan_created' || event === 'plan_ready' || event === 'plan_modified') {
      const newPlan = data?.plan || payload.plan;
      if (newPlan) {
        setPlan(newPlan);
        setCurrentStatus('plan_ready');
        setStatusMessage(payload.message || 'Execution plan generated.');
      }
    } else if (event === 'plan_execution_started' || event === 'workflow_started') {
      setIsRunning(true);
      setCurrentStatus('executing');
      setStatusMessage('Workflow executing across specialized agents...');
    } else if (event === 'task_execution_start') {
      if (plan && plan.tasks) {
        setPlan((prevPlan) => {
          if (!prevPlan) return prevPlan;
          return {
            ...prevPlan,
            tasks: prevPlan.tasks.map((t) =>
              t.task_id === data.task_id ? { ...t, status: 'in_progress' } : t
            )
          };
        });
      }
    } else if (event === 'task_execution_completed') {
      if (plan && plan.tasks) {
        setPlan((prevPlan) => {
          if (!prevPlan) return prevPlan;
          return {
            ...prevPlan,
            tasks: prevPlan.tasks.map((t) =>
              t.task_id === data.task_id ? { ...t, status: data.status || 'completed' } : t
            )
          };
        });
      }
      loadArtifacts();
    } else if (event === 'report_completed' || event === 'report_generated') {
      if (data?.final_report) {
        setFinalReport(data.final_report);
      } else {
        loadReport();
      }
      loadArtifacts();
    } else if (event === 'workflow_completed') {
      setIsRunning(false);
      // The orchestrator's run status carries the severity policy verdict:
      // completed | completed_with_warnings | failed-consistency | failed
      const verifiedStatus = payload.status || data?.run_status || 'completed';
      setCurrentStatus(verifiedStatus);
      setRunStatus(verifiedStatus);
      setStatusMessage(payload.message || 'Workflow execution completed.');
      if (data?.final_report) setFinalReport(data.final_report);
      if (data?.severity_counts) setSeverityCounts(data.severity_counts);
      if (Array.isArray(data?.findings)) setFindings(data.findings);
      if (typeof data?.repair_attempts_used === 'number') setRepairAttemptsUsed(data.repair_attempts_used);
      loadArtifacts();
      loadReport();
      loadFindings();
    } else if (event === 'workflow_cancelled') {
      setIsRunning(false);
      setCurrentStatus('cancelled');
      setStatusMessage(payload.message || 'Workflow execution stopped by user request.');
      setCancelledStep(payload.step || `Task #${payload.task_id || 'N/A'}`);
      setTerminateModalOpen(false);
      setIsTerminating(false);
      loadArtifacts();
    } else if (event === 'workflow_failed') {
      setIsRunning(false);
      setCurrentStatus('failed');
      setStatusMessage(payload.message || 'Workflow stopped due to execution error.');
      setLastError(payload.error || data?.error || 'Unknown execution error');
      loadArtifacts();
    } else if (event === 'workflow_reset') {
      setIsRunning(false);
      setCurrentStatus('idle');
      setStatusMessage('Workflow reset.');
      setPlan(null);
      setFinalReport('');
      setEvents([]);
      setRunStatus(null);
      setSeverityCounts(null);
      setFindings([]);
      setRepairAttemptsUsed(0);
      loadArtifacts();
    }
  };

  const loadDatasets = async () => {
    try {
      const res = await fetch('/api/datasets');
      const data = await res.json();
      if (data.status === 'success' && data.datasets) {
        setDatasets(data.datasets);
        if (data.datasets.length > 0 && selectedFiles.length === 0) {
          // Select first dataset by default
          setSelectedFiles([data.datasets[0].filename]);
        }
      }
    } catch (err) {
      console.error("Failed to load datasets", err);
    }
  };

  const loadArtifacts = async () => {
    try {
      const res = await fetch('/api/artifacts');
      const data = await res.json();
      if (data.status === 'success' && data.artifacts) {
        setArtifacts(data.artifacts);
      }
    } catch (err) {
      console.error("Failed to load artifacts", err);
    }
  };

  const loadReport = async () => {
    try {
      const res = await fetch('/api/report');
      const data = await res.json();
      if (data.status === 'success' && data.report) {
        setFinalReport(data.report);
      }
      if (data.run_status) {
        setRunStatus(data.run_status);
        if (data.severity_counts) setSeverityCounts(data.severity_counts);
        if (typeof data.repair_attempts_used === 'number') setRepairAttemptsUsed(data.repair_attempts_used);
      }
    } catch (err) {
      console.error("Failed to load report", err);
    }
  };

  const loadFindings = async () => {
    try {
      const res = await fetch('/api/workflow/findings');
      const data = await res.json();
      if (data.status === 'success') {
        if (data.run_status) setRunStatus(data.run_status);
        if (data.counts) setSeverityCounts(data.counts);
        if (Array.isArray(data.findings)) setFindings(data.findings);
        if (typeof data.repair_attempts_used === 'number') setRepairAttemptsUsed(data.repair_attempts_used);
      }
    } catch (err) {
      console.error("Failed to load findings", err);
    }
  };

  const checkWorkflowStatus = async () => {
    try {
      const res = await fetch('/api/workflow/status');
      const data = await res.json();
      if (data) {
        setCurrentStatus(data.status || 'idle');
        setIsRunning(data.is_running || false);
        if (data.plan) setPlan(data.plan);
        if (data.final_report) setFinalReport(data.final_report);
        if (data.message) setStatusMessage(data.message);
        if (data.run_status) {
          setRunStatus(data.run_status);
          if (data.severity_counts) setSeverityCounts(data.severity_counts);
          if (Array.isArray(data.findings)) setFindings(data.findings);
          if (typeof data.repair_attempts_used === 'number') setRepairAttemptsUsed(data.repair_attempts_used);
        } else if (data.final_report) {
          loadFindings();
        }
      }
    } catch (err) {
      console.error("Failed to check status", err);
    }
  };

  const handleToggleSelectFile = (filename) => {
    setSelectedFiles((prev) =>
      prev.includes(filename) ? prev.filter((f) => f !== filename) : [...prev, filename]
    );
  };

  const handleUploadSuccess = (newDataset) => {
    loadDatasets();
    if (newDataset?.filename) {
      setSelectedFiles((prev) => [...prev, newDataset.filename]);
    }
  };

  // Generate Plan Trigger
  const handleGeneratePlan = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim() || isRunning) return;

    setCurrentStatus('planning');
    setStatusMessage('Generating multi-agent execution plan...');
    setLastError('');

    try {
      const res = await fetch('/api/workflow/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          files: selectedFiles,
          max_tool_calls_per_task: maxToolCalls,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Plan generation failed');
      }

      const data = await res.json();
      if (data.plan) {
        setPlan(data.plan);
        setCurrentStatus('plan_ready');
        setStatusMessage(data.message || 'Execution plan ready.');
      }
    } catch (err) {
      setCurrentStatus('failed');
      setStatusMessage(`Plan failed: ${err.message}`);
      setLastError(err.message);
    }
  };

  // Modify Plan Trigger
  const handleModifyPlan = async (modificationPrompt) => {
    try {
      const res = await fetch('/api/workflow/modify-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modification_prompt: modificationPrompt }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to modify plan');
      }

      const data = await res.json();
      if (data.plan) {
        setPlan(data.plan);
        setStatusMessage(data.message || 'Plan revised.');
      }
    } catch (err) {
      alert(`Error modifying plan: ${err.message}`);
    }
  };

  // Execute Workflow Trigger
  const handleExecuteWorkflow = async () => {
    if (isRunning) return;

    setIsRunning(true);
    setCurrentStatus('executing');
    setStatusMessage('Executing multi-agent workflow...');
    setLastError('');
    setCancelledStep('');

    try {
      const res = await fetch('/api/workflow/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query || (plan ? plan.query : ''),
          files: selectedFiles,
          max_tool_calls_per_agent: maxToolCalls,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Execution start failed');
      }
    } catch (err) {
      setIsRunning(false);
      setCurrentStatus('failed');
      setStatusMessage(`Execution failed to start: ${err.message}`);
      setLastError(err.message);
    }
  };

  // Immediate Workflow Termination Handlers
  const handleOpenTerminateModal = () => {
    setTerminateModalOpen(true);
  };

  const handleConfirmTerminate = async () => {
    setIsTerminating(true);
    try {
      const res = await fetch('/api/workflow/terminate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'Terminated by user via confirmation modal' }),
      });

      const data = await res.json();
      setIsRunning(false);
      setCurrentStatus('cancelled');
      setStatusMessage(data.message || 'Workflow terminated.');
      setTerminateModalOpen(false);
      setIsTerminating(false);
      loadArtifacts();
    } catch (err) {
      console.error("Failed to terminate", err);
      setIsTerminating(false);
    }
  };

  // Reset Sandbox & Workflow
  const handleResetWorkflow = async () => {
    if (window.confirm("Reset sandbox state and current workflow?")) {
      try {
        await fetch('/api/workflow/reset', { method: 'POST' });
        setPlan(null);
        setEvents([]);
        setFinalReport('');
        setCurrentStatus('idle');
        setStatusMessage('System reset to idle.');
        loadArtifacts();
      } catch (err) {
        console.error("Reset error", err);
      }
    }
  };

  return (
    <div className="min-h-screen bg-background text-slate-100 flex flex-col selection:bg-brand-primary/30">
      {/* Top Navbar */}
      <Navbar
        wsConnected={wsConnected}
        isRunning={isRunning}
        currentStatus={currentStatus}
        onReset={handleResetWorkflow}
      />

      {/* Main App Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* Explicit Status Notification Banner (Completed / Terminated / Error) */}
        <StatusAlertBanner
          status={currentStatus}
          message={statusMessage}
          error={lastError}
          cancelledStep={cancelledStep}
        />

        {/* Top Query & Action Bar */}
        <section className="p-5 rounded-2xl bg-surface-100/90 border border-surface-border glass-panel shadow-2xl relative overflow-hidden">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-brand-primary" />
                Multi-Agent Analysis Query
              </label>

              {selectedFiles.length > 0 && (
                <span className="text-[11px] font-mono text-brand-secondary bg-surface-200 px-2.5 py-0.5 rounded-full border border-surface-border">
                  Targeting: {selectedFiles.join(', ')}
                </span>
              )}
            </div>

            <form onSubmit={handleGeneratePlan} className="flex flex-col sm:flex-row gap-3">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. Conduct full EDA, evaluate missing values, build classification models, and output key metrics..."
                  disabled={isRunning}
                  className="w-full px-4 py-3 text-sm rounded-xl bg-surface-200/90 border border-surface-border text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-primary focus:ring-1 focus:ring-brand-primary transition-all disabled:opacity-50"
                />
              </div>

              <div className="flex items-center gap-2">
                {/* Plan button */}
                <button
                  type="submit"
                  disabled={isRunning || !query.trim()}
                  className="px-4 py-3 rounded-xl text-xs font-bold bg-surface-200 hover:bg-surface-hover border border-surface-border text-slate-200 flex items-center gap-2 transition-all disabled:opacity-40"
                >
                  <Sparkles className="w-4 h-4 text-brand-secondary" />
                  Plan Tasks
                </button>

                {/* Execute Button */}
                {!isRunning ? (
                  <button
                    type="button"
                    onClick={handleExecuteWorkflow}
                    disabled={isRunning || (!query.trim() && !plan)}
                    className="px-5 py-3 rounded-xl text-xs font-bold bg-gradient-to-r from-brand-primary to-indigo-600 hover:from-brand-primaryHover hover:to-indigo-500 text-white shadow-lg shadow-brand-primary/20 flex items-center gap-2 transition-all disabled:opacity-40"
                  >
                    <Play className="w-4 h-4" />
                    Run Workflow
                  </button>
                ) : (
                  /* CRITICAL TERMINATE WORKFLOW BUTTON */
                  <button
                    type="button"
                    onClick={handleOpenTerminateModal}
                    className="px-5 py-3 rounded-xl text-xs font-bold bg-gradient-to-r from-red-600 to-rose-700 hover:from-red-500 hover:to-rose-600 text-white shadow-lg shadow-red-900/40 flex items-center gap-2 transition-all animate-pulse"
                  >
                    <Flame className="w-4 h-4" />
                    Terminate Workflow
                  </button>
                )}
              </div>
            </form>

            {/* Query suggestions */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-[11px] text-slate-500 font-mono">Suggestions:</span>
              {SUGGESTED_QUERIES.map((sq, sIdx) => (
                <button
                  key={sIdx}
                  type="button"
                  onClick={() => setQuery(sq)}
                  disabled={isRunning}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-surface-200/60 hover:bg-surface-200 text-slate-400 hover:text-slate-200 border border-surface-border/60 transition-colors truncate max-w-xs"
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Split Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left Column: Datasets & Workspace Controls (4 cols) */}
          <div className="lg:col-span-4 space-y-6">
            {/* Dataset Uploader & Selector */}
            <div className="p-5 rounded-2xl bg-surface-100/90 border border-surface-border glass-panel shadow-lg">
              <DatasetUploader
                datasets={datasets}
                selectedFiles={selectedFiles}
                onToggleSelect={handleToggleSelectFile}
                onUploadSuccess={handleUploadSuccess}
                onPreviewDataset={(ds) => setPreviewingDataset(ds)}
                disabled={isRunning}
              />
            </div>

            {/* Quick Stats & Telemetry */}
            <div className="p-4 rounded-2xl bg-surface-100/70 border border-surface-border text-xs text-slate-300 space-y-2 font-mono">
              <div className="flex items-center justify-between text-slate-400">
                <span>Active Plan ID:</span>
                <span className="text-slate-200 font-semibold">{plan?.plan_id || 'None'}</span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Completed Tasks:</span>
                <span className="text-slate-200 font-semibold">
                  {plan ? plan.tasks.filter((t) => t.status === 'completed').length : 0} / {plan?.tasks.length || 0}
                </span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Artifacts Produced:</span>
                <span className="text-slate-200 font-semibold">{artifacts.length}</span>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Max Tool Calls/Agent:</span>
                <span className="text-slate-200 font-semibold">{maxToolCalls}</span>
              </div>
            </div>
          </div>

          {/* Right Column: Execution Workspace & Deliverables (8 cols) */}
          <div className="lg:col-span-8 space-y-4">
            
            {/* Tab Navigation */}
            <div className="flex items-center justify-between border-b border-surface-border pb-2">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTab('plan_console')}
                  className={`px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all ${
                    activeTab === 'plan_console'
                      ? 'bg-brand-primary text-white shadow-md shadow-brand-primary/20'
                      : 'bg-surface-100 text-slate-400 hover:text-slate-200 border border-surface-border'
                  }`}
                >
                  <Terminal className="w-3.5 h-3.5" />
                  Plan & Live Agent Stream
                </button>

                <button
                  onClick={() => setActiveTab('report_artifacts')}
                  className={`px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all ${
                    activeTab === 'report_artifacts'
                      ? 'bg-brand-primary text-white shadow-md shadow-brand-primary/20'
                      : 'bg-surface-100 text-slate-400 hover:text-slate-200 border border-surface-border'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  Executive Report & Artifacts
                  {artifacts.length > 0 && (
                    <span className="w-5 h-5 rounded-full bg-surface-300 text-[10px] flex items-center justify-center font-mono">
                      {artifacts.length}
                    </span>
                  )}
                </button>
              </div>

              {/* In-running Emergency Terminate Button on Tab Bar */}
              {isRunning && (
                <button
                  onClick={handleOpenTerminateModal}
                  className="px-3 py-1.5 rounded-lg text-xs font-bold bg-red-950/80 hover:bg-red-900 border border-red-700 text-red-300 flex items-center gap-1.5 transition-all animate-pulse"
                >
                  <AlertOctagon className="w-3.5 h-3.5 text-red-400" />
                  Stop Execution
                </button>
              )}
            </div>

            {/* Tab 1: Plan & Agent Console */}
            {activeTab === 'plan_console' && (
              <div className="space-y-6 animate-in fade-in">
                {/* Plan Viewer */}
                <PlanViewer
                  plan={plan}
                  isRunning={isRunning}
                  onExecutePlan={handleExecuteWorkflow}
                  onModifyPlan={handleModifyPlan}
                  disabled={isRunning}
                />

                {/* Agent Live Console */}
                <AgentConsole
                  events={events}
                  isRunning={isRunning}
                />
              </div>
            )}

            {/* Tab 2: Executive Report & Artifacts */}
            {activeTab === 'report_artifacts' && (
              <div className="space-y-6 animate-in fade-in">
                {/* Verification Findings (severity-based failure policy) */}
                <FindingsPanel
                  runStatus={runStatus}
                  severityCounts={severityCounts}
                  findings={findings}
                  repairAttemptsUsed={repairAttemptsUsed}
                />

                {/* Report Viewer */}
                <ReportViewer
                  reportText={finalReport}
                />

                {/* Artifacts Gallery */}
                <ArtifactsGallery
                  artifacts={artifacts}
                />
              </div>
            )}

          </div>

        </div>

      </main>

      {/* Two-Step Termination Confirmation Modal */}
      <TerminateModal
        isOpen={terminateModalOpen}
        onClose={() => setTerminateModalOpen(false)}
        onConfirm={handleConfirmTerminate}
        terminating={isTerminating}
      />

      {/* Dataset Preview Modal */}
      <DatasetPreviewModal
        dataset={previewingDataset}
        onClose={() => setPreviewingDataset(null)}
      />
    </div>
  );
}
