import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import DatasetUploader from './components/DatasetUploader';
import PlanViewer from './components/PlanViewer';
import AgentConsole from './components/AgentConsole';
import TerminateModal from './components/TerminateModal';
import StatusAlertBanner from './components/StatusAlertBanner';
import ReportViewer from './components/ReportViewer';
import ArtifactsGallery from './components/ArtifactsGallery';
import DatasetPreviewModal from './components/DatasetPreviewModal';
import FindingsPanel from './components/FindingsPanel';
import WorkflowsModal from './components/WorkflowsModal';
import WorkspaceView from './views/WorkspaceView';

import {
  Sparkles, Play, BrainCircuit, ListChecks, Terminal, ShieldCheck, FileText, Package, LayoutDashboard
} from 'lucide-react';

const SUGGESTED_QUERIES = [
  "Perform complete EDA, correlation analysis, and train standard classification/regression models.",
  "Inspect dataset for anomalies, generate statistical visualizations, and compare machine learning models.",
  "Run feature importance ranking, evaluate baseline vs best ML model, and generate an executive report."
];

const VIEW_META = {
  workspace: { title: 'Setup & Run', desc: 'Describe the analysis, choose datasets, and start the run', icon: LayoutDashboard },
  plan: { title: 'Execution Plan', desc: 'The PlannerAgent decomposition — review, modify, then execute', icon: ListChecks },
  console: { title: 'Live Console', desc: 'Real-time agent reasoning and sandbox execution telemetry', icon: Terminal },
  findings: { title: 'Verification', desc: 'Numeric gate, cross-artifact consistency, and the LLM judge', icon: ShieldCheck },
  report: { title: 'Executive Report', desc: 'The final verified deliverable', icon: FileText },
  artifacts: { title: 'Artifacts', desc: 'Charts, datasets, and model files produced by the run', icon: Package },
};

export default function App() {
  // Navigation & View
  const [activeView, setActiveView] = useState('workspace');

  // Datasets
  const [datasets, setDatasets] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previewingDataset, setPreviewingDataset] = useState(null);

  // Workflow & State
  const [query, setQuery] = useState('');
  const [plan, setPlan] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('idle');
  const [statusMessage, setStatusMessage] = useState('Ready');
  const [lastError, setLastError] = useState('');
  const [cancelledStep, setCancelledStep] = useState('');
  const [events, setEvents] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [finalReport, setFinalReport] = useState('');
  const [maxToolCalls, setMaxToolCalls] = useState(10);

  // Verification findings (severity-based failure policy)
  const [runStatus, setRunStatus] = useState(null);
  const [severityCounts, setSeverityCounts] = useState(null);
  const [findings, setFindings] = useState([]);
  const [repairAttemptsUsed, setRepairAttemptsUsed] = useState(0);

  // Termination Confirmation Modal
  const [terminateModalOpen, setTerminateModalOpen] = useState(false);
  const [isTerminating, setIsTerminating] = useState(false);

  // Saved Workflows Modal & Paused Checkpoint
  const [workflowsModalOpen, setWorkflowsModalOpen] = useState(false);
  const [pausedCheckpoint, setPausedCheckpoint] = useState(null);

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

  const navigate = (view) => setActiveView(view);

  const handleWebSocketMessage = (payload) => {
    const { event, data, timestamp } = payload;

    setEvents((prev) => [...prev, payload]);

    if (event === 'plan_created' || event === 'plan_ready' || event === 'plan_modified') {
      const newPlan = data?.plan || payload.plan;
      if (newPlan) {
        setPlan(newPlan);
        setCurrentStatus('plan_ready');
        setStatusMessage(payload.message || 'Execution plan generated.');
        navigate('plan');
      }
    } else if (event === 'plan_execution_started' || event === 'workflow_started') {
      setIsRunning(true);
      setCurrentStatus('executing');
      setStatusMessage('Workflow executing across specialized agents...');
      navigate('console');
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
      // Route the user to the outcome: failed verification -> findings, else report
      navigate(verifiedStatus === 'failed-consistency' ? 'findings' : 'report');
    } else if (event === 'workflow_pausing') {
      setIsRunning(true);
      setCurrentStatus('pausing');
      setStatusMessage(payload.message || data?.message || 'Pausing workflow... waiting for active step to halt.');
      const cp = payload.checkpoint || data?.checkpoint;
      if (cp) setPausedCheckpoint(cp);
    } else if (event === 'workflow_paused') {
      setIsRunning(false);
      setCurrentStatus('paused');
      setStatusMessage(payload.message || data?.message || 'Workflow paused by user.');
      const cp = payload.checkpoint || data?.checkpoint;
      if (cp) setPausedCheckpoint(cp);
      if (cp?.plan) setPlan(cp.plan);
      loadArtifacts();
    } else if (event === 'workflow_resumed') {
      setIsRunning(true);
      setCurrentStatus('executing');
      const resumeTid = payload.resume_from_task || data?.resume_from_task;
      setStatusMessage(`Resumed execution from Task #${resumeTid || '...'}`);
      if (resumeTid) {
        setPlan((prevPlan) => {
          if (!prevPlan || !prevPlan.tasks) return prevPlan;
          return {
            ...prevPlan,
            tasks: prevPlan.tasks.map((t) => {
              if (t.task_id < resumeTid) {
                return { ...t, status: 'completed' };
              } else if (t.task_id === resumeTid) {
                return { ...t, status: 'in_progress' };
              } else {
                return { ...t, status: 'pending' };
              }
            })
          };
        });
      }
      navigate('console');
    } else if (event === 'workflow_loaded') {
      setIsRunning(false);
      setCurrentStatus('paused');
      setStatusMessage(payload.message || `Loaded workflow ${payload.workflow_id}`);
      if (payload.plan) setPlan(payload.plan);
      if (payload.checkpoint) setPausedCheckpoint(payload.checkpoint);
      navigate('plan');
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
      navigate('workspace');
    }
  };

  const loadDatasets = async () => {
    try {
      const res = await fetch('/api/datasets');
      const data = await res.json();
      if (data.status === 'success' && data.datasets) {
        setDatasets(data.datasets);
        if (data.datasets.length > 0 && selectedFiles.length === 0) {
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
        if (data.last_pause_checkpoint) {
          setPausedCheckpoint(data.last_pause_checkpoint);
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

  // Pause Workflow Trigger
  const handlePause = async () => {
    try {
      setStatusMessage('Pausing workflow...');
      const res = await fetch('/api/workflow/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'Paused by user via dashboard' }),
      });
      const data = await res.json();
      if (data.status === 'paused') {
        setIsRunning(false);
        setCurrentStatus('paused');
        setStatusMessage(data.message || 'Workflow paused.');
        if (data.checkpoint) {
          setPausedCheckpoint(data.checkpoint);
        }
      } else if (data.status === 'pausing') {
        setIsRunning(true);
        setCurrentStatus('pausing');
        setStatusMessage(data.message || 'Pausing workflow... waiting for active step to halt.');
        if (data.checkpoint) {
          setPausedCheckpoint(data.checkpoint);
        }
      }
    } catch (err) {
      console.error("Failed to pause workflow", err);
    }
  };

  // Resume Workflow Trigger
  const handleResume = async () => {
    if (isRunning && currentStatus !== 'paused') return;

    setIsRunning(true);
    setCurrentStatus('executing');
    setStatusMessage('Resuming workflow execution...');
    setLastError('');

    try {
      const res = await fetch('/api/workflow/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_tool_calls_per_agent: maxToolCalls,
          resume_from_task_id: pausedCheckpoint?.paused_task_id || undefined,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Resume failed to start');
      }
      navigate('console');
    } catch (err) {
      setIsRunning(false);
      setCurrentStatus('paused');
      setStatusMessage(`Resume failed: ${err.message}`);
      setLastError(err.message);
    }
  };

  // Save Paused Workflow to Disk Trigger (in workflows/ folder)
  const handleSaveWorkflow = async () => {
    try {
      const res = await fetch('/api/workflow/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: query || (plan ? plan.query : 'Saved Workflow'),
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to save workflow');
      }

      const data = await res.json();
      setStatusMessage(data.message || `Saved to workflows/${data.filename}`);
      alert(`Workflow successfully saved to disk (workflows/${data.filename})!\nYou can now close or restart the system and continue anytime.`);
    } catch (err) {
      alert(`Error saving workflow to disk: ${err.message}`);
    }
  };

  // Load Saved Workflow from Disk
  const handleLoadWorkflow = async (workflowId) => {
    try {
      const res = await fetch(`/api/workflows/${workflowId}/load`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to load workflow');
      }
      const data = await res.json();
      if (data.plan) setPlan(data.plan);
      if (data.checkpoint) setPausedCheckpoint(data.checkpoint);
      setCurrentStatus('paused');
      setStatusMessage(data.message || `Loaded workflow ${workflowId}`);
      navigate('plan');
    } catch (err) {
      alert(`Error loading workflow: ${err.message}`);
    }
  };

  // Load and Resume Saved Workflow from Disk
  const handleResumeLoadedWorkflow = async (workflowId) => {
    try {
      const res = await fetch(`/api/workflows/${workflowId}/load`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to load workflow');
      }
      const data = await res.json();
      if (data.plan) setPlan(data.plan);
      if (data.checkpoint) setPausedCheckpoint(data.checkpoint);
      setCurrentStatus('paused');

      setTimeout(() => {
        handleResume();
      }, 250);
    } catch (err) {
      alert(`Error resuming loaded workflow: ${err.message}`);
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
        setPausedCheckpoint(null);
        setRunStatus(null);
        setSeverityCounts(null);
        setFindings([]);
        setRepairAttemptsUsed(0);
        loadArtifacts();
        navigate('workspace');
      } catch (err) {
        console.error("Reset error", err);
      }
    }
  };

  // Sidebar badges
  const completedTasks = plan ? (plan.tasks || []).filter((t) => t.status === 'completed').length : 0;
  const badges = {
    plan: plan ? { count: (plan.tasks || []).length, tone: 'neutral' } : null,
    console: isRunning ? { count: events.length, tone: 'accent' } : null,
    findings: severityCounts && (severityCounts.critical > 0 || severityCounts.minor > 0)
      ? { count: severityCounts.critical + severityCounts.minor, tone: severityCounts.critical > 0 ? 'danger' : 'neutral' }
      : null,
    artifacts: artifacts.length > 0 ? { count: artifacts.length, tone: 'neutral' } : null,
  };

  const meta = VIEW_META[activeView] || VIEW_META.workspace;

  return (
    <div className="flex min-h-screen">
      <Sidebar
        activeView={activeView}
        onNavigate={navigate}
        badges={badges}
        runStatus={currentStatus}
      />

      {/* Main column */}
      <div className="flex-1 min-w-0 flex flex-col">
        <Topbar
          viewTitle={meta.title}
          viewDesc={meta.desc}
          currentStatus={currentStatus}
          wsConnected={wsConnected}
          isRunning={isRunning}
          onReset={handleResetWorkflow}
          onOpenTerminate={handleOpenTerminateModal}
          onPause={handlePause}
          onResume={handleResume}
          onSaveWorkflow={handleSaveWorkflow}
          onOpenWorkflows={() => setWorkflowsModalOpen(true)}
        />

        {/* Global status banner */}
        <div className="px-6 lg:px-8 pt-5">
          <StatusAlertBanner
            status={currentStatus}
            message={statusMessage}
            error={lastError}
            cancelledStep={cancelledStep}
          />
        </div>

        <main className="flex-1 px-6 lg:px-8 py-6">
          {/* ---------- Workspace ---------- */}
          {activeView === 'workspace' && (
            <WorkspaceView
              query={query}
              setQuery={setQuery}
              onGeneratePlan={handleGeneratePlan}
              onExecuteWorkflow={handleExecuteWorkflow}
              isRunning={isRunning}
              hasPlan={!!plan}
              suggestedQueries={SUGGESTED_QUERIES}
              datasets={datasets}
              selectedFiles={selectedFiles}
              onToggleSelect={handleToggleSelectFile}
              onUploadSuccess={handleUploadSuccess}
              onPreviewDataset={setPreviewingDataset}
              plan={plan}
              artifacts={artifacts}
              maxToolCalls={maxToolCalls}
              setMaxToolCalls={setMaxToolCalls}
              runStatus={currentStatus}
            />
          )}

          {/* ---------- Plan ---------- */}
          {activeView === 'plan' && (
            <div className="max-w-4xl mx-auto animate-fade-up">
              <PlanViewer
                plan={plan}
                isRunning={isRunning}
                onExecutePlan={handleExecuteWorkflow}
                onModifyPlan={handleModifyPlan}
                disabled={isRunning}
              />
            </div>
          )}

          {/* ---------- Console ---------- */}
          {activeView === 'console' && (
            <div className="max-w-5xl mx-auto animate-fade-up">
              <AgentConsole events={events} isRunning={isRunning} />
            </div>
          )}

          {/* ---------- Findings ---------- */}
          {activeView === 'findings' && (
            <div className="max-w-4xl mx-auto animate-fade-up">
              <FindingsPanel
                runStatus={runStatus || (currentStatus === 'completed' ? 'completed' : null)}
                severityCounts={severityCounts}
                findings={findings}
                repairAttemptsUsed={repairAttemptsUsed}
              />
            </div>
          )}

          {/* ---------- Report ---------- */}
          {activeView === 'report' && (
            <div className="max-w-3xl mx-auto animate-fade-up">
              <ReportViewer reportText={finalReport} />
            </div>
          )}

          {/* ---------- Artifacts ---------- */}
          {activeView === 'artifacts' && (
            <div className="max-w-5xl mx-auto animate-fade-up">
              <ArtifactsGallery artifacts={artifacts} />
            </div>
          )}
        </main>
      </div>

      {/* Two-Step Termination Confirmation Modal */}
      <TerminateModal
        isOpen={terminateModalOpen}
        onClose={() => setTerminateModalOpen(false)}
        onConfirm={handleConfirmTerminate}
        terminating={isTerminating}
      />

      {/* Saved Workflows Directory Modal */}
      <WorkflowsModal
        isOpen={workflowsModalOpen}
        onClose={() => setWorkflowsModalOpen(false)}
        onLoadWorkflow={handleLoadWorkflow}
        onResumeWorkflow={handleResumeLoadedWorkflow}
      />

      {/* Dataset Preview Modal */}
      <DatasetPreviewModal
        dataset={previewingDataset}
        onClose={() => setPreviewingDataset(null)}
      />
    </div>
  );
}
