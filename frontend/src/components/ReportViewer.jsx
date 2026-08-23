import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, Copy, Check, Download, ExternalLink, Code } from 'lucide-react';

export default function ReportViewer({ reportText }) {
  const [copied, setCopied] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  const handleCopy = () => {
    if (!reportText) return;
    navigator.clipboard.writeText(reportText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!reportText) return;
    const blob = new Blob([reportText], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `analystgpt_report_${new Date().toISOString().slice(0, 10)}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!reportText) {
    return (
      <div className="p-8 text-center border border-surface-border rounded-xl bg-surface-100/40">
        <FileText className="w-8 h-8 text-slate-500 mx-auto mb-2 opacity-50" />
        <p className="text-sm font-medium text-slate-400">No Report Generated Yet</p>
        <p className="text-xs text-slate-500 mt-1">
          Execute a workflow to synthesize findings into an executive markdown report.
        </p>
      </div>
    );
  }

  return (
    <div className="border border-surface-border rounded-xl bg-surface-100/70 shadow-xl overflow-hidden flex flex-col">
      {/* Header Controls */}
      <div className="px-4 py-3 bg-surface-200/90 border-b border-surface-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-brand-secondary" />
          <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
            Executive Analysis Report
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowRaw(!showRaw)}
            className="px-2.5 py-1 rounded-lg text-xs font-medium bg-surface-100 hover:bg-surface-hover border border-surface-border text-slate-300 flex items-center gap-1 transition-colors"
          >
            <Code className="w-3.5 h-3.5 text-slate-400" />
            {showRaw ? 'Rendered' : 'Raw Markdown'}
          </button>

          <button
            onClick={handleCopy}
            className="px-2.5 py-1 rounded-lg text-xs font-medium bg-surface-100 hover:bg-surface-hover border border-surface-border text-slate-300 flex items-center gap-1 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>

          <button
            onClick={handleDownload}
            className="px-2.5 py-1 rounded-lg text-xs font-medium bg-brand-primary/20 hover:bg-brand-primary/30 border border-brand-primary/40 text-indigo-300 flex items-center gap-1 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Download .md
          </button>
        </div>
      </div>

      {/* Report Content Body */}
      <div className="p-6 overflow-y-auto max-h-[600px] text-slate-200">
        {showRaw ? (
          <pre className="font-mono text-xs text-slate-300 whitespace-pre-wrap bg-surface-200 p-4 rounded-lg border border-surface-border">
            {reportText}
          </pre>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {reportText}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
