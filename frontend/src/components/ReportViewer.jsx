import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, Copy, Check, Download, Code } from 'lucide-react';

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
      <div className="p-10 text-center border border-dashed border-edge-strong rounded-xl bg-card/60">
        <FileText className="w-8 h-8 text-ink-faint/50 mx-auto mb-3" strokeWidth={1.6} />
        <p className="text-[14px] font-medium text-ink-soft">No report yet</p>
        <p className="text-[13px] text-ink-faint mt-1">
          Once a run completes, the verified executive report appears here in a readable document layout.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card border border-edge rounded-xl shadow-card overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-edge-soft flex items-center justify-between bg-card sticky top-16 z-10">
        <div className="flex items-center gap-2.5">
          <FileText className="w-4 h-4 text-ink-faint" />
          <h3 className="text-[13px] font-semibold text-ink">Executive Report</h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowRaw(!showRaw)}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md border border-edge bg-paper hover:bg-edge-soft text-[11px] font-medium text-ink-soft transition-colors"
          >
            <Code className="w-3 h-3" />
            {showRaw ? 'Rendered' : 'Raw'}
          </button>

          <button
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md border border-edge bg-paper hover:bg-edge-soft text-[11px] font-medium text-ink-soft transition-colors"
          >
            {copied ? <Check className="w-3 h-3 text-accent" /> : <Copy className="w-3 h-3" />}
            {copied ? 'Copied' : 'Copy'}
          </button>

          <button
            onClick={handleDownload}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md bg-accent hover:bg-accent-deep text-[11px] font-medium text-white transition-colors"
          >
            <Download className="w-3 h-3" />
            Download .md
          </button>
        </div>
      </div>

      {/* Document body */}
      <div className="px-8 lg:px-12 py-8 overflow-y-auto max-h-[calc(100vh-220px)]">
        {showRaw ? (
          <pre className="font-code text-[12px] text-ink-soft whitespace-pre-wrap bg-paper border border-edge p-4 rounded-lg">
            {reportText}
          </pre>
        ) : (
          <div className="markdown-body max-w-[68ch]">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {reportText}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
