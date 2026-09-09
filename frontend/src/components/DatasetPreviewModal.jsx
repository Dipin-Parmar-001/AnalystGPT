import React from 'react';
import { X, FileSpreadsheet, Hash } from 'lucide-react';

export default function DatasetPreviewModal({ dataset, onClose }) {
  if (!dataset) return null;

  const columns = dataset.columns || (dataset.preview && dataset.preview.length > 0 ? Object.keys(dataset.preview[0]) : []);
  const rows = dataset.preview || [];

  return (
    <div className="fixed inset-0 z-50 p-4 bg-ink/60 flex items-center justify-center animate-fade-up">
      <div className="bg-card border border-edge rounded-xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-pop overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 border-b border-edge-soft flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-agent-analyst/10 border border-agent-analyst/25 flex items-center justify-center">
              <FileSpreadsheet className="w-[18px] h-[18px] text-agent-analyst" />
            </div>
            <div>
              <h3 className="text-[14px] font-semibold text-ink">{dataset.filename}</h3>
              <p className="text-[12px] text-ink-faint tabular-nums">
                {dataset.row_count !== undefined ? `${dataset.row_count.toLocaleString()} rows` : ''} · {columns.length} columns
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-md text-ink-faint hover:text-ink hover:bg-paper transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Schema chips */}
        <div className="px-5 py-3 border-b border-edge-soft flex flex-wrap gap-1.5 overflow-x-auto max-h-24 bg-paper/60">
          {columns.map((col, idx) => (
            <span
              key={idx}
              className="inline-flex items-center gap-1 text-[11px] font-code px-2 py-0.5 rounded-md bg-card border border-edge text-ink-soft"
            >
              <Hash className="w-3 h-3 text-ink-faint" />
              {col}
            </span>
          ))}
        </div>

        {/* Data table */}
        <div className="flex-1 overflow-auto p-5">
          {rows.length === 0 ? (
            <p className="text-[13px] text-ink-faint text-center py-10">No sample rows available for preview.</p>
          ) : (
            <div className="border border-edge rounded-lg overflow-hidden">
              <table className="w-full text-left text-[12px] border-collapse">
                <thead>
                  <tr className="bg-paper border-b border-edge text-ink-soft font-semibold">
                    <th className="p-2.5 text-center text-ink-faint w-12">#</th>
                    {columns.map((col) => (
                      <th key={col} className="p-2.5 whitespace-nowrap">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge-soft">
                  {rows.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-paper/70 text-ink-soft">
                      <td className="p-2.5 text-center text-ink-faint text-[11px] tabular-nums">{rIdx + 1}</td>
                      {columns.map((col) => (
                        <td key={col} className="p-2.5 whitespace-nowrap truncate max-w-xs">
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-edge-soft flex items-center justify-between text-[12px] text-ink-faint">
          <span>Showing sample preview rows</span>
          <button
            onClick={onClose}
            className="h-8 px-4 rounded-lg border border-edge-strong bg-card hover:bg-paper text-ink font-medium text-[12px] transition-colors"
          >
            Close preview
          </button>
        </div>
      </div>
    </div>
  );
}
