import React from 'react';
import { X, FileSpreadsheet, Layers, Hash } from 'lucide-react';

export default function DatasetPreviewModal({ dataset, onClose }) {
  if (!dataset) return null;

  const columns = dataset.columns || (dataset.preview && dataset.preview.length > 0 ? Object.keys(dataset.preview[0]) : []);
  const rows = dataset.preview || [];

  return (
    <div className="fixed inset-0 z-50 p-4 bg-black/80 backdrop-blur-sm flex items-center justify-center animate-in fade-in">
      <div className="bg-surface-100 border border-surface-border rounded-2xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 bg-surface-200/90 border-b border-surface-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-950/80 border border-emerald-800 flex items-center justify-center">
              <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                {dataset.filename}
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                {dataset.row_count !== undefined ? `${dataset.row_count.toLocaleString()} rows` : ''} • {columns.length} columns
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-surface-100 hover:bg-surface-hover text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Column schema tags */}
        <div className="px-6 py-3 bg-surface-200/40 border-b border-surface-border flex flex-wrap gap-1.5 overflow-x-auto max-h-24">
          {columns.map((col, idx) => (
            <span
              key={idx}
              className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded-md bg-surface-300 border border-surface-border text-slate-300"
            >
              <Hash className="w-3 h-3 text-brand-secondary" />
              {col}
            </span>
          ))}
        </div>

        {/* Data Table */}
        <div className="flex-1 overflow-auto p-6">
          {rows.length === 0 ? (
            <p className="text-xs text-slate-400 text-center py-8 italic">
              No sample rows available for preview.
            </p>
          ) : (
            <div className="border border-surface-border rounded-xl overflow-hidden">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="bg-surface-200 border-b border-surface-border text-slate-300 font-semibold">
                    <th className="p-2.5 text-center text-slate-500 w-12">#</th>
                    {columns.map((col) => (
                      <th key={col} className="p-2.5 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-border bg-surface-100/60">
                  {rows.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-surface-hover/50 text-slate-300">
                      <td className="p-2.5 text-center text-slate-500 text-[10px]">{rIdx + 1}</td>
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
        <div className="px-6 py-3 bg-surface-200/90 border-t border-surface-border flex items-center justify-between text-xs text-slate-400">
          <span>Showing sample preview rows</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-surface-100 hover:bg-surface-hover text-slate-200 font-medium text-xs transition-colors"
          >
            Close Preview
          </button>
        </div>
      </div>
    </div>
  );
}
