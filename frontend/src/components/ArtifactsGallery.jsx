import React, { useState } from 'react';
import { Image, FileSpreadsheet, FileCode, Download, Eye, ExternalLink, Box } from 'lucide-react';

export default function ArtifactsGallery({ artifacts }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewCsv, setPreviewCsv] = useState(null);

  if (!artifacts || artifacts.length === 0) {
    return (
      <div className="p-8 text-center border border-surface-border rounded-xl bg-surface-100/40">
        <Box className="w-8 h-8 text-slate-500 mx-auto mb-2 opacity-50" />
        <p className="text-sm font-medium text-slate-400">No Artifacts Generated Yet</p>
        <p className="text-xs text-slate-500 mt-1">
          Visualizations (PNG), model comparisons (CSV), and metric summaries will be listed here.
        </p>
      </div>
    );
  }

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono flex items-center gap-2">
          <Box className="w-4 h-4 text-brand-secondary" />
          Generated Deliverables & Artifacts ({artifacts.length})
        </h3>
      </div>

      {/* Grid of artifacts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
        {artifacts.map((art, idx) => {
          const isImg = art.type === 'image';
          const isCsv = art.type === 'dataset';
          const isModel = art.type === 'model';
          const isReport = art.type === 'report';

          return (
            <div
              key={idx}
              className="p-3 rounded-xl bg-surface-100/80 border border-surface-border hover:border-slate-600 transition-all flex flex-col justify-between group shadow-sm hover:shadow-md"
            >
              {/* Top info */}
              <div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    {isImg && <Image className="w-4 h-4 text-sky-400" />}
                    {isCsv && <FileSpreadsheet className="w-4 h-4 text-emerald-400" />}
                    {isModel && <Box className="w-4 h-4 text-fuchsia-400" />}
                    {isReport && <FileCode className="w-4 h-4 text-amber-400" />}
                    {!isImg && !isCsv && !isModel && !isReport && <FileCode className="w-4 h-4 text-slate-400" />}
                    
                    <span className="text-[10px] uppercase font-mono font-bold px-1.5 py-0.5 rounded bg-surface-200 text-slate-400 border border-surface-border">
                      {art.type}
                    </span>
                  </div>

                  <span className="text-[10px] text-slate-500 font-mono">
                    {formatBytes(art.size_bytes)}
                  </span>
                </div>

                <p className="text-xs font-semibold text-slate-200 truncate" title={art.filename}>
                  {art.filename}
                </p>

                {/* If image, show small thumbnail */}
                {isImg && (
                  <div
                    onClick={() => setSelectedImage(art.relative_url)}
                    className="mt-2.5 h-32 rounded-lg bg-black/40 border border-surface-border overflow-hidden cursor-pointer relative group/img"
                  >
                    <img
                      src={art.relative_url}
                      alt={art.filename}
                      className="w-full h-full object-contain p-1 transition-transform group-hover/img:scale-105"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/img:opacity-100 transition-opacity flex items-center justify-center text-white text-xs font-medium gap-1">
                      <Eye className="w-4 h-4" /> Expand Chart
                    </div>
                  </div>
                )}

                {/* If CSV preview rows available */}
                {isCsv && art.preview && art.preview.length > 0 && (
                  <div className="mt-2 text-[10px] text-slate-400 font-mono">
                    <span className="text-emerald-400 font-semibold">{art.preview.length}</span> preview rows captured
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="mt-3 pt-2 border-t border-surface-border/60 flex items-center justify-between">
                <a
                  href={art.relative_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] text-brand-secondary hover:underline flex items-center gap-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  Open
                </a>

                <a
                  href={art.relative_url}
                  download={art.filename}
                  className="px-2 py-1 rounded bg-surface-200 hover:bg-surface-hover text-slate-300 text-[11px] flex items-center gap-1 transition-colors"
                >
                  <Download className="w-3 h-3" />
                  Download
                </a>
              </div>
            </div>
          );
        })}
      </div>

      {/* Image Modal Preview */}
      {selectedImage && (
        <div
          onClick={() => setSelectedImage(null)}
          className="fixed inset-0 z-50 p-6 bg-black/85 backdrop-blur-sm flex items-center justify-center cursor-pointer animate-in fade-in"
        >
          <div className="relative max-w-4xl max-h-[85vh] bg-surface-100 p-2 rounded-2xl border border-surface-border shadow-2xl overflow-hidden">
            <img
              src={selectedImage}
              alt="Visualization Zoom"
              className="max-w-full max-h-[80vh] object-contain rounded-xl"
            />
            <p className="text-center text-xs text-slate-400 py-2 font-mono">
              Click anywhere outside to close
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
