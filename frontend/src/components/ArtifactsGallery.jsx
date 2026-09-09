import React, { useState } from 'react';
import { Image, FileSpreadsheet, FileCode, Download, Eye, ExternalLink, Package } from 'lucide-react';

const TYPE_META = {
  image: { icon: Image, cls: 'text-agent-ml bg-agent-ml/10 border-agent-ml/25', label: 'Chart' },
  dataset: { icon: FileSpreadsheet, cls: 'text-agent-analyst bg-agent-analyst/10 border-agent-analyst/25', label: 'Dataset' },
  model: { icon: Package, cls: 'text-agent-planner bg-agent-planner/10 border-agent-planner/25', label: 'Model' },
  report: { icon: FileCode, cls: 'text-agent-reporter bg-agent-reporter/10 border-agent-reporter/25', label: 'Report' },
  file: { icon: FileCode, cls: 'text-ink-soft bg-paper border-edge', label: 'File' },
  json: { icon: FileCode, cls: 'text-ink-soft bg-paper border-edge', label: 'JSON' },
};

export default function ArtifactsGallery({ artifacts }) {
  const [selectedImage, setSelectedImage] = useState(null);

  if (!artifacts || artifacts.length === 0) {
    return (
      <div className="p-10 text-center border border-dashed border-edge-strong rounded-xl bg-card/60">
        <Package className="w-8 h-8 text-ink-faint/50 mx-auto mb-3" strokeWidth={1.6} />
        <p className="text-[14px] font-medium text-ink-soft">No artifacts yet</p>
        <p className="text-[13px] text-ink-faint mt-1">
          Charts, model comparison tables, and metric summaries produced by the run will be listed here.
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
      <div className="flex items-center justify-between">
        <h3 className="text-[15px] font-semibold text-ink">Generated artifacts</h3>
        <span className="text-[12px] text-ink-faint">{artifacts.length} files</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {artifacts.map((art, idx) => {
          const isImg = art.type === 'image';
          const isCsv = art.type === 'dataset';
          const meta = TYPE_META[art.type] || TYPE_META.file;
          const Icon = meta.icon;

          return (
            <div
              key={idx}
              className="flex flex-col justify-between bg-card border border-edge rounded-xl shadow-card hover:shadow-pop transition-shadow overflow-hidden"
            >
              {/* Thumbnail zone for images */}
              {isImg ? (
                <button
                  onClick={() => setSelectedImage(art.relative_url)}
                  className="relative h-36 bg-paper border-b border-edge-soft overflow-hidden group/img block w-full"
                >
                  <img
                    src={art.relative_url}
                    alt={art.filename}
                    className="w-full h-full object-contain p-2 transition-transform group-hover/img:scale-[1.03]"
                    loading="lazy"
                  />
                  <span className="absolute inset-0 bg-ink/0 group-hover/img:bg-ink/10 transition-colors flex items-center justify-center">
                    <span className="opacity-0 group-hover/img:opacity-100 transition-opacity inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-ink text-paper text-[11px] font-medium">
                      <Eye className="w-3.5 h-3.5" /> Expand
                    </span>
                  </span>
                </button>
              ) : (
                <div className="h-1.5 bg-paper border-b border-edge-soft" />
              )}

              <div className="p-4 flex-1 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[10.5px] font-semibold uppercase tracking-wide ${meta.cls}`}>
                    <Icon className="w-3 h-3" />
                    {meta.label}
                  </span>
                  <span className="text-[11px] text-ink-faint tabular-nums">{formatBytes(art.size_bytes)}</span>
                </div>

                <p className="text-[12.5px] font-medium text-ink truncate" title={art.filename}>
                  {art.filename}
                </p>

                {isCsv && art.preview && art.preview.length > 0 && (
                  <p className="mt-1 text-[11px] text-ink-faint">
                    {art.preview.length} preview rows captured
                  </p>
                )}

                <div className="mt-3 pt-3 border-t border-edge-soft flex items-center justify-between">
                  <a
                    href={art.relative_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[12px] text-accent-deep hover:text-accent font-medium inline-flex items-center gap-1 transition-colors"
                  >
                    <ExternalLink className="w-3 h-3" />
                    Open
                  </a>
                  <a
                    href={art.relative_url}
                    download={art.filename}
                    className="inline-flex items-center gap-1 h-7 px-2.5 rounded-md border border-edge bg-paper hover:bg-edge-soft text-[11px] font-medium text-ink-soft transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    Download
                  </a>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Lightbox */}
      {selectedImage && (
        <div
          onClick={() => setSelectedImage(null)}
          className="fixed inset-0 z-50 p-6 bg-ink/80 flex items-center justify-center cursor-pointer animate-fade-up"
        >
          <div className="relative max-w-4xl max-h-[85vh] bg-card p-2 rounded-xl shadow-pop border border-edge overflow-hidden">
            <img
              src={selectedImage}
              alt="Visualization zoom"
              className="max-w-full max-h-[80vh] object-contain rounded-lg"
            />
            <p className="text-center text-[11px] text-ink-faint py-2">Click anywhere outside to close</p>
          </div>
        </div>
      )}
    </div>
  );
}
