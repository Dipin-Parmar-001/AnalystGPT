import React, { useState, useRef } from 'react';
import { Upload, FileSpreadsheet, Eye, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

export default function DatasetUploader({
  datasets,
  selectedFiles,
  onToggleSelect,
  onUploadSuccess,
  onPreviewDataset,
  disabled
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await processUpload(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await processUpload(file);
    }
  };

  const processUpload = async (file) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Only CSV files (.csv) are supported.');
      return;
    }

    setUploadError('');
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      const result = await response.json();
      if (onUploadSuccess) {
        onUploadSuccess(result.data);
      }
    } catch (err) {
      setUploadError(err.message || 'Failed to upload CSV');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`border border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
          isDragging
            ? 'border-accent bg-accent-wash'
            : 'border-edge-strong bg-paper hover:bg-edge-soft hover:border-ink-faint/50'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".csv"
          className="hidden"
          disabled={disabled}
        />

        <div className="flex flex-col items-center gap-2">
          <div className="w-10 h-10 rounded-full bg-card border border-edge flex items-center justify-center">
            {uploading ? (
              <Loader2 className="w-5 h-5 text-accent animate-spin" />
            ) : (
              <Upload className="w-5 h-5 text-ink-soft" strokeWidth={1.8} />
            )}
          </div>
          <div>
            <p className="text-[13px] font-medium text-ink">
              {uploading ? 'Parsing & ingesting CSV…' : 'Click or drag a CSV here'}
            </p>
            <p className="text-[11.5px] text-ink-faint mt-0.5">Standard tabular CSV files</p>
          </div>
        </div>
      </div>

      {uploadError && (
        <div className="px-3 py-2.5 rounded-lg bg-danger-wash border border-danger-edge text-[12px] text-danger flex items-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Dataset list */}
      <div className="space-y-1.5 max-h-64 overflow-y-auto pr-0.5">
        {datasets.length === 0 ? (
          <p className="text-[12px] text-ink-faint text-center py-4">
            No datasets uploaded yet.
          </p>
        ) : (
          datasets.map((ds) => {
            const isSelected = selectedFiles.includes(ds.filename);
            return (
              <div
                key={ds.filename}
                className={`flex items-center justify-between gap-2 p-2.5 rounded-lg border transition-colors ${
                  isSelected
                    ? 'bg-accent-wash/60 border-accent-edge'
                    : 'bg-card border-edge hover:border-edge-strong'
                }`}
              >
                <div
                  className="flex items-center gap-2.5 flex-1 min-w-0 cursor-pointer"
                  onClick={() => onToggleSelect(ds.filename)}
                >
                  <span
                    className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors ${
                      isSelected ? 'bg-accent border-accent text-white' : 'border-edge-strong bg-card'
                    }`}
                  >
                    {isSelected && <CheckCircle2 className="w-3 h-3" />}
                  </span>

                  <FileSpreadsheet className="w-4 h-4 text-ink-faint shrink-0" />

                  <div className="min-w-0">
                    <p className="text-[12.5px] font-medium text-ink truncate">{ds.filename}</p>
                    <p className="text-[11px] text-ink-faint tabular-nums">
                      {ds.row_count !== undefined ? `${ds.row_count.toLocaleString()} rows · ${ds.col_count} cols · ${formatBytes(ds.size_bytes)}` : formatBytes(ds.size_bytes)}
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => onPreviewDataset(ds)}
                  title="Inspect dataset columns and rows"
                  className="p-1.5 rounded-md text-ink-faint hover:text-ink hover:bg-paper transition-colors shrink-0"
                >
                  <Eye className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
