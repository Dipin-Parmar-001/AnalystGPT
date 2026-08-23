import React, { useState, useRef } from 'react';
import { Upload, FileSpreadsheet, Eye, CheckCircle2, AlertCircle, HardDrive, Trash2 } from 'lucide-react';

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
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <HardDrive className="w-4 h-4 text-brand-secondary" />
          Dataset Workspace
        </h3>
        <span className="text-xs text-slate-400 font-mono">
          {datasets.length} available
        </span>
      </div>

      {/* Drag and Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? 'border-brand-primary bg-brand-primary/10 shadow-lg glow-primary'
            : 'border-slate-700/80 bg-surface-100/60 hover:bg-surface-100 hover:border-slate-600'
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

        <div className="flex flex-col items-center justify-center gap-2">
          <div className="w-10 h-10 rounded-full bg-brand-primary/10 text-brand-primary flex items-center justify-center">
            {uploading ? (
              <div className="w-5 h-5 border-2 border-brand-primary border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Upload className="w-5 h-5 text-indigo-400" />
            )}
          </div>
          <div>
            <p className="text-xs font-medium text-slate-200">
              {uploading ? 'Parsing & Ingesting CSV...' : 'Click or drag CSV dataset here'}
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Supports standard tabular CSV files
            </p>
          </div>
        </div>
      </div>

      {uploadError && (
        <div className="p-2.5 rounded-lg bg-red-950/60 border border-red-800/80 text-xs text-red-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Dataset List */}
      <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
        {datasets.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-4 italic">
            No datasets uploaded yet. Upload a CSV to get started.
          </p>
        ) : (
          datasets.map((ds) => {
            const isSelected = selectedFiles.includes(ds.filename);
            return (
              <div
                key={ds.filename}
                className={`p-3 rounded-lg border transition-all flex items-center justify-between ${
                  isSelected
                    ? 'bg-brand-primary/10 border-brand-primary/60 text-slate-100'
                    : 'bg-surface-100/80 border-surface-border hover:bg-surface-hover text-slate-300'
                }`}
              >
                <div
                  className="flex items-center gap-2.5 flex-1 min-w-0 cursor-pointer"
                  onClick={() => onToggleSelect(ds.filename)}
                >
                  <div
                    className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                      isSelected
                        ? 'bg-brand-primary border-brand-primary text-white'
                        : 'border-slate-600 bg-surface-200'
                    }`}
                  >
                    {isSelected && <CheckCircle2 className="w-3.5 h-3.5" />}
                  </div>

                  <FileSpreadsheet className="w-4 h-4 text-brand-secondary shrink-0" />

                  <div className="truncate">
                    <p className="text-xs font-medium text-slate-200 truncate">
                      {ds.filename}
                    </p>
                    <p className="text-[10px] text-slate-400 font-mono">
                      {ds.row_count !== undefined ? `${ds.row_count.toLocaleString()} rows • ${ds.col_count} cols` : 'Ready'} • {formatBytes(ds.size_bytes)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0 ml-2">
                  <button
                    onClick={() => onPreviewDataset(ds)}
                    title="Inspect dataset columns and rows"
                    className="p-1.5 rounded-md hover:bg-surface-200 text-slate-400 hover:text-brand-secondary transition-colors"
                  >
                    <Eye className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
