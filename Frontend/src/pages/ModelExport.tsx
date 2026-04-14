import { useEffect, useState } from 'react';

interface ModelExportStatus {
  status: string;
  progress: number;
  current_step: string;
  logs: string[];
  error?: string | null;
}

interface ModelExportResults {
  export_name: string;
  export_directory: string;
  archive_path: string;
  exported_files: string[];
  num_classes: number;
  class_labels: string[];
}

export default function ModelExport() {
  const [isExporting, setIsExporting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<ModelExportResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isExporting) {
      interval = setInterval(fetchStatus, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isExporting]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/model-export/status');
      const data: ModelExportStatus = await response.json();

      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setLogs(data.logs || []);

      if (data.status === 'completed') {
        setIsExporting(false);
        await fetchResults();
      } else if (data.status === 'error') {
        setIsExporting(false);
        setError(data.error || 'Model export failed. Please check the logs.');
      }
    } catch (err) {
      console.error('Error fetching export status:', err);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/model-export/results');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error fetching export results:', err);
      setError('Failed to fetch export results.');
    }
  };

  const startExport = async () => {
    setIsExporting(true);
    setProgress(0);
    setCurrentStep('Initializing...');
    setLogs([]);
    setResults(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/model-export/start', {
        method: 'POST',
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || 'Failed to start model export');
      }
    } catch (err) {
      console.error('Error starting model export:', err);
      setError(err instanceof Error ? err.message : 'Failed to start model export.');
      setIsExporting(false);
    }
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Model Export</h1>
          <p className="text-gray-600">
            Create an inference-ready export package so another backend can load your trained weights and predict.
          </p>
        </div>

        <div className="bg-white border-2 border-cyan-200 rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">What Gets Exported</h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-4">
              <p className="text-sm font-semibold text-blue-900 mb-1">Feature Extractor</p>
              <p className="text-sm text-blue-800">Shared CNN+Transformer weights for MRI and PET feature extraction.</p>
            </div>
            <div className="rounded-lg bg-purple-50 border border-purple-200 p-4">
              <p className="text-sm font-semibold text-purple-900 mb-1">Classifier Head</p>
              <p className="text-sm text-purple-800">The trained head that predicts the output class from combined features.</p>
            </div>
            <div className="rounded-lg bg-cyan-50 border border-cyan-200 p-4">
              <p className="text-sm font-semibold text-cyan-900 mb-1">Metadata</p>
              <p className="text-sm text-cyan-800">Architecture config, class labels, and inference notes for the other backend.</p>
            </div>
            <div className="rounded-lg bg-green-50 border border-green-200 p-4">
              <p className="text-sm font-semibold text-green-900 mb-1">Zip Archive</p>
              <p className="text-sm text-green-800">A portable package you can copy directly into another application.</p>
            </div>
          </div>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <button
            onClick={startExport}
            disabled={isExporting}
            className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
              isExporting
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-cyan-600 text-white hover:bg-cyan-700'
            }`}
          >
            {isExporting ? 'Exporting Model Package...' : 'Start Model Export'}
          </button>
        </div>

        {(isExporting || progress > 0) && (
          <div className="bg-white border-2 border-cyan-200 rounded-lg p-6 mb-6 shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-2xl font-bold text-gray-900">{currentStep || 'Model Export'}</h2>
              <span className="text-2xl font-bold text-cyan-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div
                className="bg-cyan-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="bg-gray-900 text-green-400 rounded-lg p-5 mb-6 shadow-xl border-2 border-gray-700">
            <div className="flex items-center mb-3 pb-2 border-b border-gray-700">
              <span className="text-white font-mono text-sm font-semibold">Model Export Logs</span>
              <span className="ml-auto text-gray-500 text-xs">{logs.length} entries</span>
            </div>
            <div className="font-mono text-xs max-h-80 overflow-y-auto space-y-1">
              {logs.map((log, index) => (
                <div key={index} className="hover:bg-gray-800 px-2 py-1 rounded transition-colors">
                  <span className="text-gray-500 mr-2">[{index + 1}]</span>
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 mb-6">
            <p className="text-red-800 font-semibold">{error}</p>
          </div>
        )}

        {results && (
          <div className="space-y-6">
            <div className="bg-cyan-600 rounded-lg shadow-lg p-6">
              <h2 className="text-3xl font-bold text-white mb-2">Export Complete</h2>
              <p className="text-cyan-100">Your inference package is ready for another backend to load.</p>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Export Name</p>
                <p className="text-lg font-bold text-blue-600 break-all">{results.export_name}</p>
              </div>
              <div className="bg-white border-2 border-purple-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Classes</p>
                <p className="text-4xl font-bold text-purple-600">{results.num_classes}</p>
              </div>
              <div className="bg-white border-2 border-green-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Archive</p>
                <p className="text-sm font-bold text-green-600 break-all">{results.archive_path}</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Export Locations</h3>
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Folder</p>
                    <p className="text-sm font-mono text-gray-800 break-all">{results.export_directory}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Zip</p>
                    <p className="text-sm font-mono text-gray-800 break-all">{results.archive_path}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Class Labels</h3>
                <div className="space-y-3">
                  {results.class_labels.map((label, index) => (
                    <div key={label} className="flex items-center justify-between bg-cyan-50 rounded-lg p-3 border border-cyan-200">
                      <span className="text-sm font-medium text-gray-700">Class {index}</span>
                      <span className="text-sm font-bold text-cyan-800">{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Exported Files</h3>
              <div className="space-y-3">
                {results.exported_files.map((filePath) => (
                  <div key={filePath} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <p className="text-sm font-mono text-gray-800 break-all">{filePath}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
