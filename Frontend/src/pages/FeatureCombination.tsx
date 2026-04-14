import { useEffect, useState } from 'react';

interface CombinationStatus {
  status: string;
  progress: number;
  current_step: string;
  logs: string[];
  error?: string | null;
}

interface CombinationResults {
  paired_samples: number;
  mri_feature_dimension: number;
  pet_feature_dimension: number;
  combined_feature_dimension: number;
  output_file: string;
  combination_time: number;
  combined_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
}

export default function FeatureCombination() {
  const [isCombining, setIsCombining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<CombinationResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isCombining) {
      interval = setInterval(fetchStatus, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isCombining]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/feature-combination/status');
      const data: CombinationStatus = await response.json();

      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setLogs(data.logs || []);

      if (data.status === 'completed') {
        setIsCombining(false);
        await fetchResults();
      } else if (data.status === 'error') {
        setIsCombining(false);
        setError(data.error || 'Feature combination failed. Please check the logs.');
      }
    } catch (err) {
      console.error('Error fetching combination status:', err);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/feature-combination/results');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error fetching combination results:', err);
      setError('Failed to fetch combination results.');
    }
  };

  const startCombination = async () => {
    setIsCombining(true);
    setProgress(0);
    setCurrentStep('Initializing...');
    setLogs([]);
    setResults(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/feature-combination/start', {
        method: 'POST',
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || 'Failed to start feature combination');
      }
    } catch (err) {
      console.error('Error starting feature combination:', err);
      setError(err instanceof Error ? err.message : 'Failed to start feature combination.');
      setIsCombining(false);
    }
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Feature Combination</h1>
          <p className="text-gray-600">
            Merge paired MRI and PET feature vectors into a single multimodal representation for downstream training.
          </p>
        </div>

        <div className="bg-white border-2 border-indigo-200 rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">What This Step Does</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-4">
              <p className="text-sm font-semibold text-blue-900 mb-1">Input</p>
              <p className="text-sm text-blue-800">Loads `mri_features.npy` and `pet_features.npy` from the extracted-features stage.</p>
            </div>
            <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-4">
              <p className="text-sm font-semibold text-indigo-900 mb-1">Fusion</p>
              <p className="text-sm text-indigo-800">Concatenates each MRI feature vector with its paired PET feature vector.</p>
            </div>
            <div className="rounded-lg bg-green-50 border border-green-200 p-4">
              <p className="text-sm font-semibold text-green-900 mb-1">Output</p>
              <p className="text-sm text-green-800">Saves `combined_features.npy` for the next classifier-training stage.</p>
            </div>
          </div>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <button
            onClick={startCombination}
            disabled={isCombining}
            className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
              isCombining
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {isCombining ? 'Combining Features...' : 'Start Feature Combination'}
          </button>
        </div>

        {(isCombining || progress > 0) && (
          <div className="bg-white border-2 border-indigo-200 rounded-lg p-6 mb-6 shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-2xl font-bold text-gray-900">{currentStep || 'Combining Features'}</h2>
              <span className="text-2xl font-bold text-indigo-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div
                className="bg-indigo-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <div className="grid grid-cols-3 gap-3 mt-5">
              <div className={`rounded-lg p-3 text-center ${progress >= 15 ? 'bg-blue-100 border border-blue-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Load Features</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 55 ? 'bg-indigo-100 border border-indigo-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Pair and Concatenate</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 80 ? 'bg-green-100 border border-green-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Save and Summarize</p>
              </div>
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="bg-gray-900 text-green-400 rounded-lg p-5 mb-6 shadow-xl border-2 border-gray-700">
            <div className="flex items-center mb-3 pb-2 border-b border-gray-700">
              <span className="text-white font-mono text-sm font-semibold">Feature Combination Logs</span>
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
            <div className="bg-indigo-600 rounded-lg shadow-lg p-6">
              <h2 className="text-3xl font-bold text-white mb-2">Feature Combination Complete</h2>
              <p className="text-indigo-100">Your paired MRI and PET embeddings are now fused into one multimodal feature matrix.</p>
            </div>

            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Paired Samples</p>
                <p className="text-4xl font-bold text-blue-600">{results.paired_samples}</p>
              </div>
              <div className="bg-white border-2 border-indigo-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">MRI Dim</p>
                <p className="text-4xl font-bold text-indigo-600">{results.mri_feature_dimension}</p>
              </div>
              <div className="bg-white border-2 border-purple-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">PET Dim</p>
                <p className="text-4xl font-bold text-purple-600">{results.pet_feature_dimension}</p>
              </div>
              <div className="bg-white border-2 border-green-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Combined Dim</p>
                <p className="text-4xl font-bold text-green-600">{results.combined_feature_dimension}</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Combined Feature Statistics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Mean</p>
                    <p className="text-2xl font-bold text-indigo-700">{results.combined_stats.mean.toFixed(4)}</p>
                  </div>
                  <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Std</p>
                    <p className="text-2xl font-bold text-blue-700">{results.combined_stats.std.toFixed(4)}</p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Min</p>
                    <p className="text-2xl font-bold text-green-700">{results.combined_stats.min.toFixed(4)}</p>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Max</p>
                    <p className="text-2xl font-bold text-orange-700">{results.combined_stats.max.toFixed(4)}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Output Summary</h3>
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Saved File</p>
                    <p className="text-sm font-mono text-gray-800 break-all">{results.output_file}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Combination Time</p>
                    <p className="text-2xl font-bold text-gray-900">{results.combination_time.toFixed(4)}s</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
