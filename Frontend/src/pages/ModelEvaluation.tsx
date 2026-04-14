import { useEffect, useState } from 'react';

interface EvaluationStatus {
  status: string;
  progress: number;
  current_step: string;
  logs: string[];
  error?: string | null;
}

interface EvaluationResults {
  test_samples: number;
  num_classes: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  confusion_matrix: number[][];
  per_class_accuracy: Array<{
    class_index: number;
    samples: number;
    accuracy: number;
  }>;
  artifacts: {
    model_path: string;
    labels_path: string;
  };
}

export default function ModelEvaluation() {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<EvaluationResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isEvaluating) {
      interval = setInterval(fetchStatus, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isEvaluating]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/evaluation/status');
      const data: EvaluationStatus = await response.json();

      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setLogs(data.logs || []);

      if (data.status === 'completed') {
        setIsEvaluating(false);
        await fetchResults();
      } else if (data.status === 'error') {
        setIsEvaluating(false);
        setError(data.error || 'Model evaluation failed. Please check the logs.');
      }
    } catch (err) {
      console.error('Error fetching evaluation status:', err);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/evaluation/results');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error fetching evaluation results:', err);
      setError('Failed to fetch evaluation results.');
    }
  };

  const startEvaluation = async () => {
    setIsEvaluating(true);
    setProgress(0);
    setCurrentStep('Initializing...');
    setLogs([]);
    setResults(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/evaluation/start', {
        method: 'POST',
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || 'Failed to start evaluation');
      }
    } catch (err) {
      console.error('Error starting evaluation:', err);
      setError(err instanceof Error ? err.message : 'Failed to start evaluation.');
      setIsEvaluating(false);
    }
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Model Evaluation</h1>
          <p className="text-gray-600">
            Evaluate the trained classifier on the held-out test split and inspect final performance metrics.
          </p>
        </div>

        <div className="bg-white border-2 border-rose-200 rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Evaluation Outputs</h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-4">
              <p className="text-sm font-semibold text-blue-900 mb-1">Accuracy</p>
              <p className="text-sm text-blue-800">Overall fraction of correct predictions on the test split.</p>
            </div>
            <div className="rounded-lg bg-purple-50 border border-purple-200 p-4">
              <p className="text-sm font-semibold text-purple-900 mb-1">Precision</p>
              <p className="text-sm text-purple-800">Weighted precision across the pseudo-label classes.</p>
            </div>
            <div className="rounded-lg bg-orange-50 border border-orange-200 p-4">
              <p className="text-sm font-semibold text-orange-900 mb-1">Recall & F1</p>
              <p className="text-sm text-orange-800">Balanced test metrics computed from the saved best model.</p>
            </div>
            <div className="rounded-lg bg-rose-50 border border-rose-200 p-4">
              <p className="text-sm font-semibold text-rose-900 mb-1">Confusion Matrix</p>
              <p className="text-sm text-rose-800">Class-by-class prediction breakdown for the held-out test set.</p>
            </div>
          </div>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <button
            onClick={startEvaluation}
            disabled={isEvaluating}
            className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
              isEvaluating
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-rose-600 text-white hover:bg-rose-700'
            }`}
          >
            {isEvaluating ? 'Evaluating Model...' : 'Start Model Evaluation'}
          </button>
        </div>

        {(isEvaluating || progress > 0) && (
          <div className="bg-white border-2 border-rose-200 rounded-lg p-6 mb-6 shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-2xl font-bold text-gray-900">{currentStep || 'Model Evaluation'}</h2>
              <span className="text-2xl font-bold text-rose-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div
                className="bg-rose-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <div className="grid grid-cols-4 gap-3 mt-5">
              <div className={`rounded-lg p-3 text-center ${progress >= 15 ? 'bg-blue-100 border border-blue-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Load Artifacts</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 35 ? 'bg-purple-100 border border-purple-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Build Test Split</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 60 ? 'bg-orange-100 border border-orange-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Run Predictions</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 85 ? 'bg-rose-100 border border-rose-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Compute Metrics</p>
              </div>
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="bg-gray-900 text-green-400 rounded-lg p-5 mb-6 shadow-xl border-2 border-gray-700">
            <div className="flex items-center mb-3 pb-2 border-b border-gray-700">
              <span className="text-white font-mono text-sm font-semibold">Evaluation Logs</span>
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
            <div className="bg-rose-600 rounded-lg shadow-lg p-6">
              <h2 className="text-3xl font-bold text-white mb-2">Evaluation Complete</h2>
              <p className="text-rose-100">The trained model has been scored on the held-out test set.</p>
            </div>

            <div className="grid md:grid-cols-5 gap-4">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Test Samples</p>
                <p className="text-4xl font-bold text-blue-600">{results.test_samples}</p>
              </div>
              <div className="bg-white border-2 border-green-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Accuracy</p>
                <p className="text-4xl font-bold text-green-600">{(results.metrics.accuracy * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-white border-2 border-purple-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Precision</p>
                <p className="text-4xl font-bold text-purple-600">{(results.metrics.precision * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-white border-2 border-orange-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Recall</p>
                <p className="text-4xl font-bold text-orange-600">{(results.metrics.recall * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-white border-2 border-rose-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">F1 Score</p>
                <p className="text-4xl font-bold text-rose-600">{(results.metrics.f1_score * 100).toFixed(2)}%</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Confusion Matrix</h3>
                <div className="space-y-3">
                  {results.confusion_matrix.map((row, rowIndex) => (
                    <div key={rowIndex} className="grid gap-2" style={{ gridTemplateColumns: `repeat(${row.length}, minmax(0, 1fr))` }}>
                      {row.map((value, colIndex) => (
                        <div
                          key={`${rowIndex}-${colIndex}`}
                          className={`rounded-lg p-4 text-center border ${
                            rowIndex === colIndex ? 'bg-emerald-50 border-emerald-200' : 'bg-gray-50 border-gray-200'
                          }`}
                        >
                          <p className="text-xs text-gray-500 mb-1">
                            T{rowIndex} / P{colIndex}
                          </p>
                          <p className="text-2xl font-bold text-gray-800">{value}</p>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Per-Class Accuracy</h3>
                <div className="space-y-4">
                  {results.per_class_accuracy.map((item) => (
                    <div key={item.class_index}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-gray-700">Class {item.class_index}</span>
                        <span className="font-bold text-gray-800">
                          {(item.accuracy * 100).toFixed(2)}% ({item.samples} samples)
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-rose-500 h-3 rounded-full"
                          style={{ width: `${Math.min(item.accuracy * 100, 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Artifacts Used</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <p className="text-xs text-gray-600 uppercase mb-1">Best Model</p>
                  <p className="text-sm font-mono text-gray-800 break-all">{results.artifacts.model_path}</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <p className="text-xs text-gray-600 uppercase mb-1">Pseudo Labels</p>
                  <p className="text-sm font-mono text-gray-800 break-all">{results.artifacts.labels_path}</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
