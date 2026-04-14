import { useEffect, useState } from 'react';

interface TrainingStatus {
  status: string;
  progress: number;
  current_step: string;
  logs: string[];
  error?: string | null;
  current_epoch: number;
  total_epochs: number;
  latest_metrics?: EpochMetrics | null;
}

interface EpochMetrics {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

interface TrainingResults {
  num_epochs: number;
  num_clusters: number;
  silhouette_score: number;
  dataset_split: {
    train_samples: number;
    val_samples: number;
    test_samples: number;
  };
  cluster_distribution: Array<{
    cluster: number;
    count: number;
    percentage: number;
  }>;
  best_metrics: {
    best_val_loss: number;
    best_val_accuracy: number;
  };
  latest_metrics: EpochMetrics | null;
  history: EpochMetrics[];
  artifacts: {
    model_path: string;
    labels_path: string;
  };
}

export default function ClassificationTraining() {
  const [numEpochs, setNumEpochs] = useState(50);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [latestMetrics, setLatestMetrics] = useState<EpochMetrics | null>(null);
  const [results, setResults] = useState<TrainingResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isTraining) {
      interval = setInterval(fetchStatus, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTraining]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/classification-training/status');
      const data: TrainingStatus = await response.json();

      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setLogs(data.logs || []);
      setCurrentEpoch(data.current_epoch || 0);
      setTotalEpochs(data.total_epochs || 0);
      setLatestMetrics(data.latest_metrics || null);

      if (data.status === 'completed') {
        setIsTraining(false);
        await fetchResults();
      } else if (data.status === 'error') {
        setIsTraining(false);
        setError(data.error || 'Classifier training failed. Please check the logs.');
      }
    } catch (err) {
      console.error('Error fetching training status:', err);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/classification-training/results');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error fetching training results:', err);
      setError('Failed to fetch classifier training results.');
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setProgress(0);
    setCurrentStep('Initializing...');
    setLogs([]);
    setResults(null);
    setError(null);
    setCurrentEpoch(0);
    setTotalEpochs(numEpochs);
    setLatestMetrics(null);

    try {
      const response = await fetch('http://localhost:5000/api/classification-training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ num_epochs: numEpochs }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || 'Failed to start classifier training');
      }
    } catch (err) {
      console.error('Error starting classifier training:', err);
      setError(err instanceof Error ? err.message : 'Failed to start classifier training.');
      setIsTraining(false);
    }
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Classification Training</h1>
          <p className="text-gray-600">
            Generate pseudo-labels from combined features and train the classification head epoch by epoch.
          </p>
        </div>

        <div className="bg-white border-2 border-emerald-200 rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Training Pipeline</h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-4">
              <p className="text-sm font-semibold text-blue-900 mb-1">1. Load</p>
              <p className="text-sm text-blue-800">Reads `combined_features.npy` from the previous stage.</p>
            </div>
            <div className="rounded-lg bg-purple-50 border border-purple-200 p-4">
              <p className="text-sm font-semibold text-purple-900 mb-1">2. Cluster</p>
              <p className="text-sm text-purple-800">Uses KMeans and silhouette score to generate pseudo-labels.</p>
            </div>
            <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-4">
              <p className="text-sm font-semibold text-indigo-900 mb-1">3. Train</p>
              <p className="text-sm text-indigo-800">Trains the classifier head with train and validation splits.</p>
            </div>
            <div className="rounded-lg bg-green-50 border border-green-200 p-4">
              <p className="text-sm font-semibold text-green-900 mb-1">4. Save</p>
              <p className="text-sm text-green-800">Stores the best model weights and pseudo-label artifacts.</p>
            </div>
          </div>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <div className="grid md:grid-cols-[1fr_auto] gap-4 items-end">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Number of Epochs</label>
              <input
                type="number"
                min={1}
                max={500}
                value={numEpochs}
                onChange={(event) => setNumEpochs(Number(event.target.value))}
                disabled={isTraining}
                className="w-full md:w-64 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
            <button
              onClick={startTraining}
              disabled={isTraining}
              className={`px-8 py-4 rounded-lg font-semibold text-lg transition-colors ${
                isTraining
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-emerald-600 text-white hover:bg-emerald-700'
              }`}
            >
              {isTraining ? 'Training Classifier...' : 'Start Classification Training'}
            </button>
          </div>
        </div>

        {(isTraining || progress > 0) && (
          <div className="bg-white border-2 border-emerald-200 rounded-lg p-6 mb-6 shadow-lg">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <h2 className="text-2xl font-bold text-gray-900">{currentStep || 'Classifier Training'}</h2>
              <div className="text-right">
                <p className="text-2xl font-bold text-emerald-600">{progress}%</p>
                <p className="text-sm text-gray-500">
                  Epoch {currentEpoch}/{totalEpochs || numEpochs}
                </p>
              </div>
            </div>

            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden mb-5">
              <div
                className="bg-emerald-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>

            <div className="grid grid-cols-4 gap-3">
              <div className={`rounded-lg p-3 text-center ${progress >= 10 ? 'bg-blue-100 border border-blue-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Load Features</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 20 ? 'bg-purple-100 border border-purple-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Pseudo Labels</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 30 ? 'bg-indigo-100 border border-indigo-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Data Split</p>
              </div>
              <div className={`rounded-lg p-3 text-center ${progress >= 35 ? 'bg-green-100 border border-green-300' : 'bg-gray-100'}`}>
                <p className="text-sm font-medium text-gray-700">Epoch Training</p>
              </div>
            </div>
          </div>
        )}

        {latestMetrics && (
          <div className="grid md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white border border-blue-200 rounded-lg p-4 shadow-sm">
              <p className="text-xs text-gray-600 uppercase mb-1">Train Loss</p>
              <p className="text-2xl font-bold text-blue-700">{latestMetrics.train_loss.toFixed(4)}</p>
            </div>
            <div className="bg-white border border-green-200 rounded-lg p-4 shadow-sm">
              <p className="text-xs text-gray-600 uppercase mb-1">Train Accuracy</p>
              <p className="text-2xl font-bold text-green-700">{(latestMetrics.train_accuracy * 100).toFixed(2)}%</p>
            </div>
            <div className="bg-white border border-purple-200 rounded-lg p-4 shadow-sm">
              <p className="text-xs text-gray-600 uppercase mb-1">Val Loss</p>
              <p className="text-2xl font-bold text-purple-700">{latestMetrics.val_loss.toFixed(4)}</p>
            </div>
            <div className="bg-white border border-orange-200 rounded-lg p-4 shadow-sm">
              <p className="text-xs text-gray-600 uppercase mb-1">Val Accuracy</p>
              <p className="text-2xl font-bold text-orange-700">{(latestMetrics.val_accuracy * 100).toFixed(2)}%</p>
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="bg-gray-900 text-green-400 rounded-lg p-5 mb-6 shadow-xl border-2 border-gray-700">
            <div className="flex items-center mb-3 pb-2 border-b border-gray-700">
              <span className="text-white font-mono text-sm font-semibold">Classification Training Logs</span>
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
            <div className="bg-emerald-600 rounded-lg shadow-lg p-6">
              <h2 className="text-3xl font-bold text-white mb-2">Training Complete</h2>
              <p className="text-emerald-100">The classification head has finished training and the best model has been saved.</p>
            </div>

            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Epochs</p>
                <p className="text-4xl font-bold text-blue-600">{results.num_epochs}</p>
              </div>
              <div className="bg-white border-2 border-purple-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Clusters</p>
                <p className="text-4xl font-bold text-purple-600">{results.num_clusters}</p>
              </div>
              <div className="bg-white border-2 border-indigo-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Silhouette</p>
                <p className="text-4xl font-bold text-indigo-600">{results.silhouette_score.toFixed(4)}</p>
              </div>
              <div className="bg-white border-2 border-green-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Best Val Acc</p>
                <p className="text-4xl font-bold text-green-600">{(results.best_metrics.best_val_accuracy * 100).toFixed(2)}%</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Dataset Split</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Train</p>
                    <p className="text-2xl font-bold text-blue-700">{results.dataset_split.train_samples}</p>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Validation</p>
                    <p className="text-2xl font-bold text-purple-700">{results.dataset_split.val_samples}</p>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Test</p>
                    <p className="text-2xl font-bold text-orange-700">{results.dataset_split.test_samples}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Best Validation Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Best Val Loss</p>
                    <p className="text-2xl font-bold text-red-700">{results.best_metrics.best_val_loss.toFixed(4)}</p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Best Val Accuracy</p>
                    <p className="text-2xl font-bold text-green-700">{(results.best_metrics.best_val_accuracy * 100).toFixed(2)}%</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Cluster Distribution</h3>
                <div className="space-y-4">
                  {results.cluster_distribution.map((cluster) => (
                    <div key={cluster.cluster}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-gray-700">Cluster {cluster.cluster}</span>
                        <span className="font-bold text-gray-800">
                          {cluster.count} samples ({cluster.percentage.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className="bg-emerald-500 h-3 rounded-full"
                          style={{ width: `${cluster.percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Saved Artifacts</h3>
                <div className="space-y-4">
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

            {results.history.length > 0 && (
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Recent Epoch History</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200 text-left text-gray-600">
                        <th className="py-3 pr-4">Epoch</th>
                        <th className="py-3 pr-4">Train Loss</th>
                        <th className="py-3 pr-4">Train Acc</th>
                        <th className="py-3 pr-4">Val Loss</th>
                        <th className="py-3">Val Acc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.history.slice(-10).map((epoch) => (
                        <tr key={epoch.epoch} className="border-b border-gray-100">
                          <td className="py-3 pr-4 font-semibold text-gray-800">{epoch.epoch}</td>
                          <td className="py-3 pr-4 text-blue-700">{epoch.train_loss.toFixed(4)}</td>
                          <td className="py-3 pr-4 text-green-700">{(epoch.train_accuracy * 100).toFixed(2)}%</td>
                          <td className="py-3 pr-4 text-purple-700">{epoch.val_loss.toFixed(4)}</td>
                          <td className="py-3 text-orange-700">{(epoch.val_accuracy * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
