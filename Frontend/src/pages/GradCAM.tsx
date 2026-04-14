import { useEffect, useState } from 'react';

interface GradCAMStatus {
  status: string;
  progress: number;
  current_step: string;
  logs: string[];
  error?: string | null;
}

interface GradCAMResults {
  batch_index: number;
  image_index: number;
  predicted_class: number;
  predicted_label: string;
  predicted_confidence: number;
  mri_original_image: string;
  mri_gradcam_image: string;
  pet_original_image: string;
  pet_gradcam_image: string;
}

export default function GradCAM() {
  const [batchIndex, setBatchIndex] = useState(0);
  const [imageIndex, setImageIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<GradCAMResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isRunning) {
      interval = setInterval(fetchStatus, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRunning]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/gradcam/status');
      const data: GradCAMStatus = await response.json();

      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setLogs(data.logs || []);

      if (data.status === 'completed') {
        setIsRunning(false);
        await fetchResults();
      } else if (data.status === 'error') {
        setIsRunning(false);
        setError(data.error || 'Grad-CAM generation failed. Please check the logs.');
      }
    } catch (err) {
      console.error('Error fetching Grad-CAM status:', err);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/gradcam/results');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error fetching Grad-CAM results:', err);
      setError('Failed to fetch Grad-CAM results.');
    }
  };

  const startGradCAM = async () => {
    setIsRunning(true);
    setProgress(0);
    setCurrentStep('Initializing...');
    setLogs([]);
    setResults(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/gradcam/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          batch_index: batchIndex,
          image_index: imageIndex,
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || 'Failed to start Grad-CAM');
      }
    } catch (err) {
      console.error('Error starting Grad-CAM:', err);
      setError(err instanceof Error ? err.message : 'Failed to start Grad-CAM.');
      setIsRunning(false);
    }
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Grad-CAM Visualization</h1>
          <p className="text-gray-600">
            Visualize the important regions in MRI and PET images that most influenced the predicted cluster.
          </p>
        </div>

        <div className="bg-white border-2 border-amber-200 rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Inputs</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Batch Index</label>
              <input
                type="number"
                min={0}
                value={batchIndex}
                onChange={(event) => setBatchIndex(Number(event.target.value))}
                disabled={isRunning}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Image Index</label>
              <input
                type="number"
                min={0}
                value={imageIndex}
                onChange={(event) => setImageIndex(Number(event.target.value))}
                disabled={isRunning}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
            </div>
          </div>
          <p className="text-sm text-gray-500 mt-4">
            Grad-CAM needs the saved feature extractor weights from the latest feature-extraction run.
          </p>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <button
            onClick={startGradCAM}
            disabled={isRunning}
            className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
              isRunning
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-amber-600 text-white hover:bg-amber-700'
            }`}
          >
            {isRunning ? 'Generating Grad-CAM...' : 'Start Grad-CAM'}
          </button>
        </div>

        {(isRunning || progress > 0) && (
          <div className="bg-white border-2 border-amber-200 rounded-lg p-6 mb-6 shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-2xl font-bold text-gray-900">{currentStep || 'Grad-CAM'}</h2>
              <span className="text-2xl font-bold text-amber-600">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div
                className="bg-amber-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}

        {logs.length > 0 && (
          <div className="bg-gray-900 text-green-400 rounded-lg p-5 mb-6 shadow-xl border-2 border-gray-700">
            <div className="flex items-center mb-3 pb-2 border-b border-gray-700">
              <span className="text-white font-mono text-sm font-semibold">Grad-CAM Logs</span>
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
            <div className="bg-amber-600 rounded-lg shadow-lg p-6">
              <h2 className="text-3xl font-bold text-white mb-2">Grad-CAM Complete</h2>
              <p className="text-amber-100">
                Predicted {results.predicted_label} with {(results.predicted_confidence * 100).toFixed(2)}% confidence.
              </p>
            </div>

            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Batch</p>
                <p className="text-4xl font-bold text-blue-600">{results.batch_index}</p>
              </div>
              <div className="bg-white border-2 border-purple-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Image</p>
                <p className="text-4xl font-bold text-purple-600">{results.image_index}</p>
              </div>
              <div className="bg-white border-2 border-amber-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Predicted Class</p>
                <p className="text-4xl font-bold text-amber-600">{results.predicted_class}</p>
              </div>
              <div className="bg-white border-2 border-green-300 rounded-lg p-5 shadow-lg">
                <p className="text-xs text-gray-600 font-semibold uppercase mb-2">Confidence</p>
                <p className="text-4xl font-bold text-green-600">{(results.predicted_confidence * 100).toFixed(2)}%</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">MRI Grad-CAM</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-2">Original MRI</p>
                    <img
                      src={`data:image/png;base64,${results.mri_original_image}`}
                      alt="Original MRI"
                      className="w-full rounded-lg border border-gray-200"
                    />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-2">Overlay</p>
                    <img
                      src={`data:image/png;base64,${results.mri_gradcam_image}`}
                      alt="MRI Grad-CAM"
                      className="w-full rounded-lg border border-gray-200"
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-bold text-gray-900 mb-4">PET Grad-CAM</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-2">Original PET</p>
                    <img
                      src={`data:image/png;base64,${results.pet_original_image}`}
                      alt="Original PET"
                      className="w-full rounded-lg border border-gray-200"
                    />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-2">Overlay</p>
                    <img
                      src={`data:image/png;base64,${results.pet_gradcam_image}`}
                      alt="PET Grad-CAM"
                      className="w-full rounded-lg border border-gray-200"
                    />
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
