import { useState, useEffect } from 'react';

interface FeatureStats {
  mri_features_count: number;
  pet_features_count: number;
  feature_dimension: number;
  total_samples: number;
  extraction_time: number;
  feature_stats: {
    mri_mean: number;
    mri_std: number;
    pet_mean: number;
    pet_std: number;
  };
}

interface ExtractionStatus {
  status: string;
  progress: number;
  current_step: string;
  logs: string[];
}

export default function FeatureExtraction() {
  const [isExtracting, setIsExtracting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState('');
  const [results, setResults] = useState<FeatureStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isExtracting) {
      interval = setInterval(fetchStatus, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isExtracting]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/feature-extraction/status');
      const data: ExtractionStatus = await response.json();

      setProgress(data.progress);
      setCurrentStep(data.current_step);
      setLogs(data.logs || []);

      if (data.status === 'completed') {
        setIsExtracting(false);
        await fetchResults();
      } else if (data.status === 'error') {
        setIsExtracting(false);
        setError('Feature extraction failed. Please check the logs.');
      }
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/feature-extraction/results');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error fetching results:', err);
      setError('Failed to fetch extraction results.');
    }
  };

  const startExtraction = async () => {
    setIsExtracting(true);
    setProgress(0);
    setLogs([]);
    setCurrentStep('Initializing...');
    setResults(null);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/feature-extraction/start', {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to start feature extraction');
      }
    } catch (err) {
      console.error('Error starting extraction:', err);
      setError('Failed to start feature extraction. Please ensure backend is running.');
      setIsExtracting(false);
    }
  };

  return (
    <div className="p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">🧬 Feature Extraction</h1>
        <p className="text-gray-600 mb-8">
          Extract deep learning features from batched MRI and PET images using CNN + Transformer architecture
        </p>

        {/* Model Architecture Card */}
        <div className="bg-white border-2 border-purple-200 rounded-lg p-6 mb-6 shadow-sm">
          <h2 className="text-2xl font-semibold text-black mb-4">Model Architecture</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-linear-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-blue-900 mb-2">CNN Backbone</h3>
              <ul className="space-y-1 text-sm text-blue-800">
                <li>• Custom CNN: 3-layer convolution (32 → 64 → 128 channels)</li>
                <li>• Pretrained EfficientNet-B0 (ImageNet weights)</li>
                <li>• Adaptive pooling to 7×7 feature maps</li>
                <li>• Combined feature dimension: 1408 channels</li>
              </ul>
            </div>
            <div className="bg-linear-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-purple-900 mb-2">Transformer Encoder</h3>
              <ul className="space-y-1 text-sm text-purple-800">
                <li>• Multi-head attention: 8 heads</li>
                <li>• Encoder layers: 2 layers</li>
                <li>• Positional embedding for 49 tokens (7×7)</li>
                <li>• Output: 1408-dimensional feature vector</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
          <button
            onClick={startExtraction}
            disabled={isExtracting}
            className={`w-full py-4 rounded-lg font-semibold text-lg transition-colors ${
              isExtracting
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-linear-to-r from-purple-500 to-purple-600 text-white hover:from-purple-600 hover:to-purple-700'
            }`}
          >
            {isExtracting ? 'Extracting Features...' : 'Start Feature Extraction'}
          </button>
        </div>

        {/* Progress Section */}
        {isExtracting && (
          <div className="bg-white border-2 border-blue-200 rounded-lg p-6 mb-6 shadow-lg">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">⚙️ Extraction Progress</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-700 mb-2">
                  <span className="font-semibold">{currentStep}</span>
                  <span className="text-2xl font-bold text-blue-600">{progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-5 overflow-hidden">
                  <div
                    className="bg-linear-to-r from-blue-500 to-purple-500 h-5 rounded-full transition-all duration-300 relative"
                    style={{ width: `${progress}%` }}
                  >
                    <div className="absolute inset-0 bg-white opacity-20 animate-pulse"></div>
                  </div>
                </div>
              </div>
              
              {/* Step Indicator */}
              <div className="grid grid-cols-4 gap-2 mt-4">
                <div className={`text-center p-2 rounded ${progress >= 5 ? 'bg-blue-100 border-2 border-blue-400' : 'bg-gray-100'}`}>
                  <div className="text-lg mb-1">🔧</div>
                  <p className="text-xs font-medium text-gray-700">Loading Model</p>
                </div>
                <div className={`text-center p-2 rounded ${progress >= 25 ? 'bg-blue-100 border-2 border-blue-400' : 'bg-gray-100'}`}>
                  <div className="text-lg mb-1">🧠</div>
                  <p className="text-xs font-medium text-gray-700">MRI Features</p>
                </div>
                <div className={`text-center p-2 rounded ${progress >= 55 ? 'bg-purple-100 border-2 border-purple-400' : 'bg-gray-100'}`}>
                  <div className="text-lg mb-1">💠</div>
                  <p className="text-xs font-medium text-gray-700">PET Features</p>
                </div>
                <div className={`text-center p-2 rounded ${progress >= 90 ? 'bg-green-100 border-2 border-green-400' : 'bg-gray-100'}`}>
                  <div className="text-lg mb-1">📊</div>
                  <p className="text-xs font-medium text-gray-700">Statistics</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Logs Terminal */}
        {logs.length > 0 && (
          <div className="bg-gray-900 text-green-400 rounded-lg p-5 mb-6 shadow-xl border-2 border-gray-700">
            <div className="flex items-center mb-3 pb-2 border-b border-gray-700">
              <div className="flex gap-2 mr-3">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <span className="text-white font-mono text-sm font-semibold">📝 Extraction Logs</span>
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

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 mb-6">
            <p className="text-red-800 font-semibold">{error}</p>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="space-y-6">
            <div className="bg-linear-to-r from-green-600 to-blue-600 rounded-lg shadow-lg p-6">
              <h2 className="text-3xl font-bold text-white mb-2">✅ Extraction Complete!</h2>
              <p className="text-green-100">Deep learning features successfully extracted from medical images</p>
            </div>

            {/* Summary Cards */}
            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-5 shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs text-gray-600 font-semibold uppercase">Total Samples</h3>
                  <span className="text-2xl">🎯</span>
                </div>
                <p className="text-4xl font-bold text-blue-600">{results.total_samples}</p>
                <p className="text-xs text-gray-500 mt-1">Images processed</p>
              </div>
              <div className="bg-white border-2 border-purple-300 rounded-lg p-5 shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs text-gray-600 font-semibold uppercase">Feature Dimension</h3>
                  <span className="text-2xl">📐</span>
                </div>
                <p className="text-4xl font-bold text-purple-600">{results.feature_dimension}</p>
                <p className="text-xs text-gray-500 mt-1">Vector size per image</p>
              </div>
              <div className="bg-white border-2 border-green-300 rounded-lg p-5 shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs text-gray-600 font-semibold uppercase">MRI Features</h3>
                  <span className="text-2xl">🧠</span>
                </div>
                <p className="text-4xl font-bold text-green-600">{results.mri_features_count}</p>
                <p className="text-xs text-gray-500 mt-1">Feature vectors</p>
              </div>
              <div className="bg-white border-2 border-orange-300 rounded-lg p-5 shadow-lg hover:shadow-xl transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs text-gray-600 font-semibold uppercase">PET Features</h3>
                  <span className="text-2xl">💠</span>
                </div>
                <p className="text-4xl font-bold text-orange-600">{results.pet_features_count}</p>
                <p className="text-xs text-gray-500 mt-1">Feature vectors</p>
              </div>
            </div>

            {/* Feature Statistics */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white border-2 border-blue-300 rounded-lg p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-5">
                  <span className="text-3xl">🧠</span>
                  <h3 className="text-xl font-bold text-gray-900">MRI Feature Statistics</h3>
                </div>
                <div className="space-y-4">
                  {/* <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Mean Value</p>
                    <p className="text-3xl font-bold text-blue-700">
                      {results.feature_stats.mri_mean.toFixed(4)}
                    </p>
                    <div className="mt-2 w-full bg-blue-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full"
                        style={{width: `${Math.min(Math.abs(results.feature_stats.mri_mean) * 100, 100)}%`}}
                      ></div>
                    </div>
                  </div> */}
                  <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Standard Deviation</p>
                    <p className="text-3xl font-bold text-blue-700">
                      {results.feature_stats.mri_std.toFixed(4)}
                    </p>
                    <div className="mt-2 w-full bg-blue-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full"
                        style={{width: `${Math.min(results.feature_stats.mri_std * 100, 100)}%`}}
                      ></div>
                    </div>
                  </div>
                  <div className="bg-linear-to-r from-blue-100 to-cyan-100 rounded-lg p-3">
                    <p className="text-xs text-gray-700 font-medium">Distribution Quality</p>
                    <p className="text-sm text-gray-600 mt-1">
                      {results.feature_stats.mri_std > 0.5 ? '✓ Well-distributed' : '⚠ Low variance'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white border-2 border-purple-300 rounded-lg p-6 shadow-lg">
                <div className="flex items-center gap-3 mb-5">
                  <span className="text-3xl">💠</span>
                  <h3 className="text-xl font-bold text-gray-900">PET Feature Statistics</h3>
                </div>
                <div className="space-y-4">
                  {/* <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Mean Value</p>
                    <p className="text-3xl font-bold text-purple-700">
                      {results.feature_stats.pet_mean.toFixed(4)}
                    </p>
                    <div className="mt-2 w-full bg-purple-200 rounded-full h-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full"
                        style={{width: `${Math.min(Math.abs(results.feature_stats.pet_mean) * 100, 100)}%`}}
                      ></div>
                    </div>
                  </div> */}
                  <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <p className="text-xs text-gray-600 uppercase mb-1">Standard Deviation</p>
                    <p className="text-3xl font-bold text-purple-700">
                      {results.feature_stats.pet_std.toFixed(4)}
                    </p>
                    <div className="mt-2 w-full bg-purple-200 rounded-full h-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full"
                        style={{width: `${Math.min(results.feature_stats.pet_std * 100, 100)}%`}}
                      ></div>
                    </div>
                  </div>
                  <div className="bg-linear-to-r from-purple-100 to-pink-100 rounded-lg p-3">
                    <p className="text-xs text-gray-700 font-medium">Distribution Quality</p>
                    <p className="text-sm text-gray-600 mt-1">
                      {results.feature_stats.pet_std > 0.5 ? '✓ Well-distributed' : '⚠ Low variance'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-linear-to-r from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-indigo-900 mb-4">Extraction Performance</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-700">Total Extraction Time</p>
                  <p className="text-2xl font-bold text-black">{results.extraction_time.toFixed(2)}s</p>
                </div>
                <div>
                  <p className="text-sm text-gray-700">Average Time per Sample</p>
                  <p className="text-2xl font-bold text-black">
                    {(results.extraction_time / results.total_samples).toFixed(4)}s
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-700">Throughput</p>
                  <p className="text-2xl font-bold text-black">
                    {(results.total_samples / results.extraction_time).toFixed(2)} samples/s
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
