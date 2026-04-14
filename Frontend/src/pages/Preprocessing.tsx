import { useState, useEffect, useRef } from 'react';

interface PreprocessingLog {
  message: string;
  timestamp: number;
}

interface PreprocessingStatus {
  is_running: boolean;
  current_step: string;
  progress: number;
  logs: PreprocessingLog[];
  results: any;
  error: string | null;
}

const Preprocessing = () => {
  const [status, setStatus] = useState<PreprocessingStatus>({
    is_running: false,
    current_step: '',
    progress: 0,
    logs: [],
    results: {},
    error: null
  });
  const [originalAnalysis, setOriginalAnalysis] = useState<any>(null);
  const [finalResults, setFinalResults] = useState<any>(null);
  const [showComparison, setShowComparison] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Fetch original dataset analysis
    fetchOriginalAnalysis();
  }, []);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (status.is_running) {
      interval = setInterval(fetchStatus, 1000);
    }
    return () => clearInterval(interval);
  }, [status.is_running]);

  useEffect(() => {
    // Auto-scroll logs
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [status.logs]);

  const fetchOriginalAnalysis = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/dataset-analysis');
      if (response.ok) {
        const data = await response.json();
        setOriginalAnalysis(data);
      }
    } catch (error) {
      console.error('Error fetching original analysis:', error);
    }
  };

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/preprocessing/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        
        if (!data.is_running && data.progress === 100 && !finalResults) {
          fetchFinalResults();
        }
      }
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const fetchFinalResults = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/preprocessing/results');
      if (response.ok) {
        const data = await response.json();
        setFinalResults(data);
        setShowComparison(true);
      }
    } catch (error) {
      console.error('Error fetching final results:', error);
    }
  };

  const startPreprocessing = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/preprocessing/start', {
        method: 'POST'
      });
      if (response.ok) {
        setStatus(prev => ({ ...prev, is_running: true }));
        setFinalResults(null);
        setShowComparison(false);
      } else {
        const error = await response.json();
        alert(error.error || 'Failed to start preprocessing');
      }
    } catch (error) {
      console.error('Error starting preprocessing:', error);
      alert('Failed to start preprocessing');
    }
  };

  const renderStepCard = (title: string, stepKey: string, icon: string) => {
    const stepData = status.results[stepKey];
    const isComplete = stepData !== undefined;
    const isCurrent = status.current_step.toLowerCase().includes(stepKey);

    return (
      <div className={`rounded-lg border-2 p-5 transition-all ${
        isComplete ? 'bg-green-50 border-green-400 shadow-md' : 
        isCurrent ? 'bg-blue-50 border-blue-400 animate-pulse shadow-lg' : 
        'bg-gray-50 border-gray-200'
      }`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{icon}</span>
            <h3 className="font-bold text-gray-900 text-lg">{title}</h3>
          </div>
          {isComplete && <span className="text-green-600 text-2xl font-bold">✓</span>}
          {isCurrent && <span className="text-blue-600 text-xl font-bold animate-spin">⟳</span>}
        </div>
        
        {stepData && (
          <div className="mt-4 space-y-3">
            {/* MRI Progress */}
            {stepData.mri && (
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 font-medium">MRI</span>
                  <span className="text-blue-600 font-semibold">{stepData.mri.processed || stepData.mri_success || 0}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{width: `${stepData.mri.processed ? 100 : 0}%`}}
                  ></div>
                </div>
                {stepData.mri.errors > 0 && (
                  <p className="text-xs text-red-600 mt-1">⚠ {stepData.mri.errors} errors</p>
                )}
              </div>
            )}
            
            {/* PET Progress */}
            {stepData.pet && (
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 font-medium">PET</span>
                  <span className="text-purple-600 font-semibold">{stepData.pet.processed || stepData.pet_success || 0}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-500 h-2 rounded-full transition-all"
                    style={{width: `${stepData.pet.processed ? 100 : 0}%`}}
                  ></div>
                </div>
                {stepData.pet.errors > 0 && (
                  <p className="text-xs text-red-600 mt-1">⚠ {stepData.pet.errors} errors</p>
                )}
              </div>
            )}
            
            {/* Batch Stats */}
            {stepData.mri_batches !== undefined && (
              <div className="bg-white rounded p-2 border border-gray-200">
                <p className="text-xs text-gray-600">Batches Created</p>
                <p className="text-lg font-bold text-gray-900">
                  {stepData.mri_batches} <span className="text-blue-600">MRI</span> + {stepData.pet_batches} <span className="text-purple-600">PET</span>
                </p>
              </div>
            )}
            
            {/* Augment Stats */}
            {stepData.mri_success !== undefined && (
              <div className="bg-white rounded p-2 border border-gray-200">
                <p className="text-xs text-gray-600">Image Pairs</p>
                <p className="text-lg font-bold text-green-700">{stepData.mri_success} augmented</p>
                {stepData.total_images && (
                  <p className="text-xs text-gray-500">Total: {stepData.total_images} images</p>
                )}
              </div>
            )}
          </div>
        )}
        
        {/* Show intermediate status for current step */}
        {isCurrent && !isComplete && (
          <div className="mt-3 flex items-center gap-2 text-sm text-blue-600">
            <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
            <span className="font-medium">Processing...</span>
          </div>
        )}
      </div>
    );
  };

  const renderComparison = () => {
    if (!originalAnalysis || !finalResults?.final_analysis) return null;

    const original = originalAnalysis;
    const augmented = finalResults.final_analysis.augmented_mri;
    const augmented_pet = finalResults.final_analysis.augmented_pet;
    
    const mriIncrease = ((augmented.total_files / original.mri.total_files - 1) * 100);
    const petIncrease = ((augmented_pet.total_files / original.pet.total_files - 1) * 100);

    return (
      <div className="mt-8 space-y-6">
        {/* Main Comparison Header */}
        <div className="bg-linear-to-r from-indigo-600 to-purple-600 rounded-lg shadow-lg p-6">
          <h2 className="text-3xl font-bold text-white mb-2">📊 Before vs After Comparison</h2>
          <p className="text-indigo-100">Visual analysis of preprocessing transformations</p>
        </div>

        {/* MRI vs PET Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* MRI Comparison */}
          <div className="bg-white rounded-lg shadow-lg border-2 border-blue-200 overflow-hidden">
            <div className="bg-blue-600 px-5 py-3">
              <h3 className="font-bold text-white text-lg">🧠 MRI Dataset</h3>
            </div>
            <div className="p-5 space-y-4">
              <div className="flex justify-between items-center">
                <div className="text-center flex-1">
                  <p className="text-xs text-gray-500 uppercase mb-1">Original</p>
                  <p className="text-3xl font-bold text-gray-800">{original.mri.total_files}</p>
                  <p className="text-xs text-gray-500">files</p>
                </div>
                <div className="text-green-500 text-3xl font-bold px-4">→</div>
                <div className="text-center flex-1">
                  <p className="text-xs text-gray-500 uppercase mb-1">Processed</p>
                  <p className="text-3xl font-bold text-green-600">{augmented.total_files}</p>
                  <p className="text-xs text-gray-500">files</p>
                </div>
              </div>
              
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Data Augmentation</span>
                  <span className="text-2xl font-bold text-green-600">+{mriIncrease.toFixed(0)}%</span>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-green-500 h-3 rounded-full transition-all"
                    style={{width: `${Math.min(mriIncrease, 100)}%`}}
                  ></div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-blue-50 rounded p-2 border border-blue-100">
                  <p className="text-xs text-gray-600">Dimensions</p>
                  <p className="font-bold text-gray-900">{Object.keys(augmented.dimensions)[0] || 'N/A'}</p>
                </div>
                <div className="bg-blue-50 rounded p-2 border border-blue-100">
                  <p className="text-xs text-gray-600">File Types</p>
                  <p className="font-bold text-gray-900">{Object.keys(augmented.file_types).join(', ')}</p>
                </div>
              </div>
            </div>
          </div>

          {/* PET Comparison */}
          <div className="bg-white rounded-lg shadow-lg border-2 border-purple-200 overflow-hidden">
            <div className="bg-purple-600 px-5 py-3">
              <h3 className="font-bold text-white text-lg">💠 PET Dataset</h3>
            </div>
            <div className="p-5 space-y-4">
              <div className="flex justify-between items-center">
                <div className="text-center flex-1">
                  <p className="text-xs text-gray-500 uppercase mb-1">Original</p>
                  <p className="text-3xl font-bold text-gray-800">{original.pet.total_files}</p>
                  <p className="text-xs text-gray-500">files</p>
                </div>
                <div className="text-green-500 text-3xl font-bold px-4">→</div>
                <div className="text-center flex-1">
                  <p className="text-xs text-gray-500 uppercase mb-1">Processed</p>
                  <p className="text-3xl font-bold text-green-600">{augmented_pet.total_files}</p>
                  <p className="text-xs text-gray-500">files</p>
                </div>
              </div>
              
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Data Augmentation</span>
                  <span className="text-2xl font-bold text-green-600">+{petIncrease.toFixed(0)}%</span>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-green-500 h-3 rounded-full transition-all"
                    style={{width: `${Math.min(petIncrease, 100)}%`}}
                  ></div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-purple-50 rounded p-2 border border-purple-100">
                  <p className="text-xs text-gray-600">Dimensions</p>
                  <p className="font-bold text-gray-900">{Object.keys(augmented_pet.dimensions)[0] || 'N/A'}</p>
                </div>
                <div className="bg-purple-50 rounded p-2 border border-purple-100">
                  <p className="text-xs text-gray-600">File Types</p>
                  <p className="font-bold text-gray-900">{Object.keys(augmented_pet.file_types).join(', ')}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline Statistics */}
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">📈 Pipeline Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {finalResults.summary.resize && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-xs text-gray-600 mb-1">Resized Images</p>
                <p className="text-2xl font-bold text-blue-700">
                  {(finalResults.summary.resize.mri?.processed || 0) + (finalResults.summary.resize.pet?.processed || 0)}
                </p>
              </div>
            )}
            {finalResults.summary.normalize && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <p className="text-xs text-gray-600 mb-1">Normalized Images</p>
                <p className="text-2xl font-bold text-green-700">
                  {(finalResults.summary.normalize.mri?.processed || 0) + (finalResults.summary.normalize.pet?.processed || 0)}
                </p>
              </div>
            )}
            {finalResults.summary.augment && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <p className="text-xs text-gray-600 mb-1">Image Pairs</p>
                <p className="text-2xl font-bold text-purple-700">
                  {finalResults.summary.augment.mri_success || 0}
                </p>
              </div>
            )}
            {finalResults.summary.batch && (
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                <p className="text-xs text-gray-600 mb-1">Total Batches</p>
                <p className="text-2xl font-bold text-orange-700">
                  {(finalResults.summary.batch.mri_batches || 0) + (finalResults.summary.batch.pet_batches || 0)}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex-1 p-8 overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Data Preprocessing Pipeline</h1>
          <p className="text-gray-600">
            Automated preprocessing: Resize → Normalize → Augment → Batch
          </p>
        </div>

        {/* Start Button */}
        {!status.is_running && status.progress !== 100 && (
          <div className="bg-white rounded-lg shadow-lg p-8 mb-8 text-center border border-gray-200">
            <div className="mb-4">
              <div className="text-6xl mb-4">🚀</div>
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Ready to Preprocess Data</h2>
              <p className="text-gray-600">
                This will resize, normalize, augment, and batch your MRI and PET images
              </p>
            </div>
            <button
              onClick={startPreprocessing}
              className="px-8 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md font-bold text-lg"
            >
              Start Preprocessing
            </button>
          </div>
        )}

        {/* Progress Bar */}
        {(status.is_running || status.progress > 0) && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8 border border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xl font-bold text-gray-800">{status.current_step || 'Processing...'}</h2>
              <span className="text-2xl font-bold text-blue-600">{status.progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div
                className="bg-blue-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${status.progress}%` }}
              ></div>
            </div>
            {status.error && (
              <div className="mt-4 bg-red-50 border border-red-300 rounded-lg p-3">
                <p className="text-red-700 font-medium">Error: {status.error}</p>
              </div>
            )}
          </div>
        )}

        {/* Processing Steps */}
        {(status.is_running || Object.keys(status.results).length > 0) && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {renderStepCard('Resize', 'resize', '📐')}
            {renderStepCard('Normalize', 'normalize', '⚖️')}
            {renderStepCard('Augment', 'augment', '🔄')}
            {renderStepCard('Batch', 'batch', '📦')}
          </div>
        )}

        {/* Logs */}
        {status.logs.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8 border border-gray-200">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Processing Logs</h2>
            <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
              {status.logs.map((log, index) => (
                <div key={index} className="text-green-400 mb-1">
                  <span className="text-gray-500">
                    [{new Date(log.timestamp * 1000).toLocaleTimeString()}]
                  </span>{' '}
                  {log.message}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {/* Comparison */}
        {showComparison && renderComparison()}

        {/* Final Results Summary */}
        {finalResults && (
          <div className="mt-8 bg-linear-to-r from-green-600 to-blue-600 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-white mb-4">✓ Preprocessing Complete!</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
                <p className="text-sm text-black mb-1">Total Resized</p>
                <p className="text-3xl font-bold text-black">
                  {(finalResults.summary.resize?.mri?.processed || 0) + 
                   (finalResults.summary.resize?.pet?.processed || 0)}
                </p>
              </div>
              <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
                <p className="text-sm text-black mb-1">Total Normalized</p>
                <p className="text-3xl font-bold text-black">
                  {(finalResults.summary.normalize?.mri?.processed || 0) + 
                   (finalResults.summary.normalize?.pet?.processed || 0)}
                </p>
              </div>
              <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
                <p className="text-sm text-black mb-1">Pairs Augmented</p>
                <p className="text-3xl font-bold text-black">
                  {finalResults.summary.augment?.mri_success || 0}
                </p>
              </div>
              <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
                <p className="text-sm text-black mb-1">Batches Created</p>
                <p className="text-3xl font-bold text-black">
                  {(finalResults.summary.batch?.mri_batches || 0) + 
                   (finalResults.summary.batch?.pet_batches || 0)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Preprocessing;
