import { useEffect, useState } from 'react';

interface FileInfo {
  name: string;
  size_mb: number;
  extension: string;
}

interface DatasetStats {
  total_files: number;
  total_size_mb: number;
  file_types: { [key: string]: number };
  dimensions: { [key: string]: number };
  file_list: FileInfo[];
  avg_size_mb: number;
  min_size_mb: number;
  max_size_mb: number;
  sample_images: string[];
  size_distribution: {
    tiny: number;
    small: number;
    medium: number;
    large: number;
    huge: number;
  };
  unique_dimensions: number;
  largest_file: { name: string; size_mb: number };
  smallest_file: { name: string; size_mb: number };
}

interface AnalysisData {
  mri: DatasetStats;
  pet: DatasetStats;
  comparison: {
    total_files: number;
    total_size_mb: number;
    mri_percentage: number;
    pet_percentage: number;
  };
  timestamp: number;
}

const DatasetAnalysis = () => {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalysis();
  }, []);

  const fetchAnalysis = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/api/dataset-analysis');
      if (!response.ok) {
        throw new Error('Failed to fetch dataset analysis');
      }
      const data = await response.json();
      setAnalysisData(data);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600 text-lg">Analyzing datasets...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <svg
            className="w-16 h-16 text-red-500 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p className="text-red-600 font-medium text-lg">Error loading analysis</p>
          <p className="text-gray-600 mt-2">{error}</p>
          <button
            onClick={fetchAnalysis}
            className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!analysisData) {
    return null;
  }

  const renderStatsCard = (title: string, stats: DatasetStats, color: string) => {
    const colorClasses = {
      blue: 'bg-blue-50 border-blue-300',
      purple: 'bg-purple-50 border-purple-300',
    };

    return (
      <div className={`rounded-lg border-2 p-6 ${colorClasses[color as keyof typeof colorClasses]}`}>
        <h3 className="text-2xl font-bold mb-6 text-gray-800">{title}</h3>
        
        {/* Main Stats Grid */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
            <p className="text-sm text-black-600= mb-1">Total Files</p>
            <p className="text-3xl font-bold text-gray-800">{stats.total_files}</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
            <p className="text-sm text-gray-600 mb-1">Average Size</p>
            <p className="text-2xl font-bold text-gray-800">{stats.avg_size_mb}</p>
            <p className="text-xs text-gray-500">MB per file</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
            <p className="text-sm text-gray-600 mb-1">Size Range</p>
            <p className="text-lg font-bold text-gray-800">{stats.min_size_mb} - {stats.max_size_mb}</p>
            <p className="text-xs text-gray-500">MB</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
            <p className="text-sm text-gray-600 mb-1">Unique Dimensions</p>
            <p className="text-2xl font-bold text-gray-800">{stats.unique_dimensions}</p>
            <p className="text-xs text-gray-500">Different sizes</p>
          </div>
        </div>

        {/* File Types */}
        <div className="bg-white rounded-lg p-4 shadow-sm mb-4 border border-gray-200">
          <h4 className="font-semibold mb-3 text-gray-800">File Types Distribution</h4>
          <div className="space-y-3">
            {Object.entries(stats.file_types).map(([ext, count]) => (
              <div key={ext}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="font-medium text-gray-700">{ext}</span>
                  <span className="font-bold text-gray-800">{count} ({((count / stats.total_files) * 100).toFixed(1)}%)</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`${color === 'blue' ? 'bg-blue-500' : 'bg-purple-500'} h-2 rounded-full`}
                    style={{ width: `${(count / stats.total_files) * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Dimensions */}
        {Object.keys(stats.dimensions).length > 0 && (
          <div className="bg-white rounded-lg p-4 shadow-sm mb-4 border border-gray-200">
            <h4 className="font-semibold mb-3 text-gray-800">Image Dimensions</h4>
            <div className="space-y-3">
              {Object.entries(stats.dimensions).slice(0, 5).map(([dim, count]) => (
                <div key={dim}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-gray-700">{dim}</span>
                    <span className="font-bold text-gray-800">{count} files ({((count / stats.total_files) * 100).toFixed(1)}%)</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`${color === 'blue' ? 'bg-blue-500' : 'bg-purple-500'} h-2 rounded-full`}
                      style={{ width: `${(count / stats.total_files) * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
              {Object.keys(stats.dimensions).length > 5 && (
                <p className="text-xs text-gray-500 italic mt-2">
                  + {Object.keys(stats.dimensions).length - 5} more dimension(s)
                </p>
              )}
            </div>
          </div>
        )}

        {/* Sample Files */}
        <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
          <h4 className="font-semibold mb-3 text-gray-800">Sample Files</h4>
          <div className="space-y-1">
            {stats.sample_images.map((filename, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <span className="text-xs font-bold text-gray-500">#{idx + 1}</span>
                <p className="text-xs text-gray-700 truncate font-mono flex-1">
                  {filename}
                </p>
              </div>
            ))}
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
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Dataset Analysis</h1>
          <p className="text-gray-600">
            Comprehensive analysis of MRI and PET imaging datasets
          </p>
        </div>

        {/* Overall Comparison */}
        <div className="bg-linear-to-r from-blue-600 to-purple-600 rounded-lg p-6 mb-8 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-white">Overall Statistics</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
              <p className="text-sm text-black mb-1">Total Files</p>
              <p className="text-3xl font-bold text-black">{analysisData.comparison.total_files}</p>
            </div>
            <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
              <p className="text-sm text-black mb-1">MRI Files</p>
              <p className="text-3xl font-bold text-black">{analysisData.comparison.mri_percentage}%</p>
              <p className="text-xs text-black">{analysisData.mri.total_files} files</p>
            </div>
            <div className="bg-white bg-opacity-20 rounded-lg p-4 backdrop-blur-sm">
              <p className="text-sm text-black mb-1">PET Files</p>
              <p className="text-3xl font-bold text-black">{analysisData.comparison.pet_percentage}%</p>
              <p className="text-xs text-black">{analysisData.pet.total_files} files</p>
            </div>
          </div>
          
          {/* Distribution Bar */}
          <div className="mt-6">
            <p className="text-sm mb-2 text-white">Dataset Distribution</p>
            <div className="flex h-10 rounded-lg overflow-hidden shadow-lg border-2 border-white">
              <div
                className="bg-blue-400 flex items-center justify-center font-bold text-sm text-gray-900"
                style={{ width: `${analysisData.comparison.mri_percentage}%` }}
              >
                {analysisData.comparison.mri_percentage > 10 && `MRI ${analysisData.comparison.mri_percentage}%`}
              </div>
              <div
                className="bg-purple-400 flex items-center justify-center font-bold text-sm text-gray-900"
                style={{ width: `${analysisData.comparison.pet_percentage}%` }}
              >
                {analysisData.comparison.pet_percentage > 10 && `PET ${analysisData.comparison.pet_percentage}%`}
              </div>
            </div>
          </div>

        </div>

        {/* Detailed Stats */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {renderStatsCard('MRI Dataset', analysisData.mri, 'blue')}
          {renderStatsCard('PET Dataset', analysisData.pet, 'purple')}
        </div>

        {/* Refresh Button */}
        <div className="mt-8 text-center">
          <button
            onClick={fetchAnalysis}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors shadow-md font-medium"
          >
            Refresh Analysis
          </button>
        </div>
      </div>
    </div>
  );
};

export default DatasetAnalysis;
