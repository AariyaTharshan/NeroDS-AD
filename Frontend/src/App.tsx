import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Sidebar from './pages/Sidebar';
import Home from './pages/Home';
import DatasetAnalysis from './pages/DatasetAnalysis';
import Preprocessing from './pages/Preprocessing';
import FeatureExtraction from './pages/FeatureExtraction';
import FeatureCombination from './pages/FeatureCombination';
import ClassificationTraining from './pages/ClassificationTraining';
import ModelEvaluation from './pages/ModelEvaluation';
import GradCAM from './pages/GradCAM';
import ModelExport from './pages/ModelExport';
function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-3">
      <div className="flex items-center gap-10">

        {/* Title */}
        <h1 className="text-lg font-semibold text-gray-900 tracking-tight">
          AD Detection System
        </h1>

        {/* Links */}
        <div className="flex items-center gap-1">

          <Link
            to="/"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/') 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Home
          </Link>

          <Link
            to="/dataset-analysis"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/dataset-analysis')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Dataset Analysis
          </Link>

          <Link
            to="/preprocessing"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/preprocessing')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Preprocessing
          </Link>

          <Link
            to="/feature-extraction"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/feature-extraction')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Feature Extraction
          </Link>

          <Link
            to="/feature-combination"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/feature-combination')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Feature Combine
          </Link>

          <Link
            to="/classification-training"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/classification-training')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Classification
          </Link>

          <Link
            to="/model-evaluation"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/model-evaluation')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Evaluation
          </Link>

          <Link
            to="/gradcam"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/gradcam')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Grad-CAM
          </Link>

          <Link
            to="/model-export"
            className={`px-4 py-2 rounded-md text-sm transition-colors
              ${isActive('/model-export')
                ? 'bg-blue-600 text-white'
                : 'text-gray-700 hover:bg-gray-100'
              }`}
          >
            Export
          </Link>

        </div>
      </div>
    </nav>
  );
}

function App() {
  const handleFileSelect = (filePath: string) => {
    console.log('Selected file:', filePath);
    // You can add logic here to handle file selection
  };

  return (
    <Router>
      <div className="flex h-screen bg-gray-50 overflow-hidden">
        <Sidebar onFileSelect={handleFileSelect} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Navigation />
          <div className="flex-1 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/dataset-analysis" element={<DatasetAnalysis />} />
              <Route path="/preprocessing" element={<Preprocessing />} />
              <Route path="/feature-extraction" element={<FeatureExtraction />} />
              <Route path="/feature-combination" element={<FeatureCombination />} />
              <Route path="/classification-training" element={<ClassificationTraining />} />
              <Route path="/model-evaluation" element={<ModelEvaluation />} />
              <Route path="/gradcam" element={<GradCAM />} />
              <Route path="/model-export" element={<ModelExport />} />
          </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;
