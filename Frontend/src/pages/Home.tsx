const Home = () => {
  return (
    <div className="flex-1 p-10">
      <div className="max-w-4xl mx-auto">

        <h1 className="text-3xl font-semibold text-gray-900 mb-2">
          AD Detection Application
        </h1>

        <p className="text-base text-gray-600 mb-10">
          Alzheimer's Disease detection using PET and MRI imaging.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* CARD */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-blue-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 1: Dataset Analysis</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              File count, types, dimensions and dataset structure analytics.
            </p>
          </div>

          {/* CARD */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-purple-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 2: Preprocessing</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Resize, normalize, augment and batch images with progress tracking.
            </p>
          </div>

          {/* CARD */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-green-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 3: Feature Extraction</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              CNN + Transformer-based deep feature extraction with multimodal fusion.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-indigo-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 4: Feature Combination</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Fuse paired MRI and PET embeddings into one combined multimodal feature vector.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-emerald-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-6h13M9 5l-7 7 7 7" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 5: Classification Training</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Create pseudo-labels and train the classifier head across multiple epochs.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-rose-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-rose-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3v18m0 0l-4-4m4 4l4-4M5 8h14" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 6: Model Evaluation</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Measure accuracy, precision, recall, F1 score, and confusion matrix on the test set.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-amber-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14m-6 4h2a2 2 0 002-2V8a2 2 0 00-2-2H9m-4 12h2a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 7: Grad-CAM</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Highlight the MRI and PET regions that most influenced the model prediction.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-cyan-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-cyan-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">Module 8: Model Export</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Package the trained weights and metadata for reuse in another application backend.
            </p>
          </div>

          {/* CARD */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <div className="w-11 h-11 bg-orange-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="ml-4 text-lg font-medium text-gray-900">File Browser</h3>
            </div>
            <p className="text-gray-600 text-sm leading-6">
              Browse PET-MRI directories with live updates across pipeline stages.
            </p>
          </div>

        </div>

        {/* Getting Started Box */}
        <div className="mt-10 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h4 className="text-lg font-semibold text-blue-900 mb-3">
            Getting Started
          </h4>

          <ul className="text-blue-800 space-y-2 text-sm leading-6">
            <li>Explore the file structure from the left sidebar</li>
            <li>Expand folders to inspect MRI/PET image groups</li>
            <li>Click a file to open it and view metadata</li>
          </ul>
        </div>

      </div>
    </div>
  );
};

export default Home;
