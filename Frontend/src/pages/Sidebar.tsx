import { useEffect, useState } from 'react';

interface FileNode {
  name: string;
  type: 'file' | 'folder';
  path: string;
  children?: FileNode[];
}

interface SidebarProps {
  onFileSelect?: (filePath: string) => void;
}

const Sidebar = ({ onFileSelect }: SidebarProps) => {
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastTimestamp, setLastTimestamp] = useState<number>(0);

  useEffect(() => {
    fetchFileTree();
    const pollInterval = setInterval(checkForUpdates, 2000);
    return () => clearInterval(pollInterval);
  }, [lastTimestamp]);

  const checkForUpdates = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/file-tree/timestamp');
      if (!response.ok) return;

      const data = await response.json();
      const newTimestamp = data.timestamp;

      if (lastTimestamp > 0 && newTimestamp !== lastTimestamp) {
        await fetchFileTree();
      }

      setLastTimestamp(newTimestamp);
    } catch (err) {
      console.error('Error checking for updates:', err);
    }
  };

  const fetchFileTree = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/file-tree');
      if (!response.ok) {
        throw new Error('Failed to fetch file tree');
      }
      const data = await response.json();
      setFileTree(data);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setLoading(false);
    }
  };

  const toggleFolder = (path: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      newSet.has(path) ? newSet.delete(path) : newSet.add(path);
      return newSet;
    });
  };

  const handleFileClick = (path: string) => {
    if (onFileSelect) onFileSelect(path);
  };

  const renderFileNode = (node: FileNode, depth: number = 0) => {
    const isExpanded = expandedFolders.has(node.path);
    const indent = `${depth * 1.1}rem`;

    if (node.type === 'folder') {
      return (
        <div key={node.path}>
          <div
            className="flex items-center py-2 px-3 cursor-pointer hover:bg-gray-50 transition-colors"
            style={{ paddingLeft: indent }}
            onClick={() => toggleFolder(node.path)}
          >
            <svg
              className={`w-4 h-4 mr-2 transform transition-transform ${
                isExpanded ? 'rotate-90' : ''
              }`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M7 5l6 5-6 5V5z" />
            </svg>

            <svg className="w-5 h-5 mr-2 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
            </svg>

            <span className="text-sm text-gray-700 font-medium">{node.name}</span>
          </div>

          {isExpanded && node.children && (
            <div>
              {node.children.map(child => renderFileNode(child, depth + 1))}
            </div>
          )}
        </div>
      );
    }

    return (
      <div
        key={node.path}
        className="flex items-center py-2 px-3 cursor-pointer hover:bg-blue-50 transition-colors"
        style={{ paddingLeft: indent }}
        onClick={() => handleFileClick(node.path)}
      >
        <svg className="w-4 h-4 mr-2 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
          <path d="M4 3h6l6 6v8a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2z" />
        </svg>

        <span className="text-sm text-gray-600">{node.name}</span>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="w-80 shrink-0 h-full flex items-center justify-center bg-white border-r border-gray-200">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-3 text-gray-600 text-sm">Loading files…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-80 shrink-0 h-full flex items-center justify-center bg-white border-r border-gray-200">
        <div className="text-center px-4">
          <svg className="w-10 h-10 text-red-500 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12A9 9 0 113 12a9 9 0 0118 0z" />
          </svg>

          <p className="text-red-600 font-medium text-sm">Error loading files</p>
          <p className="text-gray-600 text-xs mt-1">{error}</p>

          <button
            onClick={fetchFileTree}
            className="mt-4 px-4 py-2 text-sm bg-blue-600 text-white rounded"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 shrink-0 h-full bg-white border-r border-gray-200 flex flex-col overflow-hidden">
      <div className="shrink-0 bg-white border-b border-gray-200 px-4 py-4">
        <h2 className="text-base font-semibold text-gray-900">PET-MRI Files</h2>
        <p className="text-xs text-gray-500 mt-1">D:\AD-Detection\Python\PET-MRI</p>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {fileTree.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-500 text-sm">No files found</div>
        ) : (
          fileTree.map(node => renderFileNode(node))
        )}
      </div>
    </div>
  );
};

export default Sidebar;
