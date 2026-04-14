from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pathlib import Path
import io
import base64
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
import numpy as np
from PIL import Image
from collections import defaultdict
import shutil
import random
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)
CORS(app)

# Path to the PET-MRI directory
PET_MRI_PATH = r"D:\FYP\PET-MRI"

# Global variable to track last modification time
last_modified = time.time()
file_tree_cache = None
cache_lock = threading.Lock()

class DirectoryChangeHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def on_any_event(self, event):
        global last_modified
        # Update the last modified timestamp whenever any change occurs
        with cache_lock:
            last_modified = time.time()
        print(f"Change detected: {event.event_type} - {event.src_path}")

# Initialize file system observer
observer = Observer()
event_handler = DirectoryChangeHandler()
observer.schedule(event_handler, PET_MRI_PATH, recursive=True)
observer.start()

def build_file_tree(directory_path):
    """
    Recursively build a tree structure of files and folders.
    """
    tree = []
    
    try:
        # Get all items in the directory
        items = sorted(os.listdir(directory_path))
        
        for item in items:
            item_path = os.path.join(directory_path, item)
            
            if os.path.isdir(item_path):
                # It's a folder
                folder_node = {
                    'name': item,
                    'type': 'folder',
                    'path': item_path,
                    'children': build_file_tree(item_path)
                }
                tree.append(folder_node)
            else:
                # It's a file
                file_node = {
                    'name': item,
                    'type': 'file',
                    'path': item_path
                }
                tree.append(file_node)
    
    except PermissionError:
        print(f"Permission denied: {directory_path}")
    except Exception as e:
        print(f"Error reading directory {directory_path}: {str(e)}")
    
    return tree

@app.route('/api/file-tree', methods=['GET'])
def get_file_tree():
    """
    API endpoint to get the file tree structure of PET-MRI directory.
    """
    global file_tree_cache
    
    try:
        if not os.path.exists(PET_MRI_PATH):
            return jsonify({'error': 'PET-MRI directory not found'}), 404
        
        with cache_lock:
            file_tree = build_file_tree(PET_MRI_PATH)
            file_tree_cache = file_tree
        
        return jsonify(file_tree)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/file-tree/timestamp', methods=['GET'])
def get_timestamp():
    """
    API endpoint to get the last modification timestamp.
    Frontend can poll this to check if the tree needs to be refreshed.
    """
    with cache_lock:
        return jsonify({'timestamp': last_modified})

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

def analyze_image_folder(folder_path):
    """
    Analyze all images in a folder and return statistics.
    """
    stats = {
        'total_files': 0,
        'total_size_mb': 0,
        'file_types': defaultdict(int),
        'dimensions': defaultdict(int),
        'file_list': [],
        'avg_size_mb': 0,
        'min_size_mb': float('inf'),
        'max_size_mb': 0,
        'sample_images': [],
        'size_distribution': {
            'tiny': 0,      # < 0.1 MB
            'small': 0,     # 0.1 - 1 MB
            'medium': 0,    # 1 - 10 MB
            'large': 0,     # 10 - 100 MB
            'huge': 0       # > 100 MB
        },
        'unique_dimensions': 0,
        'largest_file': {'name': '', 'size_mb': 0},
        'smallest_file': {'name': '', 'size_mb': float('inf')}
    }
    
    if not os.path.exists(folder_path):
        return stats
    
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        stats['total_files'] = len(files)
        
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Get file size
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                stats['total_size_mb'] += file_size_mb
                stats['min_size_mb'] = min(stats['min_size_mb'], file_size_mb)
                stats['max_size_mb'] = max(stats['max_size_mb'], file_size_mb)
                
                # Track largest and smallest files
                if file_size_mb > stats['largest_file']['size_mb']:
                    stats['largest_file'] = {'name': filename, 'size_mb': round(file_size_mb, 3)}
                if file_size_mb < stats['smallest_file']['size_mb']:
                    stats['smallest_file'] = {'name': filename, 'size_mb': round(file_size_mb, 3)}
                
                # Size distribution
                if file_size_mb < 0.1:
                    stats['size_distribution']['tiny'] += 1
                elif file_size_mb < 1:
                    stats['size_distribution']['small'] += 1
                elif file_size_mb < 10:
                    stats['size_distribution']['medium'] += 1
                elif file_size_mb < 100:
                    stats['size_distribution']['large'] += 1
                else:
                    stats['size_distribution']['huge'] += 1
                
                # Get file extension
                _, ext = os.path.splitext(filename)
                ext = ext.lower() if ext else 'no_extension'
                stats['file_types'][ext] += 1
                
                # Try to get image dimensions
                try:
                    if ext in ['.npy']:
                        # Handle numpy files
                        arr = np.load(file_path)
                        dimension_key = f"{arr.shape}"
                        stats['dimensions'][dimension_key] += 1
                    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                        # Handle image files
                        with Image.open(file_path) as img:
                            dimension_key = f"{img.width}x{img.height}"
                            stats['dimensions'][dimension_key] += 1
                except Exception as e:
                    print(f"Could not read dimensions for {filename}: {str(e)}")
                
                # Add to file list
                stats['file_list'].append({
                    'name': filename,
                    'size_mb': round(file_size_mb, 3),
                    'extension': ext
                })
                
                # Add first 5 files as samples
                if len(stats['sample_images']) < 5:
                    stats['sample_images'].append(filename)
                    
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        # Calculate average size
        if stats['total_files'] > 0:
            stats['avg_size_mb'] = stats['total_size_mb'] / stats['total_files']
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['file_types'] = dict(stats['file_types'])
        stats['dimensions'] = dict(stats['dimensions'])
        stats['unique_dimensions'] = len(stats['dimensions'])
        
        # Round total size
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        stats['avg_size_mb'] = round(stats['avg_size_mb'], 3)
        stats['min_size_mb'] = round(stats['min_size_mb'], 3) if stats['min_size_mb'] != float('inf') else 0
        stats['max_size_mb'] = round(stats['max_size_mb'], 3)
        
        # Clean up smallest file if it wasn't set
        if stats['smallest_file']['size_mb'] == float('inf'):
            stats['smallest_file'] = {'name': '', 'size_mb': 0}
        
    except Exception as e:
        print(f"Error analyzing folder {folder_path}: {str(e)}")
    
    return stats

@app.route('/api/dataset-analysis', methods=['GET'])
def get_dataset_analysis():
    """
    API endpoint to get comprehensive analysis of MRI and PET datasets.
    """
    try:
        mri_path = os.path.join(PET_MRI_PATH, 'MRI')
        pet_path = os.path.join(PET_MRI_PATH, 'PET')
        
        analysis = {
            'mri': analyze_image_folder(mri_path),
            'pet': analyze_image_folder(pet_path),
            'timestamp': time.time()
        }
        
        # Add comparison statistics
        analysis['comparison'] = {
            'total_files': analysis['mri']['total_files'] + analysis['pet']['total_files'],
            'total_size_mb': round(analysis['mri']['total_size_mb'] + analysis['pet']['total_size_mb'], 2),
            'mri_percentage': round((analysis['mri']['total_files'] / (analysis['mri']['total_files'] + analysis['pet']['total_files']) * 100), 1) if (analysis['mri']['total_files'] + analysis['pet']['total_files']) > 0 else 0,
            'pet_percentage': round((analysis['pet']['total_files'] / (analysis['mri']['total_files'] + analysis['pet']['total_files']) * 100), 1) if (analysis['mri']['total_files'] + analysis['pet']['total_files']) > 0 else 0
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== FEATURE EXTRACTION MODULE ====================

class CNNTransformerFeatureExtractor(nn.Module):
    """CNN + Transformer model for feature extraction from medical images"""
    def __init__(self, num_channels=3, cnn_out_channels=32, num_heads=8, num_layers=2, pretrained=True):
        super(CNNTransformerFeatureExtractor, self).__init__()

        # Custom CNN branch (lightweight, task-specific)
        self.custom_cnn = nn.Sequential(
            nn.Conv2d(num_channels, cnn_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(cnn_out_channels, cnn_out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(cnn_out_channels * 2, cnn_out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.custom_out_channels = cnn_out_channels * 4

        # Pretrained EfficientNet-B0 branch
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.pretrained_cnn = nn.Sequential(*list(backbone.features), nn.AdaptiveAvgPool2d((7, 7)))
        self.pretrained_out_channels = 1280

        # Combined output channels
        self.total_channels = self.custom_out_channels + self.pretrained_out_channels

        # Positional embedding for Transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, self.total_channels))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_channels,
            nhead=num_heads,
            batch_first=True,
            dropout=0.1,
            dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Custom CNN features
        custom_feat = self.custom_cnn(x)  # (B, C1, 7, 7)
        # Pretrained CNN features
        pretrained_feat = self.pretrained_cnn(x)  # (B, C2, 7, 7)

        # Concatenate along channel dimension
        combined_feat = torch.cat([custom_feat, pretrained_feat], dim=1)  # (B, C1+C2, 7, 7)
        B, C, H, W = combined_feat.shape

        # Flatten spatial dimensions for Transformer
        transformer_input = combined_feat.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, C)

        # Add positional embedding
        transformer_input = transformer_input + self.pos_embedding[:, :transformer_input.size(1)]

        # Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Mean pooling over tokens
        pooled_features = transformer_output.mean(dim=1)  # (B, total_channels)

        return pooled_features

class ClassificationHead(nn.Module):
    """Classification head for combined multimodal features"""
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

class CombinedFeatureDataset(Dataset):
    """Dataset for combined feature vectors and labels"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MultimodalClassificationModel(nn.Module):
    """Wrap two feature extractors and the classification head for Grad-CAM"""
    def __init__(self, mri_extractor, pet_extractor, classifier_head):
        super(MultimodalClassificationModel, self).__init__()
        self.mri_extractor = mri_extractor
        self.pet_extractor = pet_extractor
        self.classifier_head = classifier_head

    def forward(self, mri_tensor, pet_tensor):
        mri_features = self.mri_extractor(mri_tensor)
        pet_features = self.pet_extractor(pet_tensor)
        combined_features = torch.cat([mri_features, pet_features], dim=1)
        return self.classifier_head(combined_features)

class GradCAM:
    """Simple Grad-CAM implementation for convolutional target layers"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, target_score, output_size):
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise Exception("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=output_size,
            mode='bilinear',
            align_corners=False
        )
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

# Global feature extraction state
feature_extraction_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None
}
feature_extraction_lock = threading.Lock()
feature_extractor_model = None

# Global feature combination state
feature_combination_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None
}
feature_combination_lock = threading.Lock()
classifier_training_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'latest_metrics': None
}
classifier_training_lock = threading.Lock()
evaluation_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None
}
evaluation_lock = threading.Lock()
gradcam_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None
}
gradcam_lock = threading.Lock()
model_export_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None
}
model_export_lock = threading.Lock()

def log_feature_extraction(message):
    """Add a log message to feature extraction state"""
    with feature_extraction_lock:
        feature_extraction_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        print(message)

def log_feature_combination(message):
    """Add a log message to feature combination state"""
    with feature_combination_lock:
        feature_combination_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        print(message)

def log_classifier_training(message):
    """Add a log message to classifier training state"""
    with classifier_training_lock:
        classifier_training_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        print(message)

def log_evaluation(message):
    """Add a log message to evaluation state"""
    with evaluation_lock:
        evaluation_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        print(message)

def log_gradcam(message):
    """Add a log message to Grad-CAM state"""
    with gradcam_lock:
        gradcam_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        print(message)

def log_model_export(message):
    """Add a log message to model export state"""
    with model_export_lock:
        model_export_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        print(message)

def encode_image_to_base64(image_array):
    """Encode an RGB numpy image array as base64 PNG"""
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_heatmap_overlay(image_array, heatmap):
    """Blend a normalized heatmap onto an RGB image"""
    image = image_array.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    heat = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    heat_rgb = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    overlay = (0.6 * image + 0.4 * heat_rgb)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    original = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return original, overlay

def run_feature_extraction_pipeline():
    """Run the complete feature extraction pipeline in a separate thread"""
    global feature_extractor_model
    
    try:
        with feature_extraction_lock:
            feature_extraction_state['is_running'] = True
            feature_extraction_state['current_step'] = 'Initializing'
            feature_extraction_state['progress'] = 0
            feature_extraction_state['logs'] = []
            feature_extraction_state['results'] = {}
            feature_extraction_state['error'] = None
        
        log_feature_extraction("Starting feature extraction pipeline...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_feature_extraction(f"Using device: {device}")
        
        # Initialize model
        with feature_extraction_lock:
            feature_extraction_state['current_step'] = 'Loading Model'
            feature_extraction_state['progress'] = 5
        
        log_feature_extraction("Initializing CNN + Transformer model...")
        feature_extractor_model = CNNTransformerFeatureExtractor(
            num_channels=3,
            cnn_out_channels=64,
            num_heads=8,
            num_layers=2
        )
        feature_extractor_model.to(device)
        feature_extractor_model.eval()
        
        param_count = sum(p.numel() for p in feature_extractor_model.parameters())
        log_feature_extraction(f"Model loaded with {param_count:,} parameters")
        
        # Load batched data
        with feature_extraction_lock:
            feature_extraction_state['current_step'] = 'Loading Batched Data'
            feature_extraction_state['progress'] = 15
        
        batched_path = os.path.join(PET_MRI_PATH, 'batched_images')
        if not os.path.exists(batched_path):
            raise Exception("Batched images directory not found. Please run preprocessing first.")
        
        log_feature_extraction(f"Loading batched images from {batched_path}")
        
        # Load MRI batches
        mri_batch_files = sorted([f for f in os.listdir(batched_path) if f.startswith("batch_mri_") and f.endswith(".npy")])
        pet_batch_files = sorted([f for f in os.listdir(batched_path) if f.startswith("batch_pet_") and f.endswith(".npy")])
        
        log_feature_extraction(f"Found {len(mri_batch_files)} MRI batches and {len(pet_batch_files)} PET batches")
        
        loaded_mri_batches = [np.load(os.path.join(batched_path, f)) for f in mri_batch_files]
        loaded_pet_batches = [np.load(os.path.join(batched_path, f)) for f in pet_batch_files]
        
        # Extract MRI features
        with feature_extraction_lock:
            feature_extraction_state['current_step'] = 'Extracting MRI Features'
            feature_extraction_state['progress'] = 25
        
        log_feature_extraction("Extracting features from MRI batches...")
        extraction_start = time.time()
        
        extracted_mri_features = []
        total_mri_batches = len(loaded_mri_batches)
        
        for batch_idx, batch in enumerate(loaded_mri_batches):
            batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
            
            with torch.no_grad():
                features = feature_extractor_model(batch_tensor)
                features = features.cpu().numpy()
            
            for feat in features:
                extracted_mri_features.append(feat)
            
            # Update progress (25% to 55%)
            progress = 25 + int((batch_idx + 1) / total_mri_batches * 30)
            with feature_extraction_lock:
                feature_extraction_state['progress'] = progress
            
            if (batch_idx + 1) % 10 == 0:
                log_feature_extraction(f"Processed {batch_idx + 1}/{total_mri_batches} MRI batches")
        
        log_feature_extraction(f"Extracted features from {len(extracted_mri_features)} MRI images")
        
        # Extract PET features
        with feature_extraction_lock:
            feature_extraction_state['current_step'] = 'Extracting PET Features'
            feature_extraction_state['progress'] = 55
        
        log_feature_extraction("Extracting features from PET batches...")
        
        extracted_pet_features = []
        total_pet_batches = len(loaded_pet_batches)
        
        for batch_idx, batch in enumerate(loaded_pet_batches):
            batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
            
            with torch.no_grad():
                features = feature_extractor_model(batch_tensor)
                features = features.cpu().numpy()
            
            for feat in features:
                extracted_pet_features.append(feat)
            
            # Update progress (55% to 85%)
            progress = 55 + int((batch_idx + 1) / total_pet_batches * 30)
            with feature_extraction_lock:
                feature_extraction_state['progress'] = progress
            
            if (batch_idx + 1) % 10 == 0:
                log_feature_extraction(f"Processed {batch_idx + 1}/{total_pet_batches} PET batches")
        
        log_feature_extraction(f"Extracted features from {len(extracted_pet_features)} PET images")
        
        extraction_time = time.time() - extraction_start
        
        # Convert to numpy arrays
        extracted_mri_features = np.array(extracted_mri_features)
        extracted_pet_features = np.array(extracted_pet_features)
        
        # Calculate statistics
        with feature_extraction_lock:
            feature_extraction_state['current_step'] = 'Computing Statistics'
            feature_extraction_state['progress'] = 90
        
        log_feature_extraction("Computing feature statistics...")
        
        results = {
            'mri_features_count': len(extracted_mri_features),
            'pet_features_count': len(extracted_pet_features),
            'feature_dimension': extracted_mri_features.shape[1],
            'total_samples': len(extracted_mri_features),
            'extraction_time': round(extraction_time, 2),
            'feature_stats': {
                'mri_mean': float(extracted_mri_features.mean()),
                'mri_std': float(extracted_mri_features.std()),
                'pet_mean': float(extracted_pet_features.mean()),
                'pet_std': float(extracted_pet_features.std())
            }
        }
        
        # Save features to disk
        log_feature_extraction("Saving extracted features...")
        features_output_path = os.path.join(PET_MRI_PATH, 'extracted_features')
        os.makedirs(features_output_path, exist_ok=True)
        
        np.save(os.path.join(features_output_path, 'mri_features.npy'), extracted_mri_features)
        np.save(os.path.join(features_output_path, 'pet_features.npy'), extracted_pet_features)
        torch.save(
            feature_extractor_model.state_dict(),
            os.path.join(features_output_path, 'feature_extractor_model.pth')
        )
        
        log_feature_extraction(f"Features saved to {features_output_path}")
        
        with feature_extraction_lock:
            feature_extraction_state['results'] = results
            feature_extraction_state['progress'] = 100
            feature_extraction_state['current_step'] = 'Complete'
        
        log_feature_extraction("Feature extraction pipeline completed successfully!")
        log_feature_extraction(f"Total extraction time: {extraction_time:.2f}s")
        log_feature_extraction(f"Average time per sample: {extraction_time / results['total_samples']:.4f}s")
        
    except Exception as e:
        with feature_extraction_lock:
            feature_extraction_state['error'] = str(e)
            feature_extraction_state['current_step'] = 'Error'
        log_feature_extraction(f"Error in feature extraction pipeline: {str(e)}")
    finally:
        with feature_extraction_lock:
            feature_extraction_state['is_running'] = False

@app.route('/api/feature-extraction/start', methods=['POST'])
def start_feature_extraction():
    """Start the feature extraction pipeline"""
    with feature_extraction_lock:
        if feature_extraction_state['is_running']:
            return jsonify({'error': 'Feature extraction already running'}), 400
    
    # Start feature extraction in a separate thread
    thread = threading.Thread(target=run_feature_extraction_pipeline)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Feature extraction started', 'status': 'running'})

@app.route('/api/feature-extraction/status', methods=['GET'])
def get_feature_extraction_status():
    """Get current feature extraction status"""
    with feature_extraction_lock:
        return jsonify({
            'status': 'completed' if feature_extraction_state['progress'] == 100 and not feature_extraction_state['is_running'] else ('error' if feature_extraction_state['error'] else 'running'),
            'current_step': feature_extraction_state['current_step'],
            'progress': feature_extraction_state['progress'],
            'logs': feature_extraction_state['logs'][-50:],  # Last 50 logs
            'error': feature_extraction_state['error']
        })

@app.route('/api/feature-extraction/results', methods=['GET'])
def get_feature_extraction_results():
    """Get final feature extraction results"""
    try:
        with feature_extraction_lock:
            if not feature_extraction_state['results']:
                return jsonify({'error': 'No results available. Please run feature extraction first.'}), 404
            
            return jsonify(feature_extraction_state['results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_feature_combination_pipeline():
    """Combine extracted MRI and PET features into paired multimodal feature vectors"""
    try:
        with feature_combination_lock:
            feature_combination_state['is_running'] = True
            feature_combination_state['current_step'] = 'Initializing'
            feature_combination_state['progress'] = 0
            feature_combination_state['logs'] = []
            feature_combination_state['results'] = {}
            feature_combination_state['error'] = None

        log_feature_combination("Starting feature combination pipeline...")

        features_path = os.path.join(PET_MRI_PATH, 'extracted_features')
        mri_features_path = os.path.join(features_path, 'mri_features.npy')
        pet_features_path = os.path.join(features_path, 'pet_features.npy')

        with feature_combination_lock:
            feature_combination_state['current_step'] = 'Loading Extracted Features'
            feature_combination_state['progress'] = 15

        if not os.path.exists(mri_features_path) or not os.path.exists(pet_features_path):
            raise Exception("Extracted feature files not found. Please run feature extraction first.")

        log_feature_combination("Loading MRI and PET feature arrays...")
        extracted_mri_features = np.load(mri_features_path)
        extracted_pet_features = np.load(pet_features_path)

        if extracted_mri_features.size == 0 or extracted_pet_features.size == 0:
            raise Exception("Feature arrays are empty. Please rerun feature extraction.")

        if extracted_mri_features.shape[0] != extracted_pet_features.shape[0]:
            raise Exception(
                f"Feature count mismatch: MRI={extracted_mri_features.shape[0]}, PET={extracted_pet_features.shape[0]}"
            )

        log_feature_combination(
            f"Loaded {extracted_mri_features.shape[0]} paired samples with "
            f"MRI dim {extracted_mri_features.shape[1]} and PET dim {extracted_pet_features.shape[1]}"
        )

        with feature_combination_lock:
            feature_combination_state['current_step'] = 'Combining Paired Features'
            feature_combination_state['progress'] = 55

        combination_start = time.time()
        combined_features = np.concatenate((extracted_mri_features, extracted_pet_features), axis=1)
        combination_time = time.time() - combination_start

        log_feature_combination("Computing combined feature statistics...")
        with feature_combination_lock:
            feature_combination_state['current_step'] = 'Computing Statistics'
            feature_combination_state['progress'] = 80

        combined_output_path = os.path.join(features_path, 'combined_features.npy')
        np.save(combined_output_path, combined_features)

        results = {
            'paired_samples': int(combined_features.shape[0]),
            'mri_feature_dimension': int(extracted_mri_features.shape[1]),
            'pet_feature_dimension': int(extracted_pet_features.shape[1]),
            'combined_feature_dimension': int(combined_features.shape[1]),
            'output_file': combined_output_path,
            'combination_time': round(combination_time, 4),
            'combined_stats': {
                'mean': float(combined_features.mean()),
                'std': float(combined_features.std()),
                'min': float(combined_features.min()),
                'max': float(combined_features.max())
            }
        }

        with feature_combination_lock:
            feature_combination_state['results'] = results
            feature_combination_state['progress'] = 100
            feature_combination_state['current_step'] = 'Complete'

        log_feature_combination(f"Saved combined features to {combined_output_path}")
        log_feature_combination("Feature combination completed successfully!")

    except Exception as e:
        with feature_combination_lock:
            feature_combination_state['error'] = str(e)
            feature_combination_state['current_step'] = 'Error'
        log_feature_combination(f"Error in feature combination pipeline: {str(e)}")
    finally:
        with feature_combination_lock:
            feature_combination_state['is_running'] = False

@app.route('/api/feature-combination/start', methods=['POST'])
def start_feature_combination():
    """Start the feature combination pipeline"""
    with feature_combination_lock:
        if feature_combination_state['is_running']:
            return jsonify({'error': 'Feature combination already running'}), 400

    thread = threading.Thread(target=run_feature_combination_pipeline)
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Feature combination started', 'status': 'running'})

@app.route('/api/feature-combination/status', methods=['GET'])
def get_feature_combination_status():
    """Get current feature combination status"""
    with feature_combination_lock:
        return jsonify({
            'status': 'completed' if feature_combination_state['progress'] == 100 and not feature_combination_state['is_running'] else ('error' if feature_combination_state['error'] else 'running'),
            'current_step': feature_combination_state['current_step'],
            'progress': feature_combination_state['progress'],
            'logs': feature_combination_state['logs'][-50:],
            'error': feature_combination_state['error']
        })

@app.route('/api/feature-combination/results', methods=['GET'])
def get_feature_combination_results():
    """Get feature combination results"""
    try:
        with feature_combination_lock:
            if not feature_combination_state['results']:
                return jsonify({'error': 'No results available. Please run feature combination first.'}), 404

            return jsonify(feature_combination_state['results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_classifier_training_pipeline(num_epochs=50):
    """Train a classification head on combined features using pseudo-labels"""
    try:
        with classifier_training_lock:
            classifier_training_state['is_running'] = True
            classifier_training_state['current_step'] = 'Initializing'
            classifier_training_state['progress'] = 0
            classifier_training_state['logs'] = []
            classifier_training_state['results'] = {}
            classifier_training_state['error'] = None
            classifier_training_state['current_epoch'] = 0
            classifier_training_state['total_epochs'] = num_epochs
            classifier_training_state['latest_metrics'] = None

        log_classifier_training("Starting classifier training pipeline...")

        features_path = os.path.join(PET_MRI_PATH, 'extracted_features')
        combined_features_path = os.path.join(features_path, 'combined_features.npy')

        with classifier_training_lock:
            classifier_training_state['current_step'] = 'Loading Combined Features'
            classifier_training_state['progress'] = 10

        if not os.path.exists(combined_features_path):
            raise Exception("Combined features not found. Please run feature combination first.")

        combined_features = np.load(combined_features_path)
        if combined_features.size == 0:
            raise Exception("Combined features array is empty.")

        log_classifier_training(f"Loaded combined feature matrix with shape {combined_features.shape}")

        with classifier_training_lock:
            classifier_training_state['current_step'] = 'Generating Pseudo Labels'
            classifier_training_state['progress'] = 20

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)

        best_score = -1
        best_n_clusters = 3
        best_labels = None

        for n_clusters in range(2, 6):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, cluster_labels)
            log_classifier_training(f"Silhouette score for {n_clusters} clusters: {score:.4f}")

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = cluster_labels

        if best_labels is None:
            raise Exception("Failed to generate pseudo labels from the combined features.")

        unique_labels, counts = np.unique(best_labels, return_counts=True)
        cluster_distribution = [
            {
                'cluster': int(label),
                'count': int(count),
                'percentage': round(float(count / len(best_labels) * 100), 1)
            }
            for label, count in zip(unique_labels, counts)
        ]

        log_classifier_training(
            f"Selected {best_n_clusters} clusters with silhouette score {best_score:.4f}"
        )

        with classifier_training_lock:
            classifier_training_state['current_step'] = 'Preparing Data Loaders'
            classifier_training_state['progress'] = 30

        features_tensor = torch.from_numpy(combined_features).float()
        labels_tensor = torch.from_numpy(best_labels).long()
        dataset = CombinedFeatureDataset(features_tensor, labels_tensor)

        train_size = int(0.70 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier_head = ClassificationHead(
            input_dim=combined_features.shape[1],
            num_classes=best_n_clusters
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(classifier_head.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        artifacts_path = os.path.join(PET_MRI_PATH, 'classification_artifacts')
        os.makedirs(artifacts_path, exist_ok=True)
        pseudo_labels_path = os.path.join(artifacts_path, 'pseudo_labels.npy')
        model_path = os.path.join(artifacts_path, 'best_classifier_model.pth')
        np.save(pseudo_labels_path, best_labels)

        log_classifier_training(f"Using device: {device}")
        log_classifier_training(
            f"Training for {num_epochs} epochs on {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test samples"
        )

        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        training_history = []

        with classifier_training_lock:
            classifier_training_state['current_step'] = 'Training Epochs'
            classifier_training_state['progress'] = 35

        for epoch in range(num_epochs):
            classifier_head.train()
            train_loss_total = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = classifier_head(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss = train_loss_total / max(len(train_dataset), 1)
            train_accuracy = train_correct / max(train_total, 1)

            classifier_head.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = classifier_head(inputs)
                    loss = criterion(outputs, labels)

                    val_loss_total += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss_total / max(len(val_dataset), 1)
            val_accuracy = val_correct / max(val_total, 1)
            scheduler.step(val_loss)

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': round(float(train_loss), 4),
                'train_accuracy': round(float(train_accuracy), 4),
                'val_loss': round(float(val_loss), 4),
                'val_accuracy': round(float(val_accuracy), 4)
            }
            training_history.append(epoch_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                torch.save(classifier_head.state_dict(), model_path)

            with classifier_training_lock:
                classifier_training_state['current_epoch'] = epoch + 1
                classifier_training_state['progress'] = 35 + int(((epoch + 1) / num_epochs) * 60)
                classifier_training_state['latest_metrics'] = epoch_metrics

            if (epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                log_classifier_training(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss {train_loss:.4f}, Train Acc {train_accuracy:.4f} | "
                    f"Val Loss {val_loss:.4f}, Val Acc {val_accuracy:.4f}"
                )

        results = {
            'num_epochs': num_epochs,
            'num_clusters': int(best_n_clusters),
            'silhouette_score': round(float(best_score), 4),
            'dataset_split': {
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset)
            },
            'cluster_distribution': cluster_distribution,
            'best_metrics': {
                'best_val_loss': round(float(best_val_loss), 4),
                'best_val_accuracy': round(float(best_val_accuracy), 4)
            },
            'latest_metrics': training_history[-1] if training_history else None,
            'history': training_history,
            'artifacts': {
                'model_path': model_path,
                'labels_path': pseudo_labels_path
            }
        }

        with classifier_training_lock:
            classifier_training_state['results'] = results
            classifier_training_state['progress'] = 100
            classifier_training_state['current_step'] = 'Complete'

        log_classifier_training("Classifier training completed successfully!")

    except Exception as e:
        with classifier_training_lock:
            classifier_training_state['error'] = str(e)
            classifier_training_state['current_step'] = 'Error'
        log_classifier_training(f"Error in classifier training pipeline: {str(e)}")
    finally:
        with classifier_training_lock:
            classifier_training_state['is_running'] = False

@app.route('/api/classification-training/start', methods=['POST'])
def start_classification_training():
    """Start classifier training"""
    with classifier_training_lock:
        if classifier_training_state['is_running']:
            return jsonify({'error': 'Classifier training already running'}), 400

    payload = request.get_json(silent=True) or {}
    num_epochs = int(payload.get('num_epochs', 50))
    if num_epochs < 1 or num_epochs > 500:
        return jsonify({'error': 'num_epochs must be between 1 and 500'}), 400

    thread = threading.Thread(target=run_classifier_training_pipeline, args=(num_epochs,))
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Classifier training started', 'status': 'running', 'num_epochs': num_epochs})

@app.route('/api/classification-training/status', methods=['GET'])
def get_classification_training_status():
    """Get current classifier training status"""
    with classifier_training_lock:
        return jsonify({
            'status': 'completed' if classifier_training_state['progress'] == 100 and not classifier_training_state['is_running'] else ('error' if classifier_training_state['error'] else 'running'),
            'current_step': classifier_training_state['current_step'],
            'progress': classifier_training_state['progress'],
            'logs': classifier_training_state['logs'][-50:],
            'error': classifier_training_state['error'],
            'current_epoch': classifier_training_state['current_epoch'],
            'total_epochs': classifier_training_state['total_epochs'],
            'latest_metrics': classifier_training_state['latest_metrics']
        })

@app.route('/api/classification-training/results', methods=['GET'])
def get_classification_training_results():
    """Get classifier training results"""
    try:
        with classifier_training_lock:
            if not classifier_training_state['results']:
                return jsonify({'error': 'No results available. Please run classifier training first.'}), 404

            return jsonify(classifier_training_state['results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_evaluation_pipeline():
    """Evaluate the trained classifier on the held-out test split"""
    try:
        with evaluation_lock:
            evaluation_state['is_running'] = True
            evaluation_state['current_step'] = 'Initializing'
            evaluation_state['progress'] = 0
            evaluation_state['logs'] = []
            evaluation_state['results'] = {}
            evaluation_state['error'] = None

        log_evaluation("Starting model evaluation pipeline...")

        features_path = os.path.join(PET_MRI_PATH, 'extracted_features')
        combined_features_path = os.path.join(features_path, 'combined_features.npy')
        artifacts_path = os.path.join(PET_MRI_PATH, 'classification_artifacts')
        pseudo_labels_path = os.path.join(artifacts_path, 'pseudo_labels.npy')
        model_path = os.path.join(artifacts_path, 'best_classifier_model.pth')

        with evaluation_lock:
            evaluation_state['current_step'] = 'Loading Artifacts'
            evaluation_state['progress'] = 15

        if not os.path.exists(combined_features_path):
            raise Exception("Combined features not found. Please run feature combination first.")
        if not os.path.exists(pseudo_labels_path) or not os.path.exists(model_path):
            raise Exception("Training artifacts not found. Please run classification training first.")

        combined_features = np.load(combined_features_path)
        pseudo_labels = np.load(pseudo_labels_path)

        if combined_features.shape[0] != len(pseudo_labels):
            raise Exception("Combined features and pseudo labels count do not match.")

        log_evaluation(f"Loaded {combined_features.shape[0]} samples for evaluation")

        with evaluation_lock:
            evaluation_state['current_step'] = 'Rebuilding Test Split'
            evaluation_state['progress'] = 35

        features_tensor = torch.from_numpy(combined_features).float()
        labels_tensor = torch.from_numpy(pseudo_labels).long()
        dataset = CombinedFeatureDataset(features_tensor, labels_tensor)

        train_size = int(0.70 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        _, _, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        num_classes = len(np.unique(pseudo_labels))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classifier_head = ClassificationHead(
            input_dim=combined_features.shape[1],
            num_classes=num_classes
        ).to(device)
        classifier_head.load_state_dict(torch.load(model_path, map_location=device))
        classifier_head.eval()

        with evaluation_lock:
            evaluation_state['current_step'] = 'Running Test Evaluation'
            evaluation_state['progress'] = 60

        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = classifier_head(inputs)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        if not all_true_labels:
            raise Exception("Test split is empty; unable to evaluate the model.")

        with evaluation_lock:
            evaluation_state['current_step'] = 'Computing Metrics'
            evaluation_state['progress'] = 85

        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(all_true_labels, all_predictions)

        per_class_accuracy = []
        for class_index in range(num_classes):
            class_mask = np.array(all_true_labels) == class_index
            class_total = int(class_mask.sum())
            class_correct = int((np.array(all_predictions)[class_mask] == class_index).sum()) if class_total > 0 else 0
            per_class_accuracy.append({
                'class_index': class_index,
                'samples': class_total,
                'accuracy': round(float(class_correct / class_total), 4) if class_total > 0 else 0.0
            })

        results = {
            'test_samples': len(all_true_labels),
            'num_classes': num_classes,
            'metrics': {
                'accuracy': round(float(accuracy), 4),
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
                'f1_score': round(float(f1), 4)
            },
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': per_class_accuracy,
            'artifacts': {
                'model_path': model_path,
                'labels_path': pseudo_labels_path
            }
        }

        with evaluation_lock:
            evaluation_state['results'] = results
            evaluation_state['progress'] = 100
            evaluation_state['current_step'] = 'Complete'

        log_evaluation("Model evaluation completed successfully!")

    except Exception as e:
        with evaluation_lock:
            evaluation_state['error'] = str(e)
            evaluation_state['current_step'] = 'Error'
        log_evaluation(f"Error in evaluation pipeline: {str(e)}")
    finally:
        with evaluation_lock:
            evaluation_state['is_running'] = False

@app.route('/api/evaluation/start', methods=['POST'])
def start_evaluation():
    """Start model evaluation"""
    with evaluation_lock:
        if evaluation_state['is_running']:
            return jsonify({'error': 'Evaluation is already running'}), 400

    thread = threading.Thread(target=run_evaluation_pipeline)
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Evaluation started', 'status': 'running'})

@app.route('/api/evaluation/status', methods=['GET'])
def get_evaluation_status():
    """Get current evaluation status"""
    with evaluation_lock:
        return jsonify({
            'status': 'completed' if evaluation_state['progress'] == 100 and not evaluation_state['is_running'] else ('error' if evaluation_state['error'] else 'running'),
            'current_step': evaluation_state['current_step'],
            'progress': evaluation_state['progress'],
            'logs': evaluation_state['logs'][-50:],
            'error': evaluation_state['error']
        })

@app.route('/api/evaluation/results', methods=['GET'])
def get_evaluation_results():
    """Get model evaluation results"""
    try:
        with evaluation_lock:
            if not evaluation_state['results']:
                return jsonify({'error': 'No results available. Please run evaluation first.'}), 404

            return jsonify(evaluation_state['results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_gradcam_pipeline(batch_index=0, image_index=0):
    """Generate Grad-CAM visualizations for one paired MRI/PET sample"""
    try:
        with gradcam_lock:
            gradcam_state['is_running'] = True
            gradcam_state['current_step'] = 'Initializing'
            gradcam_state['progress'] = 0
            gradcam_state['logs'] = []
            gradcam_state['results'] = {}
            gradcam_state['error'] = None

        log_gradcam("Starting Grad-CAM pipeline...")

        features_path = os.path.join(PET_MRI_PATH, 'extracted_features')
        artifacts_path = os.path.join(PET_MRI_PATH, 'classification_artifacts')
        batched_path = os.path.join(PET_MRI_PATH, 'batched_images')

        extractor_model_path = os.path.join(features_path, 'feature_extractor_model.pth')
        combined_features_path = os.path.join(features_path, 'combined_features.npy')
        pseudo_labels_path = os.path.join(artifacts_path, 'pseudo_labels.npy')
        classifier_model_path = os.path.join(artifacts_path, 'best_classifier_model.pth')

        with gradcam_lock:
            gradcam_state['current_step'] = 'Loading Artifacts'
            gradcam_state['progress'] = 15

        if not os.path.exists(extractor_model_path):
            raise Exception("Feature extractor weights not found. Please rerun feature extraction first.")
        if not os.path.exists(classifier_model_path) or not os.path.exists(pseudo_labels_path):
            raise Exception("Classifier artifacts not found. Please run classification training first.")
        if not os.path.exists(combined_features_path):
            raise Exception("Combined features not found. Please run feature combination first.")

        mri_batch_files = sorted([f for f in os.listdir(batched_path) if f.startswith("batch_mri_") and f.endswith(".npy")])
        pet_batch_files = sorted([f for f in os.listdir(batched_path) if f.startswith("batch_pet_") and f.endswith(".npy")])

        if batch_index < 0 or batch_index >= len(mri_batch_files) or batch_index >= len(pet_batch_files):
            raise Exception(f"Batch index {batch_index} is out of range.")

        mri_batch = np.load(os.path.join(batched_path, mri_batch_files[batch_index]))
        pet_batch = np.load(os.path.join(batched_path, pet_batch_files[batch_index]))

        if image_index < 0 or image_index >= len(mri_batch) or image_index >= len(pet_batch):
            raise Exception(f"Image index {image_index} is out of range for batch {batch_index}.")

        sample_mri = mri_batch[image_index]
        sample_pet = pet_batch[image_index]
        pseudo_labels = np.load(pseudo_labels_path)
        combined_features = np.load(combined_features_path)
        num_classes = len(np.unique(pseudo_labels))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with gradcam_lock:
            gradcam_state['current_step'] = 'Building Models'
            gradcam_state['progress'] = 35

        mri_extractor = CNNTransformerFeatureExtractor(
            num_channels=3,
            cnn_out_channels=64,
            num_heads=8,
            num_layers=2
        ).to(device)
        pet_extractor = CNNTransformerFeatureExtractor(
            num_channels=3,
            cnn_out_channels=64,
            num_heads=8,
            num_layers=2
        ).to(device)

        extractor_state = torch.load(extractor_model_path, map_location=device)
        mri_extractor.load_state_dict(extractor_state)
        pet_extractor.load_state_dict(extractor_state)

        classifier_head = ClassificationHead(
            input_dim=combined_features.shape[1],
            num_classes=num_classes
        ).to(device)
        classifier_head.load_state_dict(torch.load(classifier_model_path, map_location=device))

        multimodal_model = MultimodalClassificationModel(mri_extractor, pet_extractor, classifier_head).to(device)
        multimodal_model.eval()

        mri_tensor = torch.from_numpy(sample_mri).permute(2, 0, 1).float().unsqueeze(0).to(device)
        pet_tensor = torch.from_numpy(sample_pet).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with gradcam_lock:
            gradcam_state['current_step'] = 'Running Prediction'
            gradcam_state['progress'] = 55

        with torch.enable_grad():
            mri_cam = GradCAM(multimodal_model, multimodal_model.mri_extractor.custom_cnn[7])
            outputs = multimodal_model(mri_tensor, pet_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = int(torch.argmax(outputs, dim=1).item())
            predicted_confidence = float(probabilities[0, predicted_class].item())
            mri_heatmap = mri_cam.generate(outputs[0, predicted_class], output_size=(sample_mri.shape[0], sample_mri.shape[1]))
            mri_cam.close()

            pet_cam = GradCAM(multimodal_model, multimodal_model.pet_extractor.custom_cnn[7])
            outputs = multimodal_model(mri_tensor, pet_tensor)
            pet_heatmap = pet_cam.generate(outputs[0, predicted_class], output_size=(sample_pet.shape[0], sample_pet.shape[1]))
            pet_cam.close()

        with gradcam_lock:
            gradcam_state['current_step'] = 'Preparing Visualizations'
            gradcam_state['progress'] = 80

        original_mri, overlay_mri = create_heatmap_overlay(sample_mri, mri_heatmap)
        original_pet, overlay_pet = create_heatmap_overlay(sample_pet, pet_heatmap)

        results = {
            'batch_index': int(batch_index),
            'image_index': int(image_index),
            'predicted_class': predicted_class,
            'predicted_label': f"Cluster_{predicted_class}",
            'predicted_confidence': round(predicted_confidence, 4),
            'mri_original_image': encode_image_to_base64(original_mri),
            'mri_gradcam_image': encode_image_to_base64(overlay_mri),
            'pet_original_image': encode_image_to_base64(original_pet),
            'pet_gradcam_image': encode_image_to_base64(overlay_pet)
        }

        with gradcam_lock:
            gradcam_state['results'] = results
            gradcam_state['progress'] = 100
            gradcam_state['current_step'] = 'Complete'

        log_gradcam("Grad-CAM generation completed successfully!")

    except Exception as e:
        with gradcam_lock:
            gradcam_state['error'] = str(e)
            gradcam_state['current_step'] = 'Error'
        log_gradcam(f"Error in Grad-CAM pipeline: {str(e)}")
    finally:
        with gradcam_lock:
            gradcam_state['is_running'] = False

@app.route('/api/gradcam/start', methods=['POST'])
def start_gradcam():
    """Start Grad-CAM generation"""
    with gradcam_lock:
        if gradcam_state['is_running']:
            return jsonify({'error': 'Grad-CAM is already running'}), 400

    payload = request.get_json(silent=True) or {}
    batch_index = int(payload.get('batch_index', 0))
    image_index = int(payload.get('image_index', 0))

    thread = threading.Thread(target=run_gradcam_pipeline, args=(batch_index, image_index))
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Grad-CAM started', 'status': 'running'})

@app.route('/api/gradcam/status', methods=['GET'])
def get_gradcam_status():
    """Get current Grad-CAM status"""
    with gradcam_lock:
        return jsonify({
            'status': 'completed' if gradcam_state['progress'] == 100 and not gradcam_state['is_running'] else ('error' if gradcam_state['error'] else 'running'),
            'current_step': gradcam_state['current_step'],
            'progress': gradcam_state['progress'],
            'logs': gradcam_state['logs'][-50:],
            'error': gradcam_state['error']
        })

@app.route('/api/gradcam/results', methods=['GET'])
def get_gradcam_results():
    """Get Grad-CAM results"""
    try:
        with gradcam_lock:
            if not gradcam_state['results']:
                return jsonify({'error': 'No results available. Please run Grad-CAM first.'}), 404

            return jsonify(gradcam_state['results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_model_export_pipeline():
    """Export inference-ready model weights and metadata for another backend"""
    try:
        with model_export_lock:
            model_export_state['is_running'] = True
            model_export_state['current_step'] = 'Initializing'
            model_export_state['progress'] = 0
            model_export_state['logs'] = []
            model_export_state['results'] = {}
            model_export_state['error'] = None

        log_model_export("Starting model export pipeline...")

        features_path = os.path.join(PET_MRI_PATH, 'extracted_features')
        artifacts_path = os.path.join(PET_MRI_PATH, 'classification_artifacts')

        feature_extractor_path = os.path.join(features_path, 'feature_extractor_model.pth')
        classifier_model_path = os.path.join(artifacts_path, 'best_classifier_model.pth')
        pseudo_labels_path = os.path.join(artifacts_path, 'pseudo_labels.npy')
        combined_features_path = os.path.join(features_path, 'combined_features.npy')

        with model_export_lock:
            model_export_state['current_step'] = 'Validating Artifacts'
            model_export_state['progress'] = 15

        required_paths = [
            feature_extractor_path,
            classifier_model_path,
            pseudo_labels_path,
            combined_features_path
        ]
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            raise Exception(
                "Missing required artifacts for export: " + ", ".join(missing)
            )

        pseudo_labels = np.load(pseudo_labels_path)
        combined_features = np.load(combined_features_path)
        unique_labels = sorted(int(label) for label in np.unique(pseudo_labels))

        with model_export_lock:
            model_export_state['current_step'] = 'Preparing Export Folder'
            model_export_state['progress'] = 35

        export_root = os.path.join(PET_MRI_PATH, 'exported_models')
        os.makedirs(export_root, exist_ok=True)
        export_name = f"ad_detection_export_{time.strftime('%Y%m%d_%H%M%S')}"
        export_dir = os.path.join(export_root, export_name)
        os.makedirs(export_dir, exist_ok=True)

        log_model_export(f"Created export folder: {export_dir}")

        with model_export_lock:
            model_export_state['current_step'] = 'Copying Model Files'
            model_export_state['progress'] = 60

        exported_feature_extractor = os.path.join(export_dir, 'feature_extractor_model.pth')
        exported_classifier = os.path.join(export_dir, 'classification_head.pth')
        exported_labels = os.path.join(export_dir, 'pseudo_labels.npy')

        shutil.copy2(feature_extractor_path, exported_feature_extractor)
        shutil.copy2(classifier_model_path, exported_classifier)
        shutil.copy2(pseudo_labels_path, exported_labels)

        metadata = {
            'package_name': export_name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'inference_type': 'paired_mri_pet_classification',
            'paths': {
                'feature_extractor_weights': 'feature_extractor_model.pth',
                'classification_head_weights': 'classification_head.pth',
                'pseudo_labels': 'pseudo_labels.npy'
            },
            'feature_extractor_config': {
                'architecture': 'CNNTransformerFeatureExtractor',
                'num_channels': 3,
                'cnn_out_channels': 64,
                'num_heads': 8,
                'num_layers': 2,
                'pretrained_backbone': 'efficientnet_b0'
            },
            'classification_head_config': {
                'architecture': 'ClassificationHead',
                'input_dim': int(combined_features.shape[1]),
                'num_classes': len(unique_labels),
                'hidden_dims': [256, 128]
            },
            'class_labels': [f'Cluster_{label}' for label in unique_labels],
            'notes': [
                'Use two copies of the feature extractor: one for MRI and one for PET.',
                'Extract MRI and PET feature vectors separately, concatenate them, then pass the result to the classification head.',
                'Input images should follow the same preprocessing and batching conventions used by this application.'
            ]
        }

        metadata_path = os.path.join(export_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        readme_path = os.path.join(export_dir, 'README.txt')
        with open(readme_path, 'w', encoding='utf-8') as readme_file:
            readme_file.write(
                "AD Detection Export Package\n\n"
                "Contents:\n"
                "- feature_extractor_model.pth: shared weights for MRI and PET feature extraction\n"
                "- classification_head.pth: trained classifier on concatenated MRI+PET features\n"
                "- pseudo_labels.npy: cluster ids used during training\n"
                "- metadata.json: architecture and inference configuration\n\n"
                "Inference flow:\n"
                "1. Load MRI image and PET image.\n"
                "2. Apply the same preprocessing used in this project.\n"
                "3. Run each image through a CNNTransformerFeatureExtractor instance.\n"
                "4. Concatenate both feature vectors.\n"
                "5. Run the concatenated vector through the classification head.\n"
            )

        with model_export_lock:
            model_export_state['current_step'] = 'Creating Zip Archive'
            model_export_state['progress'] = 85

        archive_base = os.path.join(export_root, export_name)
        archive_path = shutil.make_archive(archive_base, 'zip', export_dir)

        results = {
            'export_name': export_name,
            'export_directory': export_dir,
            'archive_path': archive_path,
            'exported_files': [
                exported_feature_extractor,
                exported_classifier,
                exported_labels,
                metadata_path,
                readme_path
            ],
            'num_classes': len(unique_labels),
            'class_labels': metadata['class_labels']
        }

        with model_export_lock:
            model_export_state['results'] = results
            model_export_state['progress'] = 100
            model_export_state['current_step'] = 'Complete'

        log_model_export("Model export completed successfully!")

    except Exception as e:
        with model_export_lock:
            model_export_state['error'] = str(e)
            model_export_state['current_step'] = 'Error'
        log_model_export(f"Error in model export pipeline: {str(e)}")
    finally:
        with model_export_lock:
            model_export_state['is_running'] = False

@app.route('/api/model-export/start', methods=['POST'])
def start_model_export():
    """Start model export"""
    with model_export_lock:
        if model_export_state['is_running']:
            return jsonify({'error': 'Model export is already running'}), 400

    thread = threading.Thread(target=run_model_export_pipeline)
    thread.daemon = True
    thread.start()

    return jsonify({'message': 'Model export started', 'status': 'running'})

@app.route('/api/model-export/status', methods=['GET'])
def get_model_export_status():
    """Get current model export status"""
    with model_export_lock:
        return jsonify({
            'status': 'completed' if model_export_state['progress'] == 100 and not model_export_state['is_running'] else ('error' if model_export_state['error'] else 'running'),
            'current_step': model_export_state['current_step'],
            'progress': model_export_state['progress'],
            'logs': model_export_state['logs'][-50:],
            'error': model_export_state['error']
        })

@app.route('/api/model-export/results', methods=['GET'])
def get_model_export_results():
    """Get model export results"""
    try:
        with model_export_lock:
            if not model_export_state['results']:
                return jsonify({'error': 'No export results available. Please run model export first.'}), 404

            return jsonify(model_export_state['results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== PREPROCESSING MODULE ====================

# Global preprocessing state
preprocessing_state = {
    'is_running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'results': {},
    'error': None
}
preprocessing_lock = threading.Lock()

def log_preprocessing(message):
    """Add a log message to preprocessing state"""
    with preprocessing_lock:
        preprocessing_state['logs'].append({
            'message': message,
            'timestamp': time.time()
        })
        print(f"[Preprocessing] {message}")

def resize_images(input_path, output_path, target_size=(256, 256)):
    """Resize images in a folder"""
    stats = {'processed': 0, 'errors': 0, 'processing_times': []}
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png'))]
    
    for filename in files:
        try:
            start_time = time.time()
            input_image_path = os.path.join(input_path, filename)
            output_image_path = os.path.join(output_path, filename)
            
            img = Image.open(input_image_path)
            resized_img = img.resize(target_size)
            resized_img.save(output_image_path)
            img.close()
            
            stats['processed'] += 1
            stats['processing_times'].append(time.time() - start_time)
        except Exception as e:
            stats['errors'] += 1
            log_preprocessing(f"Error resizing {filename}: {str(e)}")
    
    return stats

def normalize_images(input_path, output_path):
    """Normalize images to [0, 1] range"""
    stats = {'processed': 0, 'errors': 0}
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png'))]
    
    for filename in files:
        try:
            input_image_path = os.path.join(input_path, filename)
            output_image_path = os.path.join(output_path, filename)
            
            img = Image.open(input_image_path)
            img_array = np.array(img)
            normalized_img_array = img_array.astype(np.float32) / 255.0
            normalized_img = Image.fromarray((normalized_img_array * 255).astype(np.uint8))
            normalized_img.save(output_image_path)
            img.close()
            
            stats['processed'] += 1
        except Exception as e:
            stats['errors'] += 1
            log_preprocessing(f"Error normalizing {filename}: {str(e)}")
    
    return stats

def augment_images(mri_input_path, pet_input_path, mri_output_path, pet_output_path, num_augmentations=9):
    """Apply consistent augmentation to paired images"""
    stats = {'mri_success': 0, 'pet_success': 0, 'missing_pet': 0}
    
    if not os.path.exists(mri_output_path):
        os.makedirs(mri_output_path)
    if not os.path.exists(pet_output_path):
        os.makedirs(pet_output_path)
    
    augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=(-10, 10, -10, 10)),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.6, 1.0), ratio=(0.75, 1.25)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
    ])
    
    mri_files = [f for f in os.listdir(mri_input_path) if f.endswith(('.jpg', '.png'))]
    
    for filename in mri_files:
        mri_path = os.path.join(mri_input_path, filename)
        pet_path = os.path.join(pet_input_path, filename)
        
        if not os.path.exists(pet_path):
            stats['missing_pet'] += 1
            continue
        
        try:
            mri_img = Image.open(mri_path)
            pet_img = Image.open(pet_path)
            
            for i in range(num_augmentations):
                seed = random.randint(0, 100000)
                random.seed(seed)
                torch.manual_seed(seed)
                
                augmented_mri = augmentations(mri_img)
                
                random.seed(seed)
                torch.manual_seed(seed)
                
                augmented_pet = augmentations(pet_img)
                
                base_filename = os.path.splitext(filename)[0]
                mri_output = os.path.join(mri_output_path, f"{base_filename}_aug_{i+1}.png")
                pet_output = os.path.join(pet_output_path, f"{base_filename}_aug_{i+1}.png")
                
                augmented_mri.save(mri_output)
                augmented_pet.save(pet_output)
            
            mri_img.close()
            pet_img.close()
            
            stats['mri_success'] += 1
            stats['pet_success'] += 1
        except Exception as e:
            log_preprocessing(f"Error augmenting {filename}: {str(e)}")
    
    return stats

def batch_images(mri_input_path, pet_input_path, output_path, batch_size=6):
    """Batch images into numpy arrays"""
    stats = {'mri_batches': 0, 'pet_batches': 0, 'total_images': 0}
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    mri_filenames = [f for f in os.listdir(mri_input_path) if f.endswith(('.jpg', '.png'))]
    
    batched_mri_images = []
    batched_pet_images = []
    
    for i in range(0, len(mri_filenames), batch_size):
        mri_batch = []
        pet_batch = []
        current_batch_filenames = mri_filenames[i:i + batch_size]
        
        for filename in current_batch_filenames:
            mri_path = os.path.join(mri_input_path, filename)
            pet_path = os.path.join(pet_input_path, filename)
            
            try:
                mri_img = Image.open(mri_path).convert("RGB")
                pet_img = Image.open(pet_path).convert("RGB")
                
                mri_batch.append(np.array(mri_img))
                pet_batch.append(np.array(pet_img))
                
                mri_img.close()
                pet_img.close()
            except Exception as e:
                log_preprocessing(f"Error batching {filename}: {str(e)}")
        
        if mri_batch and pet_batch:
            batched_mri_images.append(np.array(mri_batch))
            batched_pet_images.append(np.array(pet_batch))
    
    # Save batches
    for i, mri_batch in enumerate(batched_mri_images):
        mri_output_file = os.path.join(output_path, f"batch_mri_{i}.npy")
        np.save(mri_output_file, mri_batch)
        stats['mri_batches'] += 1
    
    for i, pet_batch in enumerate(batched_pet_images):
        pet_output_file = os.path.join(output_path, f"batch_pet_{i}.npy")
        np.save(pet_output_file, pet_batch)
        stats['pet_batches'] += 1
    
    stats['total_images'] = len(mri_filenames)
    
    return stats

def run_preprocessing_pipeline():
    """Execute the complete preprocessing pipeline"""
    try:
        with preprocessing_lock:
            preprocessing_state['is_running'] = True
            preprocessing_state['progress'] = 0
            preprocessing_state['logs'] = []
            preprocessing_state['results'] = {}
            preprocessing_state['error'] = None
        
        # Define paths
        mri_input = os.path.join(PET_MRI_PATH, 'MRI')
        pet_input = os.path.join(PET_MRI_PATH, 'PET')
        
        mri_resized = os.path.join(PET_MRI_PATH, 'resized_MRI')
        pet_resized = os.path.join(PET_MRI_PATH, 'resized_PET')
        
        mri_normalized = os.path.join(PET_MRI_PATH, 'normalized_MRI')
        pet_normalized = os.path.join(PET_MRI_PATH, 'normalized_PET')
        
        mri_augmented = os.path.join(PET_MRI_PATH, 'augment_MRI')
        pet_augmented = os.path.join(PET_MRI_PATH, 'augment_PET')
        
        batched_output = os.path.join(PET_MRI_PATH, 'batched_images')
        
        # Step 1: Resize (20% progress)
        with preprocessing_lock:
            preprocessing_state['current_step'] = 'Resizing Images'
            preprocessing_state['progress'] = 5
        log_preprocessing("Starting image resizing...")
        
        mri_resize_stats = resize_images(mri_input, mri_resized)
        pet_resize_stats = resize_images(pet_input, pet_resized)
        
        with preprocessing_lock:
            preprocessing_state['results']['resize'] = {
                'mri': mri_resize_stats,
                'pet': pet_resize_stats
            }
            preprocessing_state['progress'] = 20
        
        log_preprocessing(f"Resized {mri_resize_stats['processed']} MRI and {pet_resize_stats['processed']} PET images")
        
        # Step 2: Normalize (40% progress)
        with preprocessing_lock:
            preprocessing_state['current_step'] = 'Normalizing Images'
            preprocessing_state['progress'] = 25
        log_preprocessing("Starting image normalization...")
        
        mri_norm_stats = normalize_images(mri_resized, mri_normalized)
        pet_norm_stats = normalize_images(pet_resized, pet_normalized)
        
        with preprocessing_lock:
            preprocessing_state['results']['normalize'] = {
                'mri': mri_norm_stats,
                'pet': pet_norm_stats
            }
            preprocessing_state['progress'] = 40
        
        log_preprocessing(f"Normalized {mri_norm_stats['processed']} MRI and {pet_norm_stats['processed']} PET images")
        
        # Step 3: Augment (70% progress)
        with preprocessing_lock:
            preprocessing_state['current_step'] = 'Augmenting Images'
            preprocessing_state['progress'] = 45
        log_preprocessing("Starting image augmentation...")
        
        aug_stats = augment_images(mri_normalized, pet_normalized, mri_augmented, pet_augmented, num_augmentations=9)
        
        with preprocessing_lock:
            preprocessing_state['results']['augment'] = aug_stats
            preprocessing_state['progress'] = 70
        
        log_preprocessing(f"Augmented {aug_stats['mri_success']} image pairs")
        
        # Step 4: Batch (100% progress)
        with preprocessing_lock:
            preprocessing_state['current_step'] = 'Batching Images'
            preprocessing_state['progress'] = 75
        log_preprocessing("Starting image batching...")
        
        batch_stats = batch_images(mri_augmented, pet_augmented, batched_output, batch_size=6)
        
        with preprocessing_lock:
            preprocessing_state['results']['batch'] = batch_stats
            preprocessing_state['progress'] = 100
            preprocessing_state['current_step'] = 'Complete'
        
        log_preprocessing(f"Created {batch_stats['mri_batches']} MRI batches and {batch_stats['pet_batches']} PET batches")
        log_preprocessing("Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        with preprocessing_lock:
            preprocessing_state['error'] = str(e)
            preprocessing_state['current_step'] = 'Error'
        log_preprocessing(f"Error in preprocessing pipeline: {str(e)}")
    finally:
        with preprocessing_lock:
            preprocessing_state['is_running'] = False

@app.route('/api/preprocessing/start', methods=['POST'])
def start_preprocessing():
    """Start the preprocessing pipeline"""
    with preprocessing_lock:
        if preprocessing_state['is_running']:
            return jsonify({'error': 'Preprocessing already running'}), 400
    
    # Start preprocessing in a separate thread
    thread = threading.Thread(target=run_preprocessing_pipeline)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Preprocessing started', 'status': 'running'})

@app.route('/api/preprocessing/status', methods=['GET'])
def get_preprocessing_status():
    """Get current preprocessing status"""
    with preprocessing_lock:
        return jsonify({
            'is_running': preprocessing_state['is_running'],
            'current_step': preprocessing_state['current_step'],
            'progress': preprocessing_state['progress'],
            'logs': preprocessing_state['logs'][-50:],  # Last 50 logs
            'results': preprocessing_state['results'],
            'error': preprocessing_state['error']
        })

@app.route('/api/preprocessing/results', methods=['GET'])
def get_preprocessing_results():
    """Get final preprocessing results with analytics"""
    try:
        # Analyze final results
        batched_path = os.path.join(PET_MRI_PATH, 'batched_images')
        augmented_mri = os.path.join(PET_MRI_PATH, 'augment_MRI')
        augmented_pet = os.path.join(PET_MRI_PATH, 'augment_PET')
        
        results = {
            'summary': preprocessing_state['results'],
            'final_analysis': {
                'augmented_mri': analyze_image_folder(augmented_mri),
                'augmented_pet': analyze_image_folder(augmented_pet)
            },
            'logs': preprocessing_state['logs']
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        print(f"Watching directory: {PET_MRI_PATH}")
        print("Backend server starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        observer.stop()
        observer.join()
