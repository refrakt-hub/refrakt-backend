"""Utility functions for classifying artifacts"""

from pathlib import Path


def classify_artifact(file_path: Path) -> str:
    """Classify artifact type based on file extension and name"""
    suffix = file_path.suffix.lower()
    name = file_path.name.lower()
    
    if suffix in ['.pth', '.pt', '.ckpt', '.h5', '.pb', '.onnx', '.safetensors']:
        return 'model'
    elif suffix in ['.log', '.txt'] or 'log' in name:
        return 'log'
    elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf']:
        return 'visualization'
    elif 'checkpoint' in name or 'ckpt' in name:
        return 'checkpoint'
    else:
        return 'other'

