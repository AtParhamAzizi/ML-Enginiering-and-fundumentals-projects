"""Utilities for loading a trained regression checkpoint."""
from pathlib import Path
import joblib

def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return joblib.load(checkpoint_path)
