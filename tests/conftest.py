"""Shared fixtures for tests"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_dummy_audio(duration_seconds=5, sample_rate=16000):
    """Create dummy audio tensor"""
    num_samples = int(duration_seconds * sample_rate)
    return torch.randn(num_samples)


def create_dummy_audio_batch(batch_size=2, duration_seconds=5, sample_rate=16000):
    """Create batch of dummy audio tensors"""
    num_samples = int(duration_seconds * sample_rate)
    return [torch.randn(num_samples) for _ in range(batch_size)]


def create_dummy_speech_representations(batch_size=2, seq_len=100, hidden_size=1024):
    """Create dummy speech representations (HuBERT output format)"""
    return torch.randn(batch_size, seq_len, hidden_size)


def create_dummy_text_embeddings(batch_size=2, embedding_dim=4096):
    """Create dummy text embeddings"""
    embeddings = torch.randn(batch_size, embedding_dim)
    # Normalize to simulate real embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def create_dummy_audio_embeddings(batch_size=2, embedding_dim=4096):
    """Create dummy audio embeddings"""
    embeddings = torch.randn(batch_size, embedding_dim)
    # Normalize to simulate real embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def create_mock_dataset_sample():
    """Create a mock dataset sample"""
    return {
        "audio": create_dummy_audio(duration_seconds=5),
        "text": "This is a test passage about machine learning and speech processing.",
        "sample_rate": 16000
    }


def create_mock_dataset_batch(batch_size=2):
    """Create a mock dataset batch"""
    return {
        "audio": torch.stack([create_dummy_audio() for _ in range(batch_size)]),
        "text": [f"Test text {i}" for i in range(batch_size)],
        "sample_rate": torch.tensor([16000] * batch_size)
    }


def create_temp_audio_file(duration_seconds=5, sample_rate=16000):
    """Create a temporary audio file for testing"""
    import torchaudio
    
    # Create dummy audio
    num_samples = int(duration_seconds * sample_rate)
    waveform = torch.randn(1, num_samples)  # (channels, samples)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save audio
    torchaudio.save(temp_path, waveform, sample_rate)
    
    return temp_path


def cleanup_temp_file(file_path):
    """Clean up temporary file"""
    if os.path.exists(file_path):
        os.remove(file_path)



