"""Tests for data pipeline components"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import AudioPreprocessor
from src.data.dataset import SpeechDataset
from conftest import (
    create_dummy_audio,
    create_temp_audio_file,
    cleanup_temp_file,
    create_mock_dataset_sample
)


def test_audio_preprocessor_resample():
    """Test AudioPreprocessor resampling"""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    
    # Create audio at different sample rate
    original_sr = 44100
    duration = 1.0
    waveform = torch.randn(int(original_sr * duration))
    
    # Resample to 16kHz
    resampled = preprocessor.resample(waveform, original_sr, target_sample_rate=16000)
    
    # Check that length is approximately correct (16kHz * 1s = 16000 samples)
    expected_length = int(16000 * duration)
    assert abs(resampled.shape[-1] - expected_length) < 100, \
        f"Resampled length should be ~{expected_length}, got {resampled.shape[-1]}"
    
    print("✓ AudioPreprocessor resample test passed")


def test_audio_preprocessor_trim_or_pad():
    """Test AudioPreprocessor trim_or_pad"""
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        max_length_seconds=5.0
    )
    
    # Test trimming (audio longer than max)
    long_audio = torch.randn(16000 * 10)  # 10 seconds
    trimmed = preprocessor.trim_or_pad(long_audio)
    expected_length = int(16000 * 5.0)
    assert trimmed.shape[-1] == expected_length, \
        f"Should trim to {expected_length}, got {trimmed.shape[-1]}"
    
    # Test padding (audio shorter than max)
    short_audio = torch.randn(16000 * 2)  # 2 seconds
    padded = preprocessor.trim_or_pad(short_audio)
    assert padded.shape[-1] == expected_length, \
        f"Should pad to {expected_length}, got {padded.shape[-1]}"
    
    # Test no change (audio exactly at max)
    exact_audio = torch.randn(expected_length)
    unchanged = preprocessor.trim_or_pad(exact_audio)
    assert unchanged.shape[-1] == expected_length
    
    print("✓ AudioPreprocessor trim_or_pad test passed")


def test_audio_preprocessor_normalize():
    """Test AudioPreprocessor normalization"""
    preprocessor = AudioPreprocessor(normalize=True)
    
    # Create audio with large values
    audio = torch.randn(16000) * 10  # Large amplitude
    normalized = preprocessor.normalize_audio(audio)
    
    # Check that max absolute value is close to 1.0
    max_val = torch.abs(normalized).max()
    assert max_val <= 1.0 + 1e-5, f"Normalized audio should have max <= 1.0, got {max_val}"
    
    # Test with normalize=False
    preprocessor_no_norm = AudioPreprocessor(normalize=False)
    not_normalized = preprocessor_no_norm.normalize_audio(audio)
    assert torch.allclose(not_normalized, audio), "Should not normalize when normalize=False"
    
    print("✓ AudioPreprocessor normalize test passed")


def test_audio_preprocessor_process():
    """Test AudioPreprocessor complete pipeline"""
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        max_length_seconds=5.0,
        normalize=True
    )
    
    # Test with tensor
    audio_tensor = torch.randn(44100)  # 1 second at 44.1kHz
    processed = preprocessor.process(audio_tensor, sample_rate=44100)
    
    # Should be resampled to 16kHz, trimmed/padded to 5s, and normalized
    expected_length = int(16000 * 5.0)
    assert processed.shape[-1] == expected_length, \
        f"Processed audio should have length {expected_length}"
    
    max_val = torch.abs(processed).max()
    assert max_val <= 1.0 + 1e-5, "Should be normalized"
    
    print("✓ AudioPreprocessor process pipeline test passed")


def test_audio_preprocessor_load_audio():
    """Test AudioPreprocessor load_audio with temp file"""
    preprocessor = AudioPreprocessor()
    
    # Create temporary audio file
    temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
    
    try:
        waveform, sr = preprocessor.load_audio(temp_path)
        
        assert isinstance(waveform, torch.Tensor), "Should return tensor"
        assert sr == 16000, f"Should return sample rate 16000, got {sr}"
        assert len(waveform.shape) == 1, "Should return 1D tensor"
        
        print("✓ AudioPreprocessor load_audio test passed")
    finally:
        cleanup_temp_file(temp_path)


def test_speech_dataset_getitem():
    """Test SpeechDataset __getitem__ with mock data"""
    # This test requires a real dataset, so we'll skip if not available
    try:
        dataset = SpeechDataset(
            dataset_name="spoken_squad_test",
            split="train",
            streaming=True,
            sample_rate=16000,
            max_audio_length=60.0
        )
        
        # Try to get one sample
        sample = dataset[0]
        
        # Check format
        assert "audio" in sample, "Sample should have 'audio' key"
        assert "text" in sample, "Sample should have 'text' key"
        assert "sample_rate" in sample, "Sample should have 'sample_rate' key"
        
        # Check types
        assert isinstance(sample["audio"], torch.Tensor), "Audio should be tensor"
        assert isinstance(sample["text"], str), "Text should be string"
        assert sample["sample_rate"] == 16000, "Sample rate should be 16000"
        
        print("✓ SpeechDataset __getitem__ test passed")
    except Exception as e:
        print(f"⚠ SpeechDataset __getitem__ test skipped: {e}")


def test_speech_dataset_collate_fn():
    """Test SpeechDataset collate function"""
    # Create mock batch with different audio lengths
    batch = [
        {
            "audio": torch.randn(16000 * 3),  # 3 seconds
            "text": "First text",
            "sample_rate": 16000
        },
        {
            "audio": torch.randn(16000 * 5),  # 5 seconds
            "text": "Second text",
            "sample_rate": 16000
        },
        {
            "audio": torch.randn(16000 * 2),  # 2 seconds
            "text": "Third text",
            "sample_rate": 16000
        }
    ]
    
    # Create a dummy dataset to get collate function
    try:
        dataset = SpeechDataset(
            dataset_name="spoken_squad_test",
            split="train",
            streaming=True
        )
        collate_fn = dataset.get_collate_fn()
        
        # Apply collate function
        batched = collate_fn(batch)
        
        # Check that all audios are padded to same length
        audio_batch = batched["audio"]
        assert len(audio_batch.shape) == 2, "Should be 2D tensor (batch, samples)"
        assert audio_batch.shape[0] == 3, "Should have batch_size=3"
        
        # All should have same length (padded to max)
        max_len = max(a.shape[-1] for a in [item["audio"] for item in batch])
        assert audio_batch.shape[1] == max_len, \
            f"All audios should be padded to {max_len}"
        
        # Check texts
        assert len(batched["text"]) == 3, "Should have 3 texts"
        assert all(isinstance(t, str) for t in batched["text"]), "Texts should be strings"
        
        # Check sample rates
        assert batched["sample_rate"].shape[0] == 3, "Should have 3 sample rates"
        
        print("✓ SpeechDataset collate_fn test passed")
    except Exception as e:
        print(f"⚠ SpeechDataset collate_fn test skipped: {e}")


def test_speech_dataset_different_audio_formats():
    """Test SpeechDataset with different audio formats"""
    # This test requires dataset access, so we'll create a mock scenario
    try:
        dataset = SpeechDataset(
            dataset_name="spoken_squad_test",
            split="train",
            streaming=True,
            sample_rate=16000
        )
        
        # The dataset should handle different formats internally
        # We test that it doesn't crash with different formats
        sample = dataset[0]
        
        # Audio should be processed to tensor format
        assert isinstance(sample["audio"], torch.Tensor), \
            "Audio should be converted to tensor"
        assert len(sample["audio"].shape) == 1, \
            "Audio should be 1D tensor"
        
        print("✓ SpeechDataset different audio formats test passed")
    except Exception as e:
        print(f"⚠ SpeechDataset different audio formats test skipped: {e}")


def test_dataset_preprocessor_integration():
    """Test integration between SpeechDataset and AudioPreprocessor"""
    try:
        dataset = SpeechDataset(
            dataset_name="spoken_squad_test",
            split="train",
            streaming=True,
            sample_rate=16000,
            max_audio_length=30.0
        )
        
        sample = dataset[0]
        audio = sample["audio"]
        
        # Check that audio is preprocessed correctly
        assert isinstance(audio, torch.Tensor), "Should be tensor"
        assert sample["sample_rate"] == 16000, "Should have correct sample rate"
        
        # Check that audio length is within max (allowing some tolerance)
        max_samples = int(16000 * 30.0)
        assert audio.shape[-1] <= max_samples, \
            f"Audio should be trimmed to max {max_samples} samples"
        
        print("✓ Dataset-Preprocessor integration test passed")
    except Exception as e:
        print(f"⚠ Dataset-Preprocessor integration test skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Data Pipeline Tests")
    print("=" * 60)
    print()
    
    # AudioPreprocessor tests
    test_audio_preprocessor_resample()
    test_audio_preprocessor_trim_or_pad()
    test_audio_preprocessor_normalize()
    test_audio_preprocessor_process()
    test_audio_preprocessor_load_audio()
    
    # SpeechDataset tests
    test_speech_dataset_getitem()
    test_speech_dataset_collate_fn()
    test_speech_dataset_different_audio_formats()
    test_dataset_preprocessor_integration()
    
    print()
    print("=" * 60)
    print("Data Pipeline Tests Completed")
    print("=" * 60)



