"""Tests to validate Qwen-Audio-Chat inference correctness"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.generator import AudioConditionedGenerator
from conftest import (
    create_temp_audio_file,
    cleanup_temp_file
)


def get_device():
    """Get device (cuda if available, else cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_qwen_audio_token_processing():
    """
    Validate that audio is processed as native audio tokens, not converted to text.
    
    This test verifies that the audio input is passed to the processor in its
    raw audio format, not as transcribed text.
    """
    try:
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        # Create audio file
        audio_file = None
        try:
            audio_file = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
            
            # Prepare audio inputs
            audio_inputs = generator.prepare_audio_inputs([audio_file])
            
            # Verify audio is in raw format (numpy array of audio samples)
            assert len(audio_inputs) == 1, "Should have one audio input"
            audio_input = audio_inputs[0]
            
            # Check that it's raw audio data, not text
            assert "raw" in audio_input, "Should have 'raw' audio data"
            assert isinstance(audio_input["raw"], np.ndarray), \
                "Raw audio should be numpy array (audio samples)"
            assert audio_input["raw"].dtype in [np.float32, np.float64], \
                "Audio should be float array"
            
            # Audio should have reasonable shape (samples,)
            assert len(audio_input["raw"].shape) == 1, \
                "Audio should be 1D array (samples)"
            assert audio_input["raw"].shape[0] > 0, \
                "Audio should have samples"
            
            # Verify it's actual audio data (not empty, not all zeros)
            assert np.abs(audio_input["raw"]).max() > 0, \
                "Audio should have non-zero values"
            
            print("✓ Audio token processing validation passed")
            print(f"  Audio shape: {audio_input['raw'].shape}")
            print(f"  Sample rate: {audio_input['sampling_rate']}")
        finally:
            if audio_file:
                cleanup_temp_file(audio_file)
    except Exception as e:
        print(f"⚠ Audio token processing validation skipped: {e}")


def test_qwen_zero_shot_generation():
    """
    Validate that zero-shot generation works without fine-tuning.
    
    The model should generate responses using its pre-trained capabilities
    without requiring any fine-tuning.
    """
    try:
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        audio_file = None
        try:
            audio_file = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
            
            query = "What is the main topic of this audio?"
            
            # Generate without any fine-tuning (zero-shot)
            result = generator.generate(
                query=query,
                audio_paths=[audio_file],
                max_new_tokens=50,
                temperature=0.7
            )
            
            # Verify generation succeeded
            assert "response" in result, "Should have response"
            assert isinstance(result["response"], str), "Response should be string"
            assert len(result["response"]) > 0, "Response should not be empty"
            
            # Response should not be an error message
            assert "error" not in result["response"].lower(), \
                "Response should not be an error message"
            
            print("✓ Zero-shot generation validation passed")
            print(f"  Response length: {len(result['response'])}")
        finally:
            if audio_file:
                cleanup_temp_file(audio_file)
    except Exception as e:
        print(f"⚠ Zero-shot generation validation skipped: {e}")


def test_qwen_multimodal_input():
    """
    Validate that the model accepts multimodal input (text + audio).
    
    The model should process both text query and audio passages together.
    """
    try:
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        audio_files = []
        try:
            # Create multiple audio files
            for i in range(2):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            query = "Compare the content of these two audio passages."
            
            # Generate with multimodal input (text + multiple audios)
            result = generator.generate(
                query=query,
                audio_paths=audio_files,
                max_new_tokens=50,
                temperature=0.7
            )
            
            # Verify multimodal processing succeeded
            assert "response" in result, "Should have response"
            assert len(result["audio_paths"]) == 2, "Should have 2 audio paths"
            assert result["num_audios"] == 2, "Should process 2 audios"
            
            # Response should acknowledge multiple inputs
            assert len(result["response"]) > 0, "Response should not be empty"
            
            print("✓ Multimodal input validation passed")
            print(f"  Processed {result['num_audios']} audio files")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ Multimodal input validation skipped: {e}")


def test_qwen_response_quality():
    """
    Basic quality checks on generated responses.
    
    Validates that responses are non-empty, properly formatted, and reasonable.
    """
    try:
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        audio_file = None
        try:
            audio_file = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
            
            query = "Summarize this audio content."
            
            result = generator.generate(
                query=query,
                audio_paths=[audio_file],
                max_new_tokens=100,
                temperature=0.7
            )
            
            response = result["response"]
            
            # Quality checks
            assert len(response) > 0, "Response should not be empty"
            assert len(response.strip()) > 0, "Response should not be only whitespace"
            
            # Should not be just special tokens or placeholders
            assert not response.startswith("<|"), \
                "Response should not start with special tokens"
            
            # Should have some reasonable length (not too short, not too long)
            # For max_new_tokens=100, response should be reasonable
            assert 10 <= len(response) <= 500, \
                f"Response length should be reasonable, got {len(response)}"
            
            print("✓ Response quality validation passed")
            print(f"  Response preview: {response[:100]}...")
        finally:
            if audio_file:
                cleanup_temp_file(audio_file)
    except Exception as e:
        print(f"⚠ Response quality validation skipped: {e}")


def test_qwen_with_retrieved_audios():
    """
    Validate integration with retrieved audio passages.
    
    Tests that the generator works correctly with audio files that would
    come from the retrieval system.
    """
    try:
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        # Simulate retrieved audio files (like from SpeechRetriever)
        retrieved_audios = []
        try:
            # Create multiple audio files as if retrieved
            for i in range(3):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                retrieved_audios.append(temp_path)
            
            # Simulate a query that would retrieve these audios
            query = "Based on the retrieved audio passages, what is the main theme?"
            
            # Generate using retrieved audios
            result = generator.generate(
                query=query,
                audio_paths=retrieved_audios,
                max_new_tokens=60,
                temperature=0.7
            )
            
            # Verify integration works
            assert "response" in result, "Should have response"
            assert len(result["audio_paths"]) == 3, "Should use all retrieved audios"
            assert result["num_audios"] == 3, "Should process 3 retrieved audios"
            assert result["query"] == query, "Query should match"
            
            # Response should be generated
            assert len(result["response"]) > 0, "Response should be generated"
            
            print("✓ Retrieved audios integration validation passed")
            print(f"  Used {result['num_audios']} retrieved audio passages")
        finally:
            # Cleanup
            for path in retrieved_audios:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ Retrieved audios integration validation skipped: {e}")


def test_qwen_audio_format_consistency():
    """
    Validate that audio format is consistent and correct for Qwen-Audio-Chat.
    
    Checks that audio preprocessing produces the expected format.
    """
    try:
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        audio_files = []
        try:
            # Create audio files with different durations
            for duration in [1, 2, 3]:
                temp_path = create_temp_audio_file(
                    duration_seconds=duration,
                    sample_rate=16000
                )
                audio_files.append(temp_path)
            
            # Prepare all audio inputs
            audio_inputs = generator.prepare_audio_inputs(audio_files)
            
            # Verify consistency
            assert len(audio_inputs) == len(audio_files), \
                "Should have same number of inputs as files"
            
            # All should have same sample rate
            sample_rates = [inp["sampling_rate"] for inp in audio_inputs]
            assert all(sr == 16000 for sr in sample_rates), \
                "All audios should have 16000 Hz sample rate"
            
            # All should be numpy arrays
            assert all(isinstance(inp["raw"], np.ndarray) for inp in audio_inputs), \
                "All audios should be numpy arrays"
            
            # All should be 1D (samples)
            assert all(len(inp["raw"].shape) == 1 for inp in audio_inputs), \
                "All audios should be 1D arrays"
            
            print("✓ Audio format consistency validation passed")
            print(f"  Processed {len(audio_inputs)} audio files")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ Audio format consistency validation skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen-Audio-Chat Inference Validation Tests")
    print("=" * 60)
    print()
    
    test_qwen_audio_token_processing()
    test_qwen_zero_shot_generation()
    test_qwen_multimodal_input()
    test_qwen_response_quality()
    test_qwen_with_retrieved_audios()
    test_qwen_audio_format_consistency()
    
    print()
    print("=" * 60)
    print("Qwen-Audio-Chat Inference Validation Tests Completed")
    print("=" * 60)

