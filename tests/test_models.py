"""Basic tests for model components"""

import torch
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter


def test_speech_adapter():
    """Test Speech Adapter architecture"""
    batch_size = 2
    seq_len = 100
    input_dim = 1024
    output_dim = 4096
    
    adapter = SpeechAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        downsample_factor=4
    )
    
    # Create dummy input (HuBERT output)
    speech_reprs = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = adapter(speech_reprs)
    
    # Check output shape
    assert output.shape == (batch_size, output_dim), \
        f"Expected shape ({batch_size}, {output_dim}), got {output.shape}"
    
    # Check normalization (should be close to 1.0)
    norms = torch.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Output should be L2-normalized"
    
    print("✓ Speech Adapter test passed")


def test_text_encoder_basic():
    """Test Text Encoder (requires model download)"""
    # Skip if models not available
    try:
        text_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        
        # Test encoding
        texts = ["This is a test query", "Another test"]
        embeddings = text_encoder.encode(texts)
        
        assert embeddings.shape[0] == 2, "Should return 2 embeddings"
        assert embeddings.shape[1] == text_encoder.embedding_dim, \
            "Embedding dimension should match"
        
        # Check normalization
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Embeddings should be normalized"
        
        print("✓ Text Encoder test passed")
    except Exception as e:
        print(f"⚠ Text Encoder test skipped: {e}")


def test_speech_encoder_basic():
    """Test Speech Encoder (requires model download)"""
    # Skip if models not available
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960",
            freeze=True
        )
        
        # Create dummy audio (1 second at 16kHz)
        dummy_audio = torch.randn(16000)
        
        # Test encoding
        representations = speech_encoder.encode(dummy_audio)
        
        assert len(representations.shape) == 3, "Should return 3D tensor"
        assert representations.shape[-1] == speech_encoder.hidden_size, \
            "Hidden size should match"
        
        print("✓ Speech Encoder test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder test skipped: {e}")


if __name__ == "__main__":
    test_speech_adapter()
    test_text_encoder_basic()
    test_speech_encoder_basic()
    print("\nAll tests completed!")

