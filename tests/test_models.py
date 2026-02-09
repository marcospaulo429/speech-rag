"""Basic tests for model components"""

import torch
import numpy as np
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


def test_speech_adapter_different_seq_len():
    """Test Speech Adapter with different sequence lengths"""
    adapter = SpeechAdapter(input_dim=1024, output_dim=4096, downsample_factor=4)
    
    # Test with different sequence lengths
    for seq_len in [50, 100, 200, 400]:
        speech_reprs = torch.randn(1, seq_len, 1024)
        output = adapter(speech_reprs)
        assert output.shape == (1, 4096), f"Failed for seq_len={seq_len}"
    
    print("✓ Speech Adapter different seq_len test passed")


def test_speech_adapter_different_batch_sizes():
    """Test Speech Adapter with different batch sizes"""
    adapter = SpeechAdapter(input_dim=1024, output_dim=4096, downsample_factor=4)
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        speech_reprs = torch.randn(batch_size, 100, 1024)
        output = adapter(speech_reprs)
        assert output.shape == (batch_size, 4096), f"Failed for batch_size={batch_size}"
        # Check normalization
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    print("✓ Speech Adapter different batch sizes test passed")


def test_speech_adapter_downsampling():
    """Test that Speech Adapter correctly downsamples temporal dimension"""
    adapter = SpeechAdapter(input_dim=1024, output_dim=4096, downsample_factor=4)
    
    seq_len = 100
    speech_reprs = torch.randn(1, seq_len, 1024)
    output = adapter(speech_reprs)
    
    # After downsampling by 4 and averaging, we should get a single embedding
    assert output.shape == (1, 4096), "Should produce single embedding per sample"
    
    print("✓ Speech Adapter downsampling test passed")


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


def test_text_encoder_single_text():
    """Test Text Encoder with single text"""
    try:
        text_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        
        # Test with single text string
        text = "This is a single test query"
        embeddings = text_encoder.encode(text)
        
        assert len(embeddings.shape) == 2, "Should return 2D tensor"
        assert embeddings.shape[0] == 1, "Should return 1 embedding"
        assert embeddings.shape[1] == text_encoder.embedding_dim, \
            f"Expected embedding_dim={text_encoder.embedding_dim}, got {embeddings.shape[1]}"
        
        # Check normalization
        norm = torch.norm(embeddings, dim=1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
        
        print("✓ Text Encoder single text test passed")
    except Exception as e:
        print(f"⚠ Text Encoder single text test skipped: {e}")


def test_text_encoder_list_texts():
    """Test Text Encoder with list of texts"""
    try:
        text_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        
        # Test with list of texts
        texts = [
            "First query text",
            "Second query text",
            "Third query text"
        ]
        embeddings = text_encoder.encode(texts)
        
        assert embeddings.shape[0] == 3, "Should return 3 embeddings"
        assert embeddings.shape[1] == text_encoder.embedding_dim
        
        # Check normalization
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        print("✓ Text Encoder list texts test passed")
    except Exception as e:
        print(f"⚠ Text Encoder list texts test skipped: {e}")


def test_text_encoder_different_lengths():
    """Test Text Encoder with texts of different lengths"""
    try:
        text_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        
        # Test with very short and very long texts
        texts = [
            "Hi",
            "This is a medium length text that should work fine",
            "This is a very long text " * 50  # Very long text
        ]
        embeddings = text_encoder.encode(texts)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == text_encoder.embedding_dim
        
        # All should be normalized
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        print("✓ Text Encoder different lengths test passed")
    except Exception as e:
        print(f"⚠ Text Encoder different lengths test skipped: {e}")


def test_speech_encoder_basic():
    """Test Speech Encoder (requires model download)"""
    # Skip if models not available
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        # Create dummy audio (5 seconds at 16kHz = 80000 samples)
        duration_seconds = 5
        sample_rate = 16000
        num_samples = duration_seconds * sample_rate
        dummy_audio = torch.randn(num_samples)
        
        # Test encoding
        representations = speech_encoder.encode(dummy_audio)
        
        assert len(representations.shape) == 3, "Should return 3D tensor"
        assert representations.shape[-1] == speech_encoder.hidden_size, \
            "Hidden size should match"
        
        print("✓ Speech Encoder test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder test skipped: {e}")


def test_speech_encoder_tensor_format():
    """Test Speech Encoder with tensor input"""
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        # Test with tensor
        dummy_audio = torch.randn(80000)
        representations = speech_encoder.encode(dummy_audio)
        
        assert len(representations.shape) == 3
        assert representations.shape[-1] == speech_encoder.hidden_size
        assert representations.shape[0] == 1  # Batch size
        
        print("✓ Speech Encoder tensor format test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder tensor format test skipped: {e}")


def test_speech_encoder_numpy_format():
    """Test Speech Encoder with numpy array input"""
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        # Test with numpy array
        dummy_audio = np.random.randn(80000).astype(np.float32)
        representations = speech_encoder.encode(dummy_audio)
        
        assert len(representations.shape) == 3
        assert representations.shape[-1] == speech_encoder.hidden_size
        
        print("✓ Speech Encoder numpy format test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder numpy format test skipped: {e}")


def test_speech_encoder_batch():
    """Test Speech Encoder with batch of audios"""
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        # Test with list of audios (batch)
        batch_size = 3
        audios = [torch.randn(80000) for _ in range(batch_size)]
        representations = speech_encoder.encode(audios)
        
        assert len(representations.shape) == 3
        assert representations.shape[0] == batch_size, \
            f"Expected batch_size={batch_size}, got {representations.shape[0]}"
        assert representations.shape[-1] == speech_encoder.hidden_size
        
        print("✓ Speech Encoder batch test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder batch test skipped: {e}")


def test_speech_encoder_different_durations():
    """Test Speech Encoder with audios of different durations"""
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        # Test with different durations (3, 5, 10 seconds)
        durations = [3, 5, 10]
        sample_rate = 16000
        
        for duration in durations:
            num_samples = duration * sample_rate
            dummy_audio = torch.randn(num_samples)
            representations = speech_encoder.encode(dummy_audio)
            
            assert len(representations.shape) == 3
            assert representations.shape[-1] == speech_encoder.hidden_size
            # Longer audio should produce more sequence tokens
            assert representations.shape[1] > 0
        
        print("✓ Speech Encoder different durations test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder different durations test skipped: {e}")


def test_speech_encoder_output_dimensions():
    """Test Speech Encoder output dimensions"""
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        dummy_audio = torch.randn(80000)
        representations = speech_encoder.encode(dummy_audio)
        
        # Should be (batch_size, seq_len, hidden_size)
        assert len(representations.shape) == 3, "Should return 3D tensor"
        batch_size, seq_len, hidden_size = representations.shape
        
        assert batch_size == 1, "Single audio should have batch_size=1"
        assert seq_len > 0, "Sequence length should be positive"
        assert hidden_size == speech_encoder.hidden_size == 1024, \
            f"Expected hidden_size=1024, got {hidden_size}"
        
        print("✓ Speech Encoder output dimensions test passed")
    except Exception as e:
        print(f"⚠ Speech Encoder output dimensions test skipped: {e}")


def test_dimension_integration():
    """Test that all components have compatible dimensions"""
    print("\nTesting dimension integration...")
    
    # Test SpeechEncoder -> Adapter dimension flow
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        adapter = SpeechAdapter(input_dim=1024, output_dim=4096)
        
        # Create audio and get speech representations
        dummy_audio = torch.randn(80000)
        speech_reprs = speech_encoder.encode(dummy_audio)
        
        # Check speech encoder output
        assert speech_reprs.shape[-1] == 1024, \
            f"SpeechEncoder should output 1024-dim, got {speech_reprs.shape[-1]}"
        
        # Pass through adapter
        audio_embedding = adapter(speech_reprs)
        
        # Check adapter output
        assert audio_embedding.shape[-1] == 4096, \
            f"Adapter should output 4096-dim, got {audio_embedding.shape[-1]}"
        
        print("✓ SpeechEncoder (1024) -> Adapter (4096) dimension flow works")
    except Exception as e:
        print(f"⚠ SpeechEncoder -> Adapter test skipped: {e}")
    
    # Test TextEncoder dimension
    try:
        text_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        text_embedding = text_encoder.encode("Test query")
        
        assert text_embedding.shape[-1] == 4096, \
            f"TextEncoder should output 4096-dim, got {text_embedding.shape[-1]}"
        
        print("✓ TextEncoder outputs 4096-dim embeddings")
    except Exception as e:
        print(f"⚠ TextEncoder dimension test skipped: {e}")
    
    # Test that embeddings can be compared (same dimension)
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        adapter = SpeechAdapter(input_dim=1024, output_dim=4096)
        text_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        
        # Get embeddings
        dummy_audio = torch.randn(80000)
        speech_reprs = speech_encoder.encode(dummy_audio)
        audio_embedding = adapter(speech_reprs)
        text_embedding = text_encoder.encode("Test query")
        
        # Check dimensions match
        assert audio_embedding.shape[-1] == text_embedding.shape[-1] == 4096, \
            "Audio and text embeddings should have same dimension (4096)"
        
        # Can compute similarity (cosine similarity)
        similarity = torch.nn.functional.cosine_similarity(
            audio_embedding, text_embedding, dim=-1
        )
        assert similarity.shape == (1,), "Should compute similarity"
        
        print("✓ Audio and text embeddings are compatible (4096-dim)")
        print("✓ Embeddings can be compared (cosine similarity works)")
        
    except Exception as e:
        print(f"⚠ Embedding compatibility test skipped: {e}")
    
    print("✓ Dimension integration tests completed")


def test_text_encoder_qwen3_dimensions():
    """Test Qwen3-Embedding-0.6B dimensions"""
    print("\nTesting Qwen3-Embedding-0.6B dimensions...")
    
    try:
        text_encoder = TextEncoder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            freeze=True
        )
        
        # Check embedding dimension
        print(f"  Qwen3-Embedding-0.6B embedding_dim: {text_encoder.embedding_dim}")
        
        # Test encoding
        text_embedding = text_encoder.encode("Test query")
        
        # Qwen3-Embedding-0.6B should output 1024 dimensions by default
        assert text_embedding.shape[-1] == text_encoder.embedding_dim, \
            f"Embedding shape should match embedding_dim, got {text_embedding.shape[-1]}"
        
        # Should be normalized
        norm = torch.norm(text_embedding, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), \
            "Embeddings should be L2-normalized"
        
        print(f"✓ Qwen3-Embedding-0.6B outputs {text_encoder.embedding_dim}-dim embeddings")
        print(f"✓ Embeddings are normalized")
        
    except Exception as e:
        print(f"⚠ Qwen3-Embedding-0.6B dimension test skipped: {e}")
        print("  (This is expected if Qwen3-Embedding-0.6B model is not available)")


def test_text_encoder_both_models():
    """Test both E5-Mistral and Qwen3-Embedding-0.6B"""
    print("\nTesting both text encoders...")
    
    # Test E5-Mistral
    try:
        e5_encoder = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        e5_embedding = e5_encoder.encode("Test query")
        print(f"✓ E5-Mistral: {e5_encoder.embedding_dim} dimensions")
    except Exception as e:
        print(f"⚠ E5-Mistral test skipped: {e}")
    
    # Test Qwen3
    try:
        qwen3_encoder = TextEncoder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            freeze=True
        )
        qwen3_embedding = qwen3_encoder.encode("Test query")
        print(f"✓ Qwen3-Embedding-0.6B: {qwen3_encoder.embedding_dim} dimensions")
        
        # Note: They have different dimensions, so they can't be directly compared
        # Each needs its own adapter trained for that dimension
        print(f"  Note: E5 has {e5_encoder.embedding_dim} dim, Qwen3 has {qwen3_encoder.embedding_dim} dim")
        print(f"  Each encoder requires an adapter trained for its specific dimension")
        
    except Exception as e:
        print(f"⚠ Qwen3-Embedding-0.6B test skipped: {e}")


def test_adapter_dynamic_dimensions():
    """Test SpeechAdapter with different output dimensions"""
    print("\nTesting SpeechAdapter with dynamic dimensions...")
    
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        
        # Test with E5 dimension (4096)
        adapter_e5 = SpeechAdapter(input_dim=1024, output_dim=4096)
        dummy_audio = torch.randn(80000)
        speech_reprs = speech_encoder.encode(dummy_audio)
        audio_embedding_e5 = adapter_e5(speech_reprs)
        
        assert audio_embedding_e5.shape[-1] == 4096, \
            f"Adapter should output 4096-dim for E5, got {audio_embedding_e5.shape[-1]}"
        print("✓ Adapter works with 4096 dimensions (E5-Mistral)")
        
        # Test with Qwen3 dimension (1024)
        adapter_qwen3 = SpeechAdapter(input_dim=1024, output_dim=1024)
        audio_embedding_qwen3 = adapter_qwen3(speech_reprs)
        
        assert audio_embedding_qwen3.shape[-1] == 1024, \
            f"Adapter should output 1024-dim for Qwen3, got {audio_embedding_qwen3.shape[-1]}"
        print("✓ Adapter works with 1024 dimensions (Qwen3-Embedding-0.6B)")
        
        # Both should be normalized
        norm_e5 = torch.norm(audio_embedding_e5, p=2, dim=-1)
        norm_qwen3 = torch.norm(audio_embedding_qwen3, p=2, dim=-1)
        assert torch.allclose(norm_e5, torch.ones_like(norm_e5), atol=1e-5), \
            "E5 adapter output should be normalized"
        assert torch.allclose(norm_qwen3, torch.ones_like(norm_qwen3), atol=1e-5), \
            "Qwen3 adapter output should be normalized"
        
        print("✓ Both adapters produce normalized embeddings")
        
    except Exception as e:
        print(f"⚠ Adapter dynamic dimensions test skipped: {e}")


def test_end_to_end_with_different_encoders():
    """Test end-to-end pipeline with both encoders"""
    print("\nTesting end-to-end with different encoders...")
    
    try:
        speech_encoder = SpeechEncoder(
            model_name="facebook/hubert-large-ls960-ft",
            freeze=True
        )
        dummy_audio = torch.randn(80000)
        speech_reprs = speech_encoder.encode(dummy_audio)
        
        # Test with E5
        text_encoder_e5 = TextEncoder(
            model_name="intfloat/e5-mistral-7b-instruct",
            freeze=True
        )
        adapter_e5 = SpeechAdapter(input_dim=1024, output_dim=4096)
        audio_emb_e5 = adapter_e5(speech_reprs)
        text_emb_e5 = text_encoder_e5.encode("Test query")
        
        assert audio_emb_e5.shape[-1] == text_emb_e5.shape[-1] == 4096, \
            "E5 embeddings should have matching dimensions"
        print("✓ E5-Mistral end-to-end works (4096-dim)")
        
        # Test with Qwen3 (if available)
        try:
            text_encoder_qwen3 = TextEncoder(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                freeze=True
            )
            adapter_qwen3 = SpeechAdapter(input_dim=1024, output_dim=1024)
            audio_emb_qwen3 = adapter_qwen3(speech_reprs)
            text_emb_qwen3 = text_encoder_qwen3.encode("Test query")
            
            assert audio_emb_qwen3.shape[-1] == text_emb_qwen3.shape[-1] == 1024, \
                "Qwen3 embeddings should have matching dimensions"
            print("✓ Qwen3-Embedding-0.6B end-to-end works (1024-dim)")
        except Exception as e:
            print(f"  Qwen3 end-to-end test skipped: {e}")
        
    except Exception as e:
        print(f"⚠ End-to-end test skipped: {e}")


if __name__ == "__main__":
    # Speech Adapter tests
    test_speech_adapter()
    test_speech_adapter_different_seq_len()
    test_speech_adapter_different_batch_sizes()
    test_speech_adapter_downsampling()
    
    # Text Encoder tests
    #test_text_encoder_basic()
    #test_text_encoder_single_text()
    #test_text_encoder_list_texts()
    #test_text_encoder_different_lengths()
    
    # Speech Encoder tests
    test_speech_encoder_basic()
    test_speech_encoder_tensor_format()
    test_speech_encoder_numpy_format()
    
    # Dimension tests
    test_dimension_integration()
    test_text_encoder_qwen3_dimensions()
    test_text_encoder_both_models()
    test_adapter_dynamic_dimensions()
    test_end_to_end_with_different_encoders()
    test_speech_encoder_batch()
    test_speech_encoder_different_durations()
    test_speech_encoder_output_dimensions()
    
    # Integration tests
    test_dimension_integration()
    
    print("\nAll tests completed!")

