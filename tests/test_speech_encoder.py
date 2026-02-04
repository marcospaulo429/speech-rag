"""Testes para speech encoder."""

import pytest
import torch
import numpy as np
from src.speech_encoder.speech_encoder import SpeechEncoder
from src.speech_encoder.audio_processor import AudioProcessor
from src.speech_encoder.pooling import MeanPooling, MaxPooling, AttentionPooling, get_pooling_layer


def test_audio_processor_initialization():
    """Testa inicialização do processador de áudio."""
    processor = AudioProcessor(sample_rate=16000, normalize=True)
    
    assert processor.sample_rate == 16000
    assert processor.normalize == True


def test_audio_processor_process():
    """Testa processamento de áudio."""
    processor = AudioProcessor(sample_rate=16000, normalize=True)
    
    # Áudio sintético
    audio = torch.randn(16000)  # 1 segundo a 16kHz
    
    processed = processor.process(audio)
    
    assert processed.dim() == 1
    assert len(processed) == 16000
    assert torch.abs(processed).max() <= 1.0  # Normalizado


def test_audio_processor_batch():
    """Testa processamento de batch."""
    processor = AudioProcessor(sample_rate=16000)
    
    audio_list = [
        torch.randn(16000),
        torch.randn(20000),
        torch.randn(15000)
    ]
    
    batch = processor.process_batch(audio_list)
    
    assert batch.dim() == 2
    assert batch.shape[0] == 3
    assert batch.shape[1] == 20000  # Max length


def test_mean_pooling():
    """Testa mean pooling."""
    pooling = MeanPooling(dim=1)
    
    features = torch.randn(2, 10, 768)  # (B, T, D)
    pooled = pooling(features)
    
    assert pooled.shape == (2, 768)
    assert torch.allclose(pooled[0], features[0].mean(dim=0))


def test_max_pooling():
    """Testa max pooling."""
    pooling = MaxPooling(dim=1)
    
    features = torch.randn(2, 10, 768)
    pooled = pooling(features)
    
    assert pooled.shape == (2, 768)
    assert torch.allclose(pooled[0], features[0].max(dim=0)[0])


def test_attention_pooling():
    """Testa attention pooling."""
    pooling = AttentionPooling(feature_dim=768)
    
    features = torch.randn(2, 10, 768)
    pooled = pooling(features)
    
    assert pooled.shape == (2, 768)


def test_get_pooling_layer():
    """Testa factory function de pooling."""
    mean_pool = get_pooling_layer("mean")
    assert isinstance(mean_pool, MeanPooling)
    
    max_pool = get_pooling_layer("max")
    assert isinstance(max_pool, MaxPooling)
    
    attn_pool = get_pooling_layer("attention", feature_dim=768)
    assert isinstance(attn_pool, AttentionPooling)
    
    with pytest.raises(ValueError):
        get_pooling_layer("invalid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer GPU ou modelo carregado")
def test_speech_encoder_initialization():
    """Testa inicialização do speech encoder."""
    encoder = SpeechEncoder(
        model_name="facebook/hubert-base-ls960",
        pooling_method="mean",
        device="cpu",
        freeze=True
    )
    
    assert encoder.model_name == "facebook/hubert-base-ls960"
    assert encoder.feature_dim > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer GPU ou modelo carregado")
def test_speech_encoder_forward():
    """Testa forward do speech encoder."""
    encoder = SpeechEncoder(
        model_name="facebook/hubert-base-ls960",
        pooling_method="mean",
        device="cpu",
        freeze=True
    )
    
    # Áudio sintético
    audio = torch.randn(16000)  # 1 segundo
    
    # Features agregadas
    embeddings = encoder.forward(audio, return_pooled=True)
    assert embeddings.dim() == 1
    assert embeddings.shape[0] == encoder.feature_dim
    
    # Features sequenciais
    features = encoder.forward(audio, return_pooled=False)
    assert features.dim() == 2  # (T, D)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer GPU ou modelo carregado")
def test_speech_encoder_encode():
    """Testa método encode."""
    encoder = SpeechEncoder(
        model_name="facebook/hubert-base-ls960",
        pooling_method="mean",
        device="cpu",
        freeze=True
    )
    
    audio = torch.randn(16000)
    
    # Sem normalização
    embeddings = encoder.encode(audio, normalize=False)
    assert embeddings.dim() == 1
    
    # Com normalização
    embeddings_norm = encoder.encode(audio, normalize=True)
    norm = torch.norm(embeddings_norm, p=2)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer GPU ou modelo carregado")
def test_speech_encoder_batch():
    """Testa encoding de batch."""
    encoder = SpeechEncoder(
        model_name="facebook/hubert-base-ls960",
        pooling_method="mean",
        device="cpu",
        freeze=True
    )
    
    audio_list = [
        torch.randn(16000),
        torch.randn(20000),
        torch.randn(15000)
    ]
    
    embeddings = encoder.encode(audio_list, normalize=False)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == encoder.feature_dim

