"""Testes para chunking de áudio."""

import pytest
import torch
import numpy as np
from src.utils.audio_chunking import AudioChunker, AudioChunk


def test_chunker_initialization():
    """Testa inicialização do chunker."""
    chunker = AudioChunker(
        chunk_duration=10.0,
        overlap=2.0,
        sample_rate=16000
    )
    
    assert chunker.chunk_duration == 10.0
    assert chunker.overlap == 2.0
    assert chunker.sample_rate == 16000
    assert chunker.chunk_samples == 160000
    assert chunker.overlap_samples == 32000
    assert chunker.step_samples == 128000


def test_chunk_short_audio():
    """Testa chunking de áudio curto (menor que um chunk)."""
    chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
    
    # Áudio de 5 segundos
    audio = np.random.randn(5 * 16000)
    chunks = chunker.chunk_audio(audio)
    
    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 5.0
    assert chunks[0].chunk_id == 0


def test_chunk_long_audio():
    """Testa chunking de áudio longo."""
    chunker = AudioChunker(chunk_duration=10.0, overlap=2.0, sample_rate=16000)
    
    # Áudio de 30 segundos
    audio = np.random.randn(30 * 16000)
    chunks = chunker.chunk_audio(audio)
    
    # Deve ter múltiplos chunks
    assert len(chunks) > 1
    
    # Verifica que chunks têm duração correta
    for chunk in chunks:
        duration = chunk.end_time - chunk.start_time
        assert duration <= chunker.chunk_duration
    
    # Verifica overlap entre chunks consecutivos
    for i in range(len(chunks) - 1):
        overlap_time = chunks[i].end_time - chunks[i+1].start_time
        assert overlap_time >= 0  # Pode haver overlap ou não


def test_chunk_torch_tensor():
    """Testa chunking com torch.Tensor."""
    chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
    
    audio = torch.randn(30 * 16000)
    chunks = chunker.chunk_audio(audio)
    
    assert len(chunks) > 1
    assert isinstance(chunks[0].audio, np.ndarray)  # Converte para numpy internamente


def test_chunk_with_metadata():
    """Testa chunking com metadados."""
    chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
    
    audio = np.random.randn(30 * 16000)
    metadata = {"source": "test_audio.wav"}
    chunks = chunker.chunk_audio(audio, metadata=metadata)
    
    assert all(chunk.metadata == metadata for chunk in chunks)


def test_reconstruct_concatenate():
    """Testa reconstrução por concatenação."""
    chunker = AudioChunker(chunk_duration=10.0, overlap=2.0, sample_rate=16000)
    
    # Áudio de 25 segundos
    original_audio = np.random.randn(25 * 16000)
    chunks = chunker.chunk_audio(original_audio)
    
    reconstructed = chunker.reconstruct_audio(chunks, method="concatenate")
    
    # Áudio reconstruído deve ter pelo menos o tamanho original
    assert len(reconstructed) >= len(original_audio)


def test_reconstruct_overlap_average():
    """Testa reconstrução por média de overlap."""
    chunker = AudioChunker(chunk_duration=10.0, overlap=2.0, sample_rate=16000)
    
    # Áudio de 25 segundos
    original_audio = np.random.randn(25 * 16000)
    chunks = chunker.chunk_audio(original_audio)
    
    reconstructed = chunker.reconstruct_audio(chunks, method="overlap_average")
    
    # Áudio reconstruído deve ter tamanho similar ao original
    assert abs(len(reconstructed) - len(original_audio)) < chunker.sample_rate


def test_chunk_preserves_context():
    """Testa que chunks preservam contexto através do overlap."""
    chunker = AudioChunker(chunk_duration=10.0, overlap=2.0, sample_rate=16000)
    
    audio = np.random.randn(30 * 16000)
    chunks = chunker.chunk_audio(audio)
    
    # Verifica que chunks consecutivos têm overlap
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i].end_time
        chunk2_start = chunks[i+1].start_time
        
        # Deve haver overlap ou pelo menos continuidade
        assert chunk2_start <= chunk1_end


def test_reconstruct_invalid_method():
    """Testa erro com método de reconstrução inválido."""
    chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
    
    audio = np.random.randn(20 * 16000)
    chunks = chunker.chunk_audio(audio)
    
    with pytest.raises(ValueError):
        chunker.reconstruct_audio(chunks, method="invalid_method")


def test_reconstruct_empty_chunks():
    """Testa erro com lista vazia de chunks."""
    chunker = AudioChunker(chunk_duration=10.0, sample_rate=16000)
    
    with pytest.raises(ValueError):
        chunker.reconstruct_audio([])

