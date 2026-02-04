"""Testes para SLM generator."""

import pytest
import torch
from src.generator.slm_generator import SLMGenerator
from src.generator.audio_conditioning import AudioConditioner


def test_audio_conditioner_initialization():
    """Testa inicialização do condicionador de áudio."""
    conditioner = AudioConditioner(
        max_audio_length=30.0,
        chunk_duration=10.0,
        sample_rate=16000
    )
    
    assert conditioner.max_audio_length == 30.0
    assert conditioner.sample_rate == 16000


def test_audio_conditioner_prepare():
    """Testa preparação de áudio."""
    conditioner = AudioConditioner(
        max_audio_length=10.0,
        sample_rate=16000
    )
    
    audio = torch.randn(20 * 16000)  # 20 segundos
    prepared = conditioner.prepare_audio(audio)
    
    # Deve ser limitado a 10 segundos
    assert len(prepared) <= 10 * 16000


def test_audio_conditioner_format():
    """Testa formatação para SLM."""
    conditioner = AudioConditioner()
    
    audio = torch.randn(16000)
    formatted = conditioner.format_for_slm(audio, prompt="Test prompt")
    
    assert "audio" in formatted
    assert "prompt" in formatted
    assert formatted["prompt"] == "Test prompt"


@pytest.mark.skipif(True, reason="Requer modelo carregado - skip por padrão")
def test_slm_generator_initialization():
    """Testa inicialização do gerador SLM."""
    generator = SLMGenerator(
        model_name="Qwen/Qwen-Audio-Chat",
        device="cpu",
        freeze=True
    )
    
    assert generator.model_name == "Qwen/Qwen-Audio-Chat"


@pytest.mark.skipif(True, reason="Requer modelo carregado - skip por padrão")
def test_slm_generator_generate():
    """Testa geração com SLM."""
    generator = SLMGenerator(
        model_name="Qwen/Qwen-Audio-Chat",
        device="cpu",
        freeze=True
    )
    
    audio = torch.randn(16000)
    prompt = "What is this audio about?"
    
    result = generator.generate(audio, prompt)
    
    assert isinstance(result, str)
    assert len(result) > 0

