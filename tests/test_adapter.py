"""Testes para speech adapter."""

import pytest
import torch
from src.adapter.adapter import SpeechAdapter, MLPAdapter, TransformerAdapter
from src.adapter.distillation_loss import DistillationLoss, cosine_similarity_loss


def test_mlp_adapter_initialization():
    """Testa inicialização do adaptador MLP."""
    adapter = MLPAdapter(
        speech_dim=768,
        text_dim=768,
        hidden_dim=512,
        num_layers=2,
        activation="relu",
        dropout=0.1
    )
    
    assert adapter.speech_dim == 768
    assert adapter.text_dim == 768
    assert adapter.hidden_dim == 512


def test_mlp_adapter_forward():
    """Testa forward do adaptador MLP."""
    adapter = MLPAdapter(
        speech_dim=768,
        text_dim=768,
        hidden_dim=512,
        num_layers=2
    )
    
    speech_emb = torch.randn(5, 768)  # Batch de 5
    text_emb = adapter(speech_emb)
    
    assert text_emb.shape == (5, 768)


def test_transformer_adapter_initialization():
    """Testa inicialização do adaptador Transformer."""
    adapter = TransformerAdapter(
        speech_dim=768,
        text_dim=768,
        hidden_dim=512,
        num_layers=2,
        num_heads=8
    )
    
    assert adapter.speech_dim == 768
    assert adapter.text_dim == 768


def test_transformer_adapter_forward():
    """Testa forward do adaptador Transformer."""
    adapter = TransformerAdapter(
        speech_dim=768,
        text_dim=768,
        hidden_dim=512,
        num_layers=2
    )
    
    speech_emb = torch.randn(5, 768)
    text_emb = adapter(speech_emb)
    
    assert text_emb.shape == (5, 768)


def test_speech_adapter_mlp():
    """Testa SpeechAdapter com arquitetura MLP."""
    adapter = SpeechAdapter(
        architecture="mlp",
        speech_dim=768,
        text_dim=768,
        hidden_dim=512
    )
    
    speech_emb = torch.randn(5, 768)
    text_emb = adapter(speech_emb)
    
    assert text_emb.shape == (5, 768)
    assert adapter.architecture == "mlp"


def test_speech_adapter_transformer():
    """Testa SpeechAdapter com arquitetura Transformer."""
    adapter = SpeechAdapter(
        architecture="transformer",
        speech_dim=768,
        text_dim=768,
        hidden_dim=512
    )
    
    speech_emb = torch.randn(5, 768)
    text_emb = adapter(speech_emb)
    
    assert text_emb.shape == (5, 768)
    assert adapter.architecture == "transformer"


def test_speech_adapter_from_config():
    """Testa criação de adapter a partir de config."""
    import tempfile
    import yaml
    import os
    
    config = {
        "architecture": "mlp",
        "speech_dim": 768,
        "text_dim": 768,
        "hidden_dim": 512,
        "num_layers": 2,
        "activation": "relu",
        "dropout": 0.1
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        adapter = SpeechAdapter.from_config(config_path)
        assert adapter.architecture == "mlp"
        assert adapter.speech_dim == 768
    finally:
        os.unlink(config_path)


def test_distillation_loss_mse():
    """Testa loss de distilação MSE."""
    loss_fn = DistillationLoss(loss_type="mse")
    
    speech_emb = torch.randn(5, 768)
    text_emb = torch.randn(5, 768)
    
    loss = loss_fn(speech_emb, text_emb, normalize=False)
    
    assert loss.item() >= 0
    assert loss.dim() == 0  # Scalar


def test_distillation_loss_cosine():
    """Testa loss de distilação Cosine."""
    loss_fn = DistillationLoss(loss_type="cosine")
    
    speech_emb = torch.randn(5, 768)
    text_emb = torch.randn(5, 768)
    
    loss = loss_fn(speech_emb, text_emb, normalize=True)
    
    assert loss.item() >= 0


def test_distillation_loss_combined():
    """Testa loss de distilação combinada."""
    loss_fn = DistillationLoss(loss_type="mse+cosine", alpha=0.5)
    
    speech_emb = torch.randn(5, 768)
    text_emb = torch.randn(5, 768)
    
    loss = loss_fn(speech_emb, text_emb, normalize=True)
    
    assert loss.item() >= 0


def test_cosine_similarity_loss():
    """Testa loss de similaridade de cosseno."""
    speech_emb = torch.randn(5, 768)
    text_emb = torch.randn(5, 768)
    
    loss = cosine_similarity_loss(speech_emb, text_emb, normalize=True)
    
    assert loss.item() >= 0
    assert loss.item() <= 2.0  # Cosine similarity está entre -1 e 1


def test_adapter_alignment():
    """Testa que adapter pode alinhar embeddings."""
    adapter = SpeechAdapter(
        architecture="mlp",
        speech_dim=768,
        text_dim=768,
        hidden_dim=512
    )
    
    speech_emb = torch.randn(5, 768)
    text_emb = torch.randn(5, 768)
    
    # Projeta speech embeddings
    projected_emb = adapter(speech_emb)
    
    # Calcula loss
    loss_fn = DistillationLoss(loss_type="mse")
    loss = loss_fn(projected_emb, text_emb, normalize=False)
    
    assert loss.item() >= 0

