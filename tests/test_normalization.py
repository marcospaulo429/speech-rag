"""Testes para normalização de embeddings."""

import pytest
import torch
import numpy as np
from src.utils.normalization import normalize_l2, normalize_batch, cosine_similarity


def test_normalize_l2_torch():
    """Testa normalização L2 com torch.Tensor."""
    embeddings = torch.randn(10, 768)
    normalized = normalize_l2(embeddings)
    
    # Verifica que a norma é aproximadamente 1
    norms = torch.norm(normalized, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    # Verifica que a dimensão é mantida
    assert normalized.shape == embeddings.shape


def test_normalize_l2_numpy():
    """Testa normalização L2 com numpy.ndarray."""
    embeddings = np.random.randn(10, 768)
    normalized = normalize_l2(embeddings)
    
    # Verifica que a norma é aproximadamente 1
    norms = np.linalg.norm(normalized, ord=2, axis=-1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)
    
    # Verifica que a dimensão é mantida
    assert normalized.shape == embeddings.shape


def test_normalize_l2_single_vector():
    """Testa normalização de um único vetor."""
    embeddings = torch.randn(768)
    normalized = normalize_l2(embeddings)
    
    norm = torch.norm(normalized, p=2)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)


def test_normalize_batch_torch():
    """Testa normalização de batch com torch.Tensor."""
    embeddings = torch.randn(32, 768)
    normalized = normalize_batch(embeddings)
    
    norms = torch.norm(normalized, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_normalize_batch_numpy():
    """Testa normalização de batch com numpy.ndarray."""
    embeddings = np.random.randn(32, 768)
    normalized = normalize_batch(embeddings)
    
    norms = np.linalg.norm(normalized, ord=2, axis=-1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)


def test_cosine_similarity_torch():
    """Testa cálculo de similaridade de cosseno com torch."""
    emb1 = normalize_l2(torch.randn(5, 768))
    emb2 = normalize_l2(torch.randn(3, 768))
    
    similarity = cosine_similarity(emb1, emb2)
    
    # Verifica dimensões
    assert similarity.shape == (5, 3)
    
    # Verifica que valores estão entre -1 e 1
    assert torch.all(similarity >= -1.0) and torch.all(similarity <= 1.0)
    
    # Diagonal de similaridade consigo mesmo deve ser ~1
    self_sim = cosine_similarity(emb1, emb1)
    assert torch.allclose(torch.diag(self_sim), torch.ones(5), atol=1e-5)


def test_cosine_similarity_numpy():
    """Testa cálculo de similaridade de cosseno com numpy."""
    emb1 = normalize_l2(np.random.randn(5, 768))
    emb2 = normalize_l2(np.random.randn(3, 768))
    
    similarity = cosine_similarity(emb1, emb2)
    
    # Verifica dimensões
    assert similarity.shape == (5, 3)
    
    # Verifica que valores estão entre -1 e 1
    assert np.all(similarity >= -1.0) and np.all(similarity <= 1.0)


def test_normalize_zero_vector():
    """Testa normalização de vetor zero (deve evitar divisão por zero)."""
    embeddings = torch.zeros(10, 768)
    normalized = normalize_l2(embeddings)
    
    # Não deve gerar NaN ou Inf
    assert not torch.any(torch.isnan(normalized))
    assert not torch.any(torch.isinf(normalized))

