"""Testes para text embedder."""

import pytest
import torch
from src.text_embedder.text_embedder import TextEmbedder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer modelo carregado")
def test_text_embedder_initialization():
    """Testa inicialização do text embedder."""
    embedder = TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        device="cpu",
        freeze=True,
        use_sentence_transformers=True
    )
    
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.feature_dim > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer modelo carregado")
def test_text_embedder_forward_single():
    """Testa forward com texto único."""
    embedder = TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        device="cpu",
        use_sentence_transformers=True
    )
    
    text = "This is a test sentence."
    embeddings = embedder.forward(text, return_pooled=True)
    
    assert embeddings.dim() == 1
    assert embeddings.shape[0] == embedder.feature_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer modelo carregado")
def test_text_embedder_forward_batch():
    """Testa forward com batch de textos."""
    embedder = TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        device="cpu",
        use_sentence_transformers=True
    )
    
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence."
    ]
    
    embeddings = embedder.forward(texts, return_pooled=True)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == embedder.feature_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer modelo carregado")
def test_text_embedder_encode():
    """Testa método encode."""
    embedder = TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        device="cpu",
        use_sentence_transformers=True
    )
    
    text = "Test sentence for encoding."
    embeddings = embedder.encode(text, normalize=True)
    
    # Verifica normalização
    norm = torch.norm(embeddings, p=2)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer modelo carregado")
def test_text_embedder_similarity():
    """Testa similaridade entre embeddings."""
    embedder = TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        device="cpu",
        use_sentence_transformers=True
    )
    
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    text3 = "The weather is nice today."
    
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    emb3 = embedder.encode(text3)
    
    # Textos similares devem ter maior similaridade
    sim_12 = torch.dot(emb1, emb2).item()
    sim_13 = torch.dot(emb1, emb3).item()
    
    assert sim_12 > sim_13  # text1 e text2 são mais similares


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requer modelo carregado")
def test_text_embedder_get_feature_dim():
    """Testa obtenção de dimensão de features."""
    embedder = TextEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        use_sentence_transformers=True
    )
    
    dim = embedder.get_feature_dim()
    assert dim > 0
    assert isinstance(dim, int)

