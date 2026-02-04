"""Testes para retriever."""

import pytest
import torch
import numpy as np
from src.retriever.indexer import AudioIndexer
from src.retriever.index_manager import IndexManager
from src.retriever.similarity_search import SimilaritySearch
import tempfile
import os


def test_audio_indexer_initialization():
    """Testa inicialização do indexador."""
    indexer = AudioIndexer(
        embedding_dim=768,
        index_type="flat",
        normalize=True
    )
    
    assert indexer.embedding_dim == 768
    assert indexer.index_type == "flat"


def test_audio_indexer_add():
    """Testa adição de embeddings ao índice."""
    indexer = AudioIndexer(embedding_dim=768, normalize=True)
    
    embeddings = np.random.randn(10, 768).astype(np.float32)
    metadata = [{"id": i} for i in range(10)]
    
    indexer.add(embeddings, metadata)
    
    assert indexer.get_size() == 10


def test_audio_indexer_search():
    """Testa busca no índice."""
    indexer = AudioIndexer(embedding_dim=768, normalize=True)
    
    # Adiciona embeddings
    embeddings = np.random.randn(10, 768).astype(np.float32)
    indexer.add(embeddings)
    
    # Busca
    query = np.random.randn(768).astype(np.float32)
    distances, indices = indexer.search(query, k=5)
    
    assert len(indices) == 5
    assert len(distances) == 5


def test_index_manager_save_load():
    """Testa salvamento e carregamento de índices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IndexManager(index_dir=tmpdir)
        
        # Cria índice
        indexer = AudioIndexer(embedding_dim=768)
        embeddings = np.random.randn(10, 768).astype(np.float32)
        indexer.add(embeddings)
        
        # Salva
        metadata = {"test": "data"}
        path = manager.save_index(indexer.index, metadata, "test_index")
        
        assert os.path.exists(path)
        
        # Carrega
        loaded_index, loaded_metadata = manager.load_index("test_index")
        
        assert loaded_index.ntotal == 10
        assert loaded_metadata["test"] == "data"


def test_similarity_search():
    """Testa busca por similaridade."""
    search = SimilaritySearch(metric="cosine", normalize=True)
    
    query = np.random.randn(768)
    candidates = np.random.randn(20, 768)
    
    scores, indices = search.search(query, candidates, top_k=5)
    
    assert len(scores) == 5
    assert len(indices) == 5
    assert all(0 <= idx < 20 for idx in indices)


def test_similarity_search_batch():
    """Testa busca por similaridade com batch."""
    search = SimilaritySearch(metric="cosine", normalize=True)
    
    queries = np.random.randn(3, 768)
    candidates = np.random.randn(20, 768)
    
    scores, indices = search.search(queries, candidates, top_k=5)
    
    assert scores.shape == (3, 5)
    assert indices.shape == (3, 5)

