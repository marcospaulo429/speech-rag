"""Cross-modal Retriever: busca passagens de áudio usando consultas em texto."""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import yaml

from ..speech_encoder.speech_encoder import SpeechEncoder
from ..text_embedder.text_embedder import TextEmbedder
from ..adapter.adapter import SpeechAdapter
from .indexer import AudioIndexer
from .index_manager import IndexManager
from .similarity_search import SimilaritySearch


class SpeechRetriever:
    """Retriever cross-modal para busca de áudio a partir de texto."""
    
    def __init__(
        self,
        speech_encoder: SpeechEncoder,
        text_embedder: TextEmbedder,
        adapter: SpeechAdapter,
        top_k: int = 5,
        index_type: str = "flat",
        normalize_embeddings: bool = True,
        device: Optional[str] = None
    ):
        """
        Inicializa o retriever.
        
        Args:
            speech_encoder: Encoder de fala
            text_embedder: Embedder de texto
            adapter: Adaptador de fala para texto
            top_k: Número de resultados a retornar
            index_type: Tipo de índice FAISS
            normalize_embeddings: Se True, normaliza embeddings
            device: Device para processamento
        """
        self.speech_encoder = speech_encoder
        self.text_embedder = text_embedder
        self.adapter = adapter
        self.top_k = top_k
        self.normalize_embeddings = normalize_embeddings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move adapter para device
        self.adapter = self.adapter.to(self.device)
        
        # Indexador
        embedding_dim = text_embedder.get_feature_dim()
        self.indexer = AudioIndexer(
            embedding_dim=embedding_dim,
            index_type=index_type,
            normalize=normalize_embeddings
        )
        
        # Index manager
        self.index_manager = IndexManager()
        
        # Similarity search (fallback)
        self.similarity_search = SimilaritySearch(
            metric="cosine",
            normalize=normalize_embeddings
        )
        
        # Armazena passagens de áudio
        self.audio_passages: List[Dict[str, Any]] = []
    
    @classmethod
    def from_config(
        cls,
        config_path: str,
        speech_encoder: SpeechEncoder,
        text_embedder: TextEmbedder,
        adapter: SpeechAdapter,
        device: Optional[str] = None
    ):
        """
        Cria retriever a partir de configuração.
        
        Args:
            config_path: Caminho para arquivo YAML
            speech_encoder: Encoder de fala
            text_embedder: Embedder de texto
            adapter: Adaptador
            device: Device
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            speech_encoder=speech_encoder,
            text_embedder=text_embedder,
            adapter=adapter,
            top_k=config.get("top_k", 5),
            index_type=config.get("index_type", "flat"),
            normalize_embeddings=config.get("normalize_embeddings", True),
            device=device
        )
    
    def index_audio_passages(
        self,
        audio_passages: List[Union[torch.Tensor, str]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Indexa passagens de áudio.
        
        Args:
            audio_passages: Lista de áudios (tensores ou caminhos)
            metadata: Metadados opcionais para cada passagem
        """
        # Armazena passagens
        self.audio_passages = [
            {"audio": audio, "metadata": meta or {}}
            for audio, meta in zip(audio_passages, metadata or [{}] * len(audio_passages))
        ]
        
        # Extrai embeddings de áudio
        audio_embeddings = []
        for audio in audio_passages:
            # Encoding de áudio
            speech_emb = self.speech_encoder.encode(audio, normalize=False)
            
            # Projeção via adapter
            with torch.no_grad():
                text_emb = self.adapter(speech_emb)
            
            audio_embeddings.append(text_emb.cpu().numpy())
        
        # Converte para numpy
        audio_embeddings = np.array(audio_embeddings)
        
        # Indexa
        self.indexer.add(audio_embeddings, metadata)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca passagens de áudio relevantes para uma consulta em texto.
        
        Args:
            query: Consulta em texto
            top_k: Número de resultados (None = usa self.top_k)
        
        Returns:
            Lista de resultados com passagens e scores
        """
        top_k = top_k or self.top_k
        
        # Embedding da consulta
        query_emb = self.text_embedder.encode(query, normalize=self.normalize_embeddings)
        query_emb = query_emb.cpu().numpy()
        
        # Busca no índice
        distances, indices = self.indexer.search(query_emb, k=top_k)
        
        # Converte distâncias para scores (assumindo inner product ou L2)
        if self.normalize_embeddings:
            # Inner product: maior = mais similar
            scores = distances[0] if distances.ndim > 1 else distances
        else:
            # L2: menor = mais similar, converte para score
            scores = 1.0 / (1.0 + distances[0] if distances.ndim > 1 else distances)
        
        # Obtém resultados
        results = []
        indices_flat = indices[0] if indices.ndim > 1 else indices
        
        for i, idx in enumerate(indices_flat):
            if idx < len(self.audio_passages):
                passage = self.audio_passages[int(idx)]
                results.append({
                    "audio": passage["audio"],
                    "metadata": passage["metadata"],
                    "score": float(scores[i]),
                    "rank": i + 1
                })
        
        return results
    
    def save_index(self, name: str, version: Optional[str] = None) -> str:
        """
        Salva índice e metadados.
        
        Args:
            name: Nome do índice
            version: Versão (None = timestamp)
        
        Returns:
            Caminho do arquivo salvo
        """
        metadata = {
            "audio_passages": self.audio_passages,
            "top_k": self.top_k,
            "index_type": self.indexer.index_type,
            "normalize_embeddings": self.normalize_embeddings
        }
        
        return self.index_manager.save_index(
            self.indexer.index,
            metadata,
            name,
            version
        )
    
    def load_index(self, name: str, version: Optional[str] = None):
        """
        Carrega índice e metadados.
        
        Args:
            name: Nome do índice
            version: Versão (None = mais recente)
        """
        index, metadata = self.index_manager.load_index(name, version)
        
        self.indexer.index = index
        self.audio_passages = metadata.get("audio_passages", [])
        self.top_k = metadata.get("top_k", 5)
        self.indexer.index_type = metadata.get("index_type", "flat")
        self.normalize_embeddings = metadata.get("normalize_embeddings", True)

