"""Sistema de indexação de embeddings de áudio."""

import faiss
import numpy as np
from typing import List, Optional, Dict, Any
from ..utils.normalization import normalize_l2


class AudioIndexer:
    """Indexa embeddings de áudio usando FAISS."""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        normalize: bool = True,
        use_gpu: bool = False
    ):
        """
        Inicializa o indexador.
        
        Args:
            embedding_dim: Dimensão dos embeddings
            index_type: Tipo de índice ("flat", "ivf", "hnsw")
            normalize: Se True, normaliza embeddings antes de indexar
            use_gpu: Se True, usa GPU para busca
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.normalize = normalize
        self.use_gpu = use_gpu
        
        # Cria índice
        self.index = self._create_index()
        
        # Mapeamento de IDs
        self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
    
    def _create_index(self) -> faiss.Index:
        """Cria índice FAISS."""
        if self.index_type == "flat":
            # Índice plano (exato, mas lento para grandes datasets)
            if self.normalize:
                index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product para embeddings normalizados
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance
        elif self.index_type == "ivf":
            # IVF (Inverted File Index) - mais rápido para grandes datasets
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)  # 100 clusters
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) - rápido e preciso
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 conexões
        else:
            raise ValueError(f"Tipo de índice desconhecido: {self.index_type}")
        
        # Move para GPU se solicitado
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index
    
    def add(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Adiciona embeddings ao índice.
        
        Args:
            embeddings: Embeddings para indexar (N, D)
            metadata: Lista de metadados para cada embedding
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings devem ser 2D (N, D), recebido: {embeddings.shape}")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Dimensão incorreta: esperado {self.embedding_dim}, recebido {embeddings.shape[1]}")
        
        # Normaliza se solicitado
        if self.normalize:
            embeddings = normalize_l2(embeddings)
        
        # Converte para float32 (requerido pelo FAISS)
        embeddings = embeddings.astype(np.float32)
        
        # Adiciona ao índice
        if self.index_type == "ivf" and not self.index.is_trained:
            # Treina índice IVF
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        
        # Adiciona metadados
        num_embeddings = embeddings.shape[0]
        if metadata is None:
            metadata = [{}] * num_embeddings
        
        for i, meta in enumerate(metadata):
            self.id_to_metadata[self.next_id] = meta
            self.next_id += 1
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> tuple:
        """
        Busca no índice.
        
        Args:
            query_embeddings: Embeddings de consulta (M, D) ou (D,)
            k: Número de resultados a retornar
        
        Returns:
            Tupla (distances, indices) onde:
            - distances: Distâncias (M, k) ou (k,)
            - indices: Índices dos resultados (M, k) ou (k,)
        """
        # Adiciona dimensão se necessário
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Dimensão incorreta: esperado {self.embedding_dim}, recebido {query_embeddings.shape[1]}")
        
        # Normaliza se solicitado
        if self.normalize:
            query_embeddings = normalize_l2(query_embeddings)
        
        # Converte para float32
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Busca
        distances, indices = self.index.search(query_embeddings, k)
        
        if squeeze_output:
            distances = distances[0]
            indices = indices[0]
        
        return distances, indices
    
    def get_metadata(self, indices: np.ndarray) -> List[Dict[str, Any]]:
        """
        Obtém metadados para índices.
        
        Args:
            indices: Índices dos resultados
        
        Returns:
            Lista de metadados
        """
        if indices.ndim == 1:
            return [self.id_to_metadata.get(int(idx), {}) for idx in indices]
        else:
            return [[self.id_to_metadata.get(int(idx), {}) for idx in row] for row in indices]
    
    def get_size(self) -> int:
        """Retorna número de embeddings indexados."""
        return self.index.ntotal
    
    def reset(self):
        """Reseta o índice."""
        self.index = self._create_index()
        self.id_to_metadata = {}
        self.next_id = 0

