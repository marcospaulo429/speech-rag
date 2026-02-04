"""Busca por similaridade."""

import numpy as np
import torch
from typing import Union, List, Tuple
from ..utils.normalization import normalize_l2, cosine_similarity


class SimilaritySearch:
    """Classe para busca por similaridade."""
    
    def __init__(self, metric: str = "cosine", normalize: bool = True):
        """
        Inicializa busca por similaridade.
        
        Args:
            metric: Métrica de similaridade ("cosine", "euclidean")
            normalize: Se True, normaliza embeddings
        """
        self.metric = metric
        self.normalize = normalize
    
    def search(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        candidate_embeddings: Union[torch.Tensor, np.ndarray],
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca top-k embeddings mais similares.
        
        Args:
            query_embeddings: Embeddings de consulta (M, D) ou (D,)
            candidate_embeddings: Embeddings candidatos (N, D)
            top_k: Número de resultados
        
        Returns:
            Tupla (scores, indices) onde:
            - scores: Scores de similaridade (M, k) ou (k,)
            - indices: Índices dos resultados (M, k) ou (k,)
        """
        # Converte para numpy se necessário
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()
        if isinstance(candidate_embeddings, torch.Tensor):
            candidate_embeddings = candidate_embeddings.cpu().numpy()
        
        # Adiciona dimensão se necessário
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normaliza se solicitado
        if self.normalize:
            query_embeddings = normalize_l2(query_embeddings)
            candidate_embeddings = normalize_l2(candidate_embeddings)
        
        # Calcula similaridade
        if self.metric == "cosine":
            # Cosine similarity = dot product de embeddings normalizados
            sim_matrix = np.dot(query_embeddings, candidate_embeddings.T)  # (M, N)
        elif self.metric == "euclidean":
            # Euclidean distance (menor = mais similar)
            # Calcula distâncias
            distances = np.sqrt(
                np.sum((query_embeddings[:, np.newaxis, :] - candidate_embeddings[np.newaxis, :, :]) ** 2, axis=2)
            )
            # Converte para similaridade (inverso da distância)
            sim_matrix = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Métrica desconhecida: {self.metric}")
        
        # Top-k
        top_k = min(top_k, candidate_embeddings.shape[0])
        indices = np.argsort(sim_matrix, axis=1)[:, -top_k:][:, ::-1]  # (M, k)
        scores = np.take_along_axis(sim_matrix, indices, axis=1)  # (M, k)
        
        if squeeze_output:
            scores = scores[0]
            indices = indices[0]
        
        return scores, indices

