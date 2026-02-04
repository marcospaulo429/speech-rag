"""Avaliação durante treinamento."""

import torch
from typing import Dict, Any
from ..utils.normalization import normalize_l2, cosine_similarity


class AdapterEvaluator:
    """Avalia adaptador durante treinamento."""
    
    def __init__(self, normalize: bool = True):
        """
        Inicializa evaluator.
        
        Args:
            normalize: Se True, normaliza embeddings antes de calcular métricas
        """
        self.normalize = normalize
    
    def evaluate(
        self,
        speech_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        projected_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Avalia alinhamento de embeddings.
        
        Args:
            speech_embeddings: Embeddings de fala originais
            text_embeddings: Embeddings de texto (ground truth)
            projected_embeddings: Embeddings de fala projetados
        
        Returns:
            Dicionário com métricas
        """
        if self.normalize:
            text_embeddings = normalize_l2(text_embeddings)
            projected_embeddings = normalize_l2(projected_embeddings)
        
        # Cosine similarity média
        cosine_sim = (projected_embeddings * text_embeddings).sum(dim=1).mean().item()
        
        # MSE
        mse = torch.nn.functional.mse_loss(projected_embeddings, text_embeddings).item()
        
        # L2 distance média
        l2_dist = torch.nn.functional.pairwise_distance(
            projected_embeddings, text_embeddings
        ).mean().item()
        
        return {
            'cosine_similarity': cosine_sim,
            'mse': mse,
            'l2_distance': l2_dist
        }

