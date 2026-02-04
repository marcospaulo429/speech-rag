"""Normalização L2 de embeddings para busca por similaridade."""

import torch
import numpy as np
from typing import Union


def normalize_l2(embeddings: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Normaliza embeddings usando norma L2.
    
    Args:
        embeddings: Tensor ou array de embeddings (N, D) ou (D,)
    
    Returns:
        Embeddings normalizados com norma unitária
    """
    if isinstance(embeddings, torch.Tensor):
        norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        # Evitar divisão por zero
        norm = torch.clamp(norm, min=1e-8)
        return embeddings / norm
    else:
        norm = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
        norm = np.clip(norm, a_min=1e-8, a_max=None)
        return embeddings / norm


def normalize_batch(embeddings: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Normaliza um batch de embeddings.
    
    Args:
        embeddings: Tensor ou array de embeddings (B, N, D) ou (B, D)
    
    Returns:
        Embeddings normalizados
    """
    if isinstance(embeddings, torch.Tensor):
        # Se for 2D, adiciona dimensão de batch
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        # Normaliza ao longo da última dimensão
        norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return embeddings / norm
    else:
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        norm = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
        norm = np.clip(norm, a_min=1e-8, a_max=None)
        return embeddings / norm


def cosine_similarity(
    embeddings1: Union[torch.Tensor, np.ndarray],
    embeddings2: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calcula similaridade de cosseno entre dois conjuntos de embeddings.
    Assume que os embeddings já estão normalizados.
    
    Args:
        embeddings1: Primeiro conjunto de embeddings (N, D)
        embeddings2: Segundo conjunto de embeddings (M, D)
    
    Returns:
        Matriz de similaridade (N, M)
    """
    if isinstance(embeddings1, torch.Tensor):
        return torch.matmul(embeddings1, embeddings2.t())
    else:
        return np.dot(embeddings1, embeddings2.T)

