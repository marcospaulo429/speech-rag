"""Loss de distilação para treinamento do adaptador."""

import torch
import torch.nn as nn
from typing import Optional


class DistillationLoss(nn.Module):
    """Loss de distilação para alinhar embeddings de fala com embeddings de texto."""
    
    def __init__(
        self,
        loss_type: str = "mse",
        temperature: float = 1.0,
        alpha: float = 1.0
    ):
        """
        Inicializa loss de distilação.
        
        Args:
            loss_type: Tipo de loss ("mse", "cosine", "mse+cosine")
            temperature: Temperatura para distilação (apenas para cosine)
            alpha: Peso para combinação de losses (apenas para mse+cosine)
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.temperature = temperature
        self.alpha = alpha
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "cosine":
            self.criterion = nn.CosineEmbeddingLoss()
        elif loss_type == "mse+cosine":
            self.mse_criterion = nn.MSELoss()
            self.cosine_criterion = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Tipo de loss desconhecido: {loss_type}")
    
    def forward(
        self,
        speech_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Calcula loss de distilação.
        
        Args:
            speech_embeddings: Embeddings de fala projetados (B, D)
            text_embeddings: Embeddings de texto (B, D)
            normalize: Se True, normaliza embeddings antes de calcular loss
        
        Returns:
            Loss de distilação
        """
        if normalize:
            from ..utils.normalization import normalize_l2
            speech_embeddings = normalize_l2(speech_embeddings)
            text_embeddings = normalize_l2(text_embeddings)
        
        if self.loss_type == "mse":
            return self.criterion(speech_embeddings, text_embeddings)
        
        elif self.loss_type == "cosine":
            # CosineEmbeddingLoss requer targets (1 para similar, -1 para dissimilar)
            targets = torch.ones(speech_embeddings.size(0), device=speech_embeddings.device)
            return self.criterion(speech_embeddings, text_embeddings, targets)
        
        elif self.loss_type == "mse+cosine":
            # Combina MSE e Cosine
            mse_loss = self.mse_criterion(speech_embeddings, text_embeddings)
            
            targets = torch.ones(speech_embeddings.size(0), device=speech_embeddings.device)
            cosine_loss = self.cosine_criterion(speech_embeddings, text_embeddings, targets)
            
            return self.alpha * mse_loss + (1 - self.alpha) * cosine_loss
        
        else:
            raise ValueError(f"Tipo de loss desconhecido: {self.loss_type}")


def cosine_similarity_loss(
    speech_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Calcula loss baseado em similaridade de cosseno (1 - cosine_similarity).
    
    Args:
        speech_embeddings: Embeddings de fala projetados (B, D)
        text_embeddings: Embeddings de texto (B, D)
        normalize: Se True, normaliza embeddings
    
    Returns:
        Loss (1 - média de cosine similarity)
    """
    if normalize:
        from ..utils.normalization import normalize_l2
        speech_embeddings = normalize_l2(speech_embeddings)
        text_embeddings = normalize_l2(text_embeddings)
    
    # Cosine similarity
    cosine_sim = (speech_embeddings * text_embeddings).sum(dim=1)  # (B,)
    
    # Loss = 1 - similarity (queremos maximizar similarity)
    loss = 1.0 - cosine_sim.mean()
    
    return loss

