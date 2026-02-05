"""Loss functions for distillation training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class DistillationLoss(nn.Module):
    """
    Distillation loss for aligning speech embeddings to text embeddings.
    
    Compares audio embedding (from adapter) with text embedding (ground truth).
    """
    
    def __init__(
        self,
        loss_type: Literal["mse", "cosine", "both"] = "mse",
        temperature: float = 1.0,
        alpha: float = 0.5
    ):
        """
        Args:
            loss_type: Type of loss - "mse", "cosine", or "both"
            temperature: Temperature for soft distillation (if used)
            alpha: Weight for cosine loss when loss_type="both"
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.alpha = alpha
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            audio_embeddings: Embeddings from speech adapter
                Shape: (batch_size, embedding_dim)
            text_embeddings: Target embeddings from text encoder
                Shape: (batch_size, embedding_dim)
        
        Returns:
            Loss scalar
        """
        if self.loss_type == "mse":
            # Mean Squared Error loss
            loss = self.mse_loss(audio_embeddings, text_embeddings)
        
        elif self.loss_type == "cosine":
            # Cosine similarity loss (1 - cosine_similarity)
            # Since embeddings are normalized, this is equivalent to MSE on normalized vectors
            cosine_sim = F.cosine_similarity(audio_embeddings, text_embeddings, dim=1)
            loss = (1 - cosine_sim).mean()
        
        elif self.loss_type == "both":
            # Combined MSE and cosine loss
            mse = self.mse_loss(audio_embeddings, text_embeddings)
            cosine_sim = F.cosine_similarity(audio_embeddings, text_embeddings, dim=1)
            cosine = (1 - cosine_sim).mean()
            loss = (1 - self.alpha) * mse + self.alpha * cosine
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def compute_similarity(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings (for evaluation).
        
        Args:
            audio_embeddings: Embeddings from speech adapter
            text_embeddings: Embeddings from text encoder
        
        Returns:
            Cosine similarity scores (batch_size,)
        """
        return F.cosine_similarity(audio_embeddings, text_embeddings, dim=1)

