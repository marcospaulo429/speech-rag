"""Agregação de features sequenciais de áudio."""

import torch
import torch.nn as nn
from typing import Optional


class MeanPooling(nn.Module):
    """Mean pooling sobre dimensão temporal."""
    
    def __init__(self, dim: int = 1):
        """
        Args:
            dim: Dimensão sobre a qual fazer pooling (default: 1 para (B, T, D))
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica mean pooling.
        
        Args:
            features: Features sequenciais (B, T, D) ou (T, D)
            attention_mask: Máscara de atenção (B, T) ou (T,)
        
        Returns:
            Features agregadas (B, D) ou (D,)
        """
        if attention_mask is not None:
            # Mean pooling com máscara
            mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
            sum_features = torch.sum(features * mask_expanded, dim=self.dim)
            sum_mask = torch.clamp(mask_expanded.sum(dim=self.dim), min=1e-9)
            return sum_features / sum_mask
        else:
            # Mean pooling simples
            return torch.mean(features, dim=self.dim)


class MaxPooling(nn.Module):
    """Max pooling sobre dimensão temporal."""
    
    def __init__(self, dim: int = 1):
        """
        Args:
            dim: Dimensão sobre a qual fazer pooling (default: 1 para (B, T, D))
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica max pooling.
        
        Args:
            features: Features sequenciais (B, T, D) ou (T, D)
            attention_mask: Máscara de atenção (B, T) ou (T,)
        
        Returns:
            Features agregadas (B, D) ou (D,)
        """
        if attention_mask is not None:
            # Max pooling com máscara (substitui valores mascarados por -inf)
            mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
            features = features.masked_fill(mask_expanded == 0, float('-inf'))
        
        return torch.max(features, dim=self.dim)[0]


class AttentionPooling(nn.Module):
    """Attention pooling com mecanismo de atenção aprendido."""
    
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Dimensão das features de entrada
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica attention pooling.
        
        Args:
            features: Features sequenciais (B, T, D) ou (T, D)
            attention_mask: Máscara de atenção (B, T) ou (T,)
        
        Returns:
            Features agregadas (B, D) ou (D,)
        """
        # Adiciona dimensão de batch se necessário
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Calcula scores de atenção
        attention_scores = self.attention(features)  # (B, T, 1)
        
        # Aplica máscara se fornecida
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T, 1)
        
        # Weighted sum
        pooled = torch.sum(attention_weights * features, dim=1)  # (B, D)
        
        if squeeze_output:
            pooled = pooled.squeeze(0)
        
        return pooled


def get_pooling_layer(method: str, feature_dim: Optional[int] = None) -> nn.Module:
    """
    Factory function para obter camada de pooling.
    
    Args:
        method: Método de pooling ("mean", "max", "attention")
        feature_dim: Dimensão das features (necessário apenas para attention)
    
    Returns:
        Módulo de pooling
    """
    if method == "mean":
        return MeanPooling()
    elif method == "max":
        return MaxPooling()
    elif method == "attention":
        if feature_dim is None:
            raise ValueError("feature_dim é necessário para attention pooling")
        return AttentionPooling(feature_dim)
    else:
        raise ValueError(f"Método de pooling desconhecido: {method}")

