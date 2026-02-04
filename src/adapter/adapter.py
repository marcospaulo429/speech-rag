"""Speech Adapter: Adaptador leve que projeta embeddings de fala no espaço de texto."""

import torch
import torch.nn as nn
from typing import Optional
import yaml


class MLPAdapter(nn.Module):
    """Adaptador MLP para projeção de embeddings."""
    
    def __init__(
        self,
        speech_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        """
        Inicializa adaptador MLP.
        
        Args:
            speech_dim: Dimensão das features de fala
            text_dim: Dimensão das features de texto
            hidden_dim: Dimensão da camada escondida
            num_layers: Número de camadas
            activation: Função de ativação ("relu", "gelu", "tanh")
            dropout: Taxa de dropout
        """
        super().__init__()
        
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Função de ativação
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Ativação desconhecida: {activation}")
        
        # Constroi camadas
        layers = []
        
        # Primeira camada
        layers.append(nn.Linear(speech_dim, hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Camadas intermediárias
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        
        # Última camada
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, text_dim))
        else:
            layers.append(nn.Linear(speech_dim, text_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Projeta embeddings de fala no espaço de texto.
        
        Args:
            speech_embeddings: Embeddings de fala (B, speech_dim) ou (speech_dim,)
        
        Returns:
            Embeddings projetados (B, text_dim) ou (text_dim,)
        """
        return self.layers(speech_embeddings)


class TransformerAdapter(nn.Module):
    """Adaptador Transformer leve para projeção de embeddings."""
    
    def __init__(
        self,
        speech_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Inicializa adaptador Transformer.
        
        Args:
            speech_dim: Dimensão das features de fala
            text_dim: Dimensão das features de texto
            hidden_dim: Dimensão da camada escondida
            num_layers: Número de camadas Transformer
            num_heads: Número de heads de atenção
            dropout: Taxa de dropout
        """
        super().__init__()
        
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projeção de entrada
        self.input_projection = nn.Linear(speech_dim, hidden_dim)
        
        # Camadas Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projeção de saída
        self.output_projection = nn.Linear(hidden_dim, text_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Projeta embeddings de fala no espaço de texto.
        
        Args:
            speech_embeddings: Embeddings de fala (B, speech_dim) ou (speech_dim,)
        
        Returns:
            Embeddings projetados (B, text_dim) ou (text_dim,)
        """
        # Adiciona dimensão de sequência se necessário
        if speech_embeddings.dim() == 1:
            speech_embeddings = speech_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, speech_dim)
            squeeze_output = True
        elif speech_embeddings.dim() == 2:
            speech_embeddings = speech_embeddings.unsqueeze(1)  # (B, 1, speech_dim)
            squeeze_output = False
        else:
            squeeze_output = False
        
        # Projeção de entrada
        x = self.input_projection(speech_embeddings)  # (B, 1, hidden_dim)
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)  # (B, 1, hidden_dim)
        
        # Projeção de saída
        x = self.output_projection(x)  # (B, 1, text_dim)
        x = x.squeeze(1)  # (B, text_dim)
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x


class SpeechAdapter(nn.Module):
    """Wrapper para adaptador de fala."""
    
    def __init__(
        self,
        architecture: str = "mlp",
        speech_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.1,
        num_heads: Optional[int] = None
    ):
        """
        Inicializa speech adapter.
        
        Args:
            architecture: Arquitetura ("mlp" ou "transformer")
            speech_dim: Dimensão das features de fala
            text_dim: Dimensão das features de texto
            hidden_dim: Dimensão da camada escondida
            num_layers: Número de camadas
            activation: Função de ativação (apenas para MLP)
            dropout: Taxa de dropout
            num_heads: Número de heads de atenção (apenas para Transformer)
        """
        super().__init__()
        
        if architecture == "mlp":
            self.adapter = MLPAdapter(
                speech_dim=speech_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
                dropout=dropout
            )
        elif architecture == "transformer":
            if num_heads is None:
                num_heads = hidden_dim // 64  # Heurística
            self.adapter = TransformerAdapter(
                speech_dim=speech_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Arquitetura desconhecida: {architecture}")
        
        self.architecture = architecture
        self.speech_dim = speech_dim
        self.text_dim = text_dim
    
    @classmethod
    def from_config(cls, config_path: str):
        """
        Cria adapter a partir de arquivo de configuração.
        
        Args:
            config_path: Caminho para arquivo YAML de configuração
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            architecture=config.get("architecture", "mlp"),
            speech_dim=config.get("speech_dim", 768),
            text_dim=config.get("text_dim", 768),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.1)
        )
    
    def forward(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Projeta embeddings de fala no espaço de texto.
        
        Args:
            speech_embeddings: Embeddings de fala
        
        Returns:
            Embeddings projetados
        """
        return self.adapter(speech_embeddings)

