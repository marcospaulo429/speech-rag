"""Speech Encoder usando modelos pré-treinados do Hugging Face."""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel, AutoProcessor
from typing import Optional, Union
import yaml
import os

from .audio_processor import AudioProcessor
from .pooling import get_pooling_layer


class SpeechEncoder(nn.Module):
    """Encoder de fala usando HuBERT ou Wav2Vec2."""
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        pooling_method: str = "mean",
        device: Optional[str] = None,
        freeze: bool = True
    ):
        """
        Inicializa o speech encoder.
        
        Args:
            model_name: Nome do modelo no Hugging Face
            pooling_method: Método de pooling ("mean", "max", "attention")
            device: Device para processamento
            freeze: Se True, congela os parâmetros do modelo
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega modelo
        if "hubert" in model_name.lower():
            self.model = HubertModel.from_pretrained(model_name)
        elif "wav2vec2" in model_name.lower():
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")
        
        # Move para device
        self.model = self.model.to(self.device)
        
        # Congela parâmetros se solicitado
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Carrega processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except:
            self.processor = None
        
        # Dimensão das features
        self.feature_dim = self.model.config.hidden_size
        
        # Pooling layer
        self.pooling = get_pooling_layer(pooling_method, self.feature_dim)
        self.pooling = self.pooling.to(self.device)
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=16000,
            normalize=True,
            device=self.device
        )
    
    @classmethod
    def from_config(cls, config_path: str, device: Optional[str] = None):
        """
        Cria encoder a partir de arquivo de configuração.
        
        Args:
            config_path: Caminho para arquivo YAML de configuração
            device: Device para processamento
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            model_name=config.get("model_name", "facebook/hubert-base-ls960"),
            pooling_method=config.get("pooling_method", "mean"),
            device=device,
            freeze=True
        )
    
    def forward(
        self,
        audio: Union[torch.Tensor, str, list],
        return_pooled: bool = True
    ) -> Union[torch.Tensor, tuple]:
        """
        Extrai features de áudio.
        
        Args:
            audio: Áudio como tensor, caminho ou lista
            return_pooled: Se True, retorna features agregadas; se False, retorna sequenciais
        
        Returns:
            Features de áudio (B, D) se pooled, (B, T, D) se não pooled
        """
        self.model.eval()
        
        # Processa áudio
        if isinstance(audio, list):
            processed_audio = self.audio_processor.process_batch(audio)
        else:
            processed_audio = self.audio_processor.process(audio)
        
        # Adiciona dimensão de batch se necessário
        if processed_audio.dim() == 1:
            processed_audio = processed_audio.unsqueeze(0)
        
        # Move para device
        processed_audio = processed_audio.to(self.device)
        
        # Extrai features
        with torch.no_grad():
            outputs = self.model(processed_audio)
            features = outputs.last_hidden_state  # (B, T, D)
        
        if return_pooled:
            # Aplica pooling
            pooled_features = self.pooling(features)
            return pooled_features
        else:
            return features
    
    def encode(
        self,
        audio: Union[torch.Tensor, str, list],
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Método conveniente para encoding com normalização opcional.
        
        Args:
            audio: Áudio para encoding
            normalize: Se True, normaliza embeddings
        
        Returns:
            Embeddings de áudio
        """
        embeddings = self.forward(audio, return_pooled=True)
        
        if normalize:
            from ..utils.normalization import normalize_l2
            embeddings = normalize_l2(embeddings)
        
        return embeddings
    
    def get_feature_dim(self) -> int:
        """Retorna dimensão das features."""
        return self.feature_dim

