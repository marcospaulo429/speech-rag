"""Text Embedder usando modelos pré-treinados do Hugging Face."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import Union, List, Optional
import yaml


class TextEmbedder(nn.Module):
    """Embedder de texto usando modelos LLM-based."""
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        normalize: bool = True,
        device: Optional[str] = None,
        freeze: bool = True,
        use_sentence_transformers: bool = False
    ):
        """
        Inicializa o text embedder.
        
        Args:
            model_name: Nome do modelo no Hugging Face
            normalize: Se True, normaliza embeddings
            device: Device para processamento
            freeze: Se True, congela os parâmetros do modelo
            use_sentence_transformers: Se True, usa SentenceTransformers
        """
        super().__init__()
        
        self.model_name = model_name
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sentence_transformers = use_sentence_transformers
        
        # Carrega modelo
        if use_sentence_transformers:
            self.model = SentenceTransformer(model_name)
            self.model = self.model.to(self.device)
            self.tokenizer = None
            # Dimensão das features
            self.feature_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            # Dimensão das features
            self.feature_dim = self.model.config.hidden_size
        
        # Congela parâmetros se solicitado
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    @classmethod
    def from_config(cls, config_path: str, device: Optional[str] = None):
        """
        Cria embedder a partir de arquivo de configuração.
        
        Args:
            config_path: Caminho para arquivo YAML de configuração
            device: Device para processamento
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            model_name=config.get("model_name", "intfloat/e5-base-v2"),
            normalize=config.get("normalize", True),
            device=device,
            freeze=True,
            use_sentence_transformers="sentence-transformers" in config.get("model_name", "")
        )
    
    def forward(
        self,
        texts: Union[str, List[str]],
        return_pooled: bool = True
    ) -> torch.Tensor:
        """
        Extrai embeddings de texto.
        
        Args:
            texts: Texto ou lista de textos
            return_pooled: Se True, retorna embeddings agregados
        
        Returns:
            Embeddings de texto (B, D) ou (B, T, D)
        """
        self.model.eval()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_sentence_transformers:
            # Usa SentenceTransformers
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize,
                    device=self.device
                )
            return embeddings
        else:
            # Usa transformers padrão
            # Tokeniza
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Extrai embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                if return_pooled:
                    # Mean pooling sobre tokens
                    attention_mask = encoded["attention_mask"]
                    embeddings = outputs.last_hidden_state
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # Normaliza se solicitado
                    if self.normalize:
                        from ..utils.normalization import normalize_l2
                        embeddings = normalize_l2(embeddings)
                    
                    return embeddings
                else:
                    return outputs.last_hidden_state
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Método conveniente para encoding.
        
        Args:
            texts: Texto ou lista de textos
            normalize: Se True, normaliza embeddings (sobrescreve self.normalize)
        
        Returns:
            Embeddings de texto
        """
        if normalize is not None:
            original_normalize = self.normalize
            self.normalize = normalize
            embeddings = self.forward(texts, return_pooled=True)
            self.normalize = original_normalize
        else:
            embeddings = self.forward(texts, return_pooled=True)
        
        return embeddings
    
    def get_feature_dim(self) -> int:
        """Retorna dimensão das features."""
        return self.feature_dim

