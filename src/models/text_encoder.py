"""Text Encoder using E5-Mistral-7B-Instruct"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Union


class TextEncoder(nn.Module):
    """
    Text encoder wrapper for E5-Mistral-7B-Instruct.
    Generates normalized text embeddings.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct", freeze: bool = True):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze model parameters
        """
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze all parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
    
    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        normalize: bool = True,
        device: str = None
    ) -> torch.Tensor:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text string or list of texts
            max_length: Maximum sequence length
            normalize: Whether to L2-normalize embeddings
            device: Device to run on (auto-detect if None)
        
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if device is None:
            device = next(self.model.parameters()).device
        
        # Tokenize with E5 format
        # E5 models expect input format: "query: {text}" or "passage: {text}"
        formatted_texts = [f"query: {text}" for text in texts]
        
        encoded = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**encoded)
            # Use mean pooling over sequence length
            # Get attention mask
            attention_mask = encoded["attention_mask"]
            # Sum embeddings over sequence dimension
            embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            # Average over sequence length (excluding padding)
            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # L2 normalize
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Forward pass for compatibility with nn.Module"""
        return self.encode(texts, **kwargs)

