"""Speech Adapter for aligning speech embeddings to text embedding space"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechAdapter(nn.Module):
    """
    Speech adapter that projects HuBERT representations to text embedding space.
    
    Architecture:
    - Input: HuBERT output (batch_size, seq_len, 1024)
    - Downsampling: Average pooling temporal (4x reduction)
    - Projection: Linear layer (1024 → 4096)
    - Normalization: L2 normalization
    - Output: Audio embedding (batch_size, 4096)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # HuBERT-large hidden size
        output_dim: int = 4096,  # Text encoder embedding dim (E5-Mistral: 4096, Qwen3-0.6B: 1024, Qwen3-4B: 2560)
        downsample_factor: int = 4
    ):
        """
        Args:
            input_dim: Input dimension from speech encoder (HuBERT)
            output_dim: Output dimension matching text encoder 
                       (E5-Mistral: 4096, Qwen3-Embedding-0.6B: 1024, Qwen3-Embedding-4B: 2560)
            downsample_factor: Temporal downsampling factor
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.downsample_factor = downsample_factor
        
        # Temporal downsampling via average pooling
        # We'll use adaptive pooling or manual pooling
        self.downsample = nn.AvgPool1d(
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0
        )
        
        # Projection layer
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, speech_representations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            speech_representations: HuBERT output
                Shape: (batch_size, seq_len, input_dim)
        
        Returns:
            Audio embeddings
                Shape: (batch_size, output_dim)
        """
        batch_size, seq_len, hidden_dim = speech_representations.shape
        
        # Transpose for 1D pooling: (batch, seq_len, hidden) -> (batch, hidden, seq_len)
        x = speech_representations.transpose(1, 2)  # (batch, hidden, seq_len)
        
        # Temporal downsampling: reduce seq_len by factor of 4
        x = self.downsample(x)  # (batch, hidden, seq_len/4)
        
        # Transpose back: (batch, hidden, seq_len/4) -> (batch, seq_len/4, hidden)
        x = x.transpose(1, 2)
        
        # Average over temporal dimension to get single embedding per sample
        # This gives us (batch_size, hidden_dim)
        x = x.mean(dim=1)  # (batch, hidden)
        
        # Project to output dimension
        x = self.projection(x)  # (batch, output_dim)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # L2 normalization for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension"""
        return self.output_dim

