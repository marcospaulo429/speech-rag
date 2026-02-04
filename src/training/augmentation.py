"""Data augmentation para áudio."""

import torch
import torchaudio
import numpy as np
from typing import Optional


class AudioAugmentation:
    """Augmentações de áudio para treinamento."""
    
    def __init__(
        self,
        noise_injection: bool = False,
        speed_variation: bool = False,
        noise_level: float = 0.01,
        speed_range: tuple = (0.9, 1.1)
    ):
        """
        Inicializa augmentações.
        
        Args:
            noise_injection: Se True, injeta ruído
            speed_variation: Se True, varia velocidade
            noise_level: Nível de ruído (std)
            speed_range: Range de variação de velocidade (min, max)
        """
        self.noise_injection = noise_injection
        self.speed_variation = speed_variation
        self.noise_level = noise_level
        self.speed_range = speed_range
    
    def inject_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Injeta ruído gaussiano no áudio."""
        noise = torch.randn_like(audio) * self.noise_level
        return audio + noise
    
    def vary_speed(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Varia velocidade do áudio."""
        speed = np.random.uniform(self.speed_range[0], self.speed_range[1])
        
        # Usa time_stretch do torchaudio
        try:
            effects = [["speed", str(speed)]]
            stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio.unsqueeze(0) if audio.dim() == 1 else audio,
                sample_rate,
                effects
            )
            return stretched.squeeze(0) if stretched.dim() > 1 else stretched
        except:
            # Fallback simples: resample
            new_length = int(len(audio) / speed)
            indices = torch.linspace(0, len(audio) - 1, new_length).long()
            return audio[indices]
    
    def apply(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Aplica augmentações ao áudio.
        
        Args:
            audio: Áudio para augmentar
            sample_rate: Taxa de amostragem
        
        Returns:
            Áudio augmentado
        """
        augmented = audio.clone()
        
        if self.noise_injection:
            augmented = self.inject_noise(augmented)
        
        if self.speed_variation:
            augmented = self.vary_speed(augmented, sample_rate)
        
        return augmented

