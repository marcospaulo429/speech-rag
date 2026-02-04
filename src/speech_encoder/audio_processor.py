"""Pré-processamento de áudio para speech encoder."""

import torch
import torchaudio
from typing import Union, Optional


class AudioProcessor:
    """Classe para pré-processar áudio antes do encoding."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        device: Optional[str] = None
    ):
        """
        Inicializa o processador de áudio.
        
        Args:
            sample_rate: Taxa de amostragem desejada
            normalize: Se True, normaliza o áudio
            device: Device para processamento (cuda/cpu)
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def process(
        self,
        audio: Union[torch.Tensor, str],
        return_tensors: str = "pt"
    ) -> torch.Tensor:
        """
        Processa áudio para o formato esperado pelo encoder.
        
        Args:
            audio: Áudio como tensor ou caminho para arquivo
            return_tensors: Formato de retorno ("pt" para PyTorch)
        
        Returns:
            Tensor de áudio processado (1, T) ou (T,)
        """
        # Carrega áudio se for caminho
        if isinstance(audio, str):
            waveform, orig_sr = torchaudio.load(audio)
        else:
            waveform = audio
            orig_sr = self.sample_rate
        
        # Move para device
        waveform = waveform.to(self.device)
        
        # Resample se necessário
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform)
        
        # Converte para mono se estéreo
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normaliza se solicitado
        if self.normalize:
            waveform = self._normalize(waveform)
        
        # Remove dimensão extra se necessário
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        
        return waveform
    
    def process_batch(
        self,
        audio_list: list,
        pad_to_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Processa um batch de áudios.
        
        Args:
            audio_list: Lista de áudios (tensores ou caminhos)
            pad_to_length: Comprimento para padding (None = max length)
        
        Returns:
            Tensor batch de áudios processados (B, T)
        """
        processed = []
        lengths = []
        
        for audio in audio_list:
            processed_audio = self.process(audio)
            processed.append(processed_audio)
            lengths.append(len(processed_audio))
        
        # Determina comprimento máximo
        max_length = pad_to_length or max(lengths)
        
        # Padding
        batch = []
        for audio in processed:
            if len(audio) < max_length:
                padding = torch.zeros(max_length - len(audio), device=self.device)
                audio = torch.cat([audio, padding])
            elif len(audio) > max_length:
                audio = audio[:max_length]
            batch.append(audio)
        
        return torch.stack(batch)
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normaliza áudio para [-1, 1]."""
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

