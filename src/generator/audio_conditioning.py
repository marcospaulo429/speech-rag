"""Preparação de contexto de áudio para o gerador SLM."""

import torch
from typing import Union, List, Optional
from ..utils.audio_chunking import AudioChunker


class AudioConditioner:
    """Prepara áudio para condicionamento em SLM."""
    
    def __init__(
        self,
        max_audio_length: Optional[float] = None,
        chunk_duration: Optional[float] = None,
        sample_rate: int = 16000
    ):
        """
        Inicializa o condicionador de áudio.
        
        Args:
            max_audio_length: Comprimento máximo de áudio em segundos (None = sem limite)
            chunk_duration: Duração de chunks se necessário (None = sem chunking)
            sample_rate: Taxa de amostragem
        """
        self.max_audio_length = max_audio_length
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        if chunk_duration is not None:
            self.chunker = AudioChunker(
                chunk_duration=chunk_duration,
                sample_rate=sample_rate
            )
        else:
            self.chunker = None
    
    def prepare_audio(
        self,
        audio: Union[torch.Tensor, List[torch.Tensor]],
        return_chunks: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Prepara áudio para condicionamento.
        
        Args:
            audio: Áudio ou lista de áudios
            return_chunks: Se True, retorna lista de chunks
        
        Returns:
            Áudio preparado ou lista de chunks
        """
        if isinstance(audio, list):
            # Processa cada áudio
            prepared = [self.prepare_audio(a, return_chunks) for a in audio]
            return prepared
        
        # Converte para tensor se necessário
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        
        # Limita comprimento se necessário
        if self.max_audio_length is not None:
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        
        # Chunking se necessário
        if self.chunker is not None and return_chunks:
            chunks = self.chunker.chunk_audio(audio)
            return [chunk.audio for chunk in chunks]
        
        return audio
    
    def format_for_slm(
        self,
        audio: Union[torch.Tensor, List[torch.Tensor]],
        prompt: Optional[str] = None
    ) -> dict:
        """
        Formata áudio e prompt para entrada do SLM.
        
        Args:
            audio: Áudio preparado
            prompt: Prompt de texto opcional
        
        Returns:
            Dicionário formatado para SLM
        """
        if isinstance(audio, list):
            # Múltiplos chunks
            return {
                "audio": audio,
                "prompt": prompt,
                "is_chunked": True
            }
        else:
            return {
                "audio": audio,
                "prompt": prompt,
                "is_chunked": False
            }

