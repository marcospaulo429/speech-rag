"""Chunking de áudio longo em segmentos menores."""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Representa um chunk de áudio."""
    audio: Union[torch.Tensor, np.ndarray]
    start_time: float
    end_time: float
    chunk_id: int
    metadata: Optional[dict] = None


class AudioChunker:
    """Classe para segmentar áudios longos em chunks."""
    
    def __init__(
        self,
        chunk_duration: float = 10.0,
        overlap: float = 2.0,
        sample_rate: int = 16000
    ):
        """
        Inicializa o chunker de áudio.
        
        Args:
            chunk_duration: Duração de cada chunk em segundos
            overlap: Overlap entre chunks em segundos
            sample_rate: Taxa de amostragem do áudio
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples
    
    def chunk_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        metadata: Optional[dict] = None
    ) -> List[AudioChunk]:
        """
        Segmenta áudio em chunks com overlap.
        
        Args:
            audio: Áudio a ser segmentado (1D tensor ou array)
            metadata: Metadados opcionais para cada chunk
        
        Returns:
            Lista de AudioChunk
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        audio_length = len(audio)
        chunks = []
        
        # Se o áudio é menor que um chunk, retorna o áudio completo
        if audio_length <= self.chunk_samples:
            chunk = AudioChunk(
                audio=audio,
                start_time=0.0,
                end_time=audio_length / self.sample_rate,
                chunk_id=0,
                metadata=metadata
            )
            return [chunk]
        
        # Segmenta o áudio
        start_idx = 0
        chunk_id = 0
        
        while start_idx < audio_length:
            end_idx = min(start_idx + self.chunk_samples, audio_length)
            chunk_audio = audio[start_idx:end_idx]
            
            # Padding se necessário (último chunk)
            if len(chunk_audio) < self.chunk_samples:
                if isinstance(audio, np.ndarray):
                    padding = np.zeros(self.chunk_samples - len(chunk_audio))
                    chunk_audio = np.concatenate([chunk_audio, padding])
                else:
                    padding = torch.zeros(self.chunk_samples - len(chunk_audio))
                    chunk_audio = torch.cat([chunk_audio, padding])
            
            start_time = start_idx / self.sample_rate
            end_time = end_idx / self.sample_rate
            
            chunk = AudioChunk(
                audio=chunk_audio,
                start_time=start_time,
                end_time=end_time,
                chunk_id=chunk_id,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # Move para o próximo chunk
            start_idx += self.step_samples
            chunk_id += 1
        
        return chunks
    
    def reconstruct_audio(
        self,
        chunks: List[AudioChunk],
        method: str = "overlap_average"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Reconstrói áudio a partir de chunks.
        
        Args:
            chunks: Lista de AudioChunk
            method: Método de reconstrução ("overlap_average" ou "concatenate")
        
        Returns:
            Áudio reconstruído
        """
        if not chunks:
            raise ValueError("Lista de chunks vazia")
        
        if method == "concatenate":
            # Simplesmente concatena (pode ter duplicação no overlap)
            audio_parts = [chunk.audio[:int((chunk.end_time - chunk.start_time) * self.sample_rate)] 
                          for chunk in chunks]
            if isinstance(chunks[0].audio, torch.Tensor):
                return torch.cat(audio_parts)
            else:
                return np.concatenate(audio_parts)
        
        elif method == "overlap_average":
            # Média ponderada nas regiões de overlap
            total_samples = int(chunks[-1].end_time * self.sample_rate)
            
            if isinstance(chunks[0].audio, torch.Tensor):
                reconstructed = torch.zeros(total_samples)
                weights = torch.zeros(total_samples)
            else:
                reconstructed = np.zeros(total_samples)
                weights = np.zeros(total_samples)
            
            for chunk in chunks:
                start_idx = int(chunk.start_time * self.sample_rate)
                end_idx = int(chunk.end_time * self.sample_rate)
                chunk_samples = end_idx - start_idx
                
                chunk_audio = chunk.audio[:chunk_samples]
                
                if isinstance(chunk_audio, torch.Tensor):
                    reconstructed[start_idx:end_idx] += chunk_audio
                    weights[start_idx:end_idx] += 1.0
                else:
                    reconstructed[start_idx:end_idx] += chunk_audio
                    weights[start_idx:end_idx] += 1.0
            
            # Normaliza pela média
            weights = np.clip(weights, a_min=1e-8, a_max=None)
            if isinstance(reconstructed, torch.Tensor):
                weights = torch.from_numpy(weights)
                reconstructed = reconstructed / weights
            else:
                reconstructed = reconstructed / weights
            
            return reconstructed
        
        else:
            raise ValueError(f"Método desconhecido: {method}")

