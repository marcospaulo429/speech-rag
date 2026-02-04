"""DataLoader para pares (áudio, transcrição)."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union
import os


class AudioTextDataset(Dataset):
    """Dataset para pares (áudio, transcrição)."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        audio_dir: Optional[str] = None,
        augmentation: Optional[Any] = None
    ):
        """
        Inicializa dataset.
        
        Args:
            data: Lista de dicionários com 'audio' e 'text' (ou 'transcription')
            audio_dir: Diretório base para arquivos de áudio (se 'audio' for caminho)
            augmentation: Objeto de augmentation opcional
        """
        self.data = data
        self.audio_dir = audio_dir
        self.augmentation = augmentation
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retorna item do dataset.
        
        Returns:
            Dicionário com 'audio' e 'text'
        """
        item = self.data[idx]
        
        # Carrega áudio
        audio = item.get('audio')
        if isinstance(audio, str):
            # Caminho para arquivo
            if self.audio_dir:
                audio_path = os.path.join(self.audio_dir, audio)
            else:
                audio_path = audio
            
            import torchaudio
            audio, _ = torchaudio.load(audio_path)
            if audio.dim() > 1:
                audio = audio.mean(dim=0)  # Mono
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        
        # Aplica augmentation se disponível
        if self.augmentation is not None:
            audio = self.augmentation.apply(audio)
        
        # Texto
        text = item.get('text') or item.get('transcription', '')
        
        return {
            'audio': audio,
            'text': text
        }


def create_dataloader(
    data: List[Dict[str, Any]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    audio_dir: Optional[str] = None,
    augmentation: Optional[Any] = None
) -> DataLoader:
    """
    Cria DataLoader.
    
    Args:
        data: Lista de dados
        batch_size: Tamanho do batch
        shuffle: Se True, embaralha dados
        num_workers: Número de workers
        audio_dir: Diretório de áudio
        augmentation: Augmentation opcional
    
    Returns:
        DataLoader
    """
    dataset = AudioTextDataset(data, audio_dir, augmentation)
    
    def collate_fn(batch):
        """Collate function para batch de áudios de tamanhos diferentes."""
        audios = [item['audio'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Padding de áudios
        max_len = max(len(a) for a in audios)
        padded_audios = []
        for audio in audios:
            if len(audio) < max_len:
                padding = torch.zeros(max_len - len(audio))
                audio = torch.cat([audio, padding])
            padded_audios.append(audio)
        
        return {
            'audio': torch.stack(padded_audios),
            'text': texts
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

