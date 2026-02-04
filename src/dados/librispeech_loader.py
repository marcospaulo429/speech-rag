"""Carregador de LibriSpeech do Hugging Face."""

from datasets import load_dataset
from typing import List, Dict, Any, Optional
import torch
import torchaudio
import json
from pathlib import Path


def load_librispeech_split(
    split: str = "train-clean-100",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Carrega split do LibriSpeech do Hugging Face.
    
    Args:
        split: Nome do split (train-clean-100, train-clean-360, dev-clean, test-clean)
        limit: Limitar número de amostras (None = todas)
        cache_dir: Diretório para cache do dataset
    
    Returns:
        Lista de dicionários com 'audio' (tensor) e 'text' (transcrição)
    """
    print(f"Carregando LibriSpeech split: {split}...")
    
    # Carrega dataset do Hugging Face
    dataset = load_dataset(
        "librispeech_asr",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Limita se solicitado
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    # Converte para formato esperado
    data = []
    for item in dataset:
        # Áudio já vem como array numpy do datasets
        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]
        
        # Converte para tensor
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
        # Resample para 16kHz se necessário
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        
        # Texto (transcrição)
        text = item["text"].strip()
        
        data.append({
            "audio": audio_tensor,
            "text": text
        })
    
    print(f"Carregadas {len(data)} amostras do split {split}")
    return data


def load_librispeech_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Carrega dados de LibriSpeech de arquivo JSON.
    
    Args:
        json_path: Caminho para arquivo JSON
    
    Returns:
        Lista de dicionários
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Converte paths de áudio para tensores se necessário
    for item in data:
        if isinstance(item.get("audio"), str):
            # É um caminho, será carregado pelo DataLoader
            pass
    
    return data


def save_librispeech_to_json(
    data: List[Dict[str, Any]],
    json_path: str,
    save_audio_paths: bool = False
):
    """
    Salva dados do LibriSpeech em JSON.
    
    Args:
        data: Lista de dados
        json_path: Caminho para salvar
        save_audio_paths: Se True, salva paths; se False, salva apenas texto
    """
    output_data = []
    
    for item in data:
        output_item = {"text": item["text"]}
        
        if save_audio_paths and isinstance(item.get("audio"), str):
            output_item["audio"] = item["audio"]
        elif not save_audio_paths:
            # Não salva áudio, apenas texto (para referência)
            output_item["audio"] = None
        
        output_data.append(output_item)
    
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Dados salvos em {json_path}")


def get_librispeech_splits() -> List[str]:
    """Retorna lista de splits disponíveis do LibriSpeech."""
    return [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other"
    ]

