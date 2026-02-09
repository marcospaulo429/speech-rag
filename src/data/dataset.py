"""Dataset loader for Spoken Squad and other speech datasets"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset as HFDataset
from .preprocessing import AudioPreprocessor
import numpy as np


class SpeechDataset(Dataset):
    """
    Dataset for speech-text pairs.
    Supports Spoken Squad and other HuggingFace datasets.
    """
    
    def __init__(
        self,
        dataset_name: str = "spoken_squad_test",
        dataset_config: Optional[str] = None,
        split: str = "train",
        audio_column: str = "audio",
        text_column: str = "passage_text",
        sample_rate: int = 16000,
        max_audio_length: float = 60.0,
        cache_dir: Optional[str] = None,
        streaming: bool = False
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (default: "spoken_squad_test")
            dataset_config: Dataset configuration (None for Spoken Squad)
            split: Dataset split (train, validation, test)
            audio_column: Column name for audio data (default: "audio")
            text_column: Column name for text transcriptions (default: "passage_text")
            sample_rate: Target sample rate
            max_audio_length: Maximum audio length in seconds
            cache_dir: Cache directory for datasets
            streaming: Whether to use streaming mode
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.audio_column = audio_column
        self.text_column = text_column
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            max_length_seconds=max_audio_length
        )
        
        # Load dataset
        try:
            if dataset_config:
                self.dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split,
                    cache_dir=cache_dir,
                    streaming=streaming
                )
            else:
                # For Spoken Squad and datasets without config
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=cache_dir,
                    streaming=streaming
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset {dataset_name} with config {dataset_config}: {e}\n"
                f"Make sure the dataset is available on HuggingFace or provide a custom loader."
            )
        
        # Convert to list if not streaming (for length calculation)
        if not streaming:
            self.dataset = list(self.dataset)
    
    def __len__(self) -> int:
        """Get dataset length"""
        if isinstance(self.dataset, list):
            return len(self.dataset)
        elif hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        else:
            # Streaming dataset - return a large number
            return 1000000
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with 'audio' (tensor) and 'text' (string)
        """
        # Get sample
        if isinstance(self.dataset, list):
            sample = self.dataset[idx]
        else:
            # For streaming or iterable datasets
            sample = list(self.dataset.skip(idx).take(1))[0]
        
        # Extract audio
        audio_data = sample[self.audio_column]
        
        # Handle different audio formats
        if isinstance(audio_data, dict):
            # HuggingFace audio format: {"array": ..., "sampling_rate": ...}
            audio_array = audio_data["array"]
            audio_sr = audio_data.get("sampling_rate", self.sample_rate)
            waveform = torch.from_numpy(audio_array).float()
        elif isinstance(audio_data, np.ndarray):
            waveform = torch.from_numpy(audio_data).float()
            audio_sr = self.sample_rate
        elif isinstance(audio_data, torch.Tensor):
            waveform = audio_data
            audio_sr = self.sample_rate
        else:
            # Assume it's a file path
            waveform, audio_sr = self.preprocessor.load_audio(audio_data)
        
        # Preprocess audio
        waveform = self.preprocessor.process(waveform, sample_rate=audio_sr)
        
        # Extract text
        text = sample[self.text_column]
        if not isinstance(text, str):
            text = str(text)
        
        return {
            "audio": waveform,
            "text": text,
            "sample_rate": self.sample_rate
        }
    
    def get_collate_fn(self):
        """
        Get collate function for DataLoader.
        Handles variable-length audio sequences.
        """
        def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            """
            Collate function that pads audio sequences.
            
            Args:
                batch: List of samples
            
            Returns:
                Batched tensors
            """
            texts = [item["text"] for item in batch]
            audios = [item["audio"] for item in batch]
            
            # Pad audio sequences to same length
            max_len = max(audio.shape[-1] for audio in audios)
            padded_audios = []
            for audio in audios:
                pad_len = max_len - audio.shape[-1]
                padded = torch.nn.functional.pad(audio, (0, pad_len))
                padded_audios.append(padded)
            
            return {
                "audio": torch.stack(padded_audios),
                "text": texts,
                "sample_rate": torch.tensor([item["sample_rate"] for item in batch])
            }
        
        return collate_fn

