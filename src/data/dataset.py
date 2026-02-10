"""Dataset loader for Spoken Squad and other speech datasets"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset as HFDataset, Audio
from .preprocessing import AudioPreprocessor
import numpy as np
import io
import soundfile as sf


def speech_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    texts = [item["text"] for item in batch]
    audios = [item["audio"] for item in batch]
    max_len = max(audio.shape[-1] for audio in audios)
    padded_audios = [
        torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1]))
        for audio in audios
    ]
    return {
        "audio": torch.stack(padded_audios),
        "text": texts,
        "sample_rate": torch.tensor([item["sample_rate"] for item in batch])
    }

class SpeechDataset(Dataset):
    """
    Dataset for speech-text pairs.
    Supports Spoken Squad and other HuggingFace datasets.
    """
    
    def __init__(
        self,
        dataset_name: str = "AudioLLMs/spoken_squad_test",
        dataset_config: Optional[str] = None,
        split: str = "test",
        audio_column: str = "context",
        text_column: str = "instruction",
        sample_rate: int = 16000,
        max_audio_length: float = 60.0,
        cache_dir: Optional[str] = None,
        streaming: bool = False
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (default: "AudioLLMs/spoken_squad_test")
            dataset_config: Dataset configuration (None for Spoken Squad)
            split: Dataset split (train, validation, test)
            audio_column: Column name for audio data (default: "context")
            text_column: Column name for text transcriptions (default: "instruction")
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

        if not streaming:
            features = self.dataset.features.copy()
            features[self.audio_column] = Audio(decode=False)
            self.dataset = self.dataset.cast(features)
            self.dataset = self.dataset.with_format(None)
            assert isinstance(self.dataset.features[self.audio_column], Audio)
            assert self.dataset.features[self.audio_column].decode is False
    def _decode_audio(self, audio_data) -> tuple:
        """
        Decode audio data manually without requiring torchcodec.
        
        Args:
            audio_data: Audio data in various formats (dict, bytes, array, etc.)
        
        Returns:
            Tuple of (waveform: torch.Tensor, sample_rate: int)
        """
        # If already decoded (dict format from HuggingFace)
        if isinstance(audio_data, dict):
            # Prefer bytes/path when decode=False
            if "bytes" in audio_data and audio_data["bytes"] is not None:
                return self._decode_audio(audio_data["bytes"])
            if "path" in audio_data and audio_data["path"] is not None:
                return self.preprocessor.load_audio(audio_data["path"])
            if "array" in audio_data:
                audio_array = audio_data["array"]
                audio_sr = audio_data.get("sampling_rate", self.sample_rate)
                # Ensure numeric dtype
                audio_array = np.asarray(audio_array, dtype=np.float32)
                waveform = torch.from_numpy(audio_array).float()
                return waveform, audio_sr
        
        # If it's a numpy array
        if isinstance(audio_data, np.ndarray):
            waveform = torch.from_numpy(audio_data).float()
            return waveform, self.sample_rate
        
        # If it's already a torch tensor
        if isinstance(audio_data, torch.Tensor):
            return audio_data, self.sample_rate
        
        # If it's bytes (compressed audio), decode with soundfile
        if isinstance(audio_data, bytes):
            try:
                # Decode using soundfile (doesn't require torchcodec)
                audio_array, audio_sr = sf.read(io.BytesIO(audio_data))
                waveform = torch.from_numpy(audio_array).float()
                # Convert to mono if stereo
                if len(waveform.shape) > 1:
                    waveform = torch.mean(waveform, dim=-1)
                return waveform, audio_sr
            except Exception as e:
                # Fallback to librosa
                try:
                    import librosa
                    audio_array, audio_sr = librosa.load(
                        io.BytesIO(audio_data), 
                        sr=None, 
                        mono=True
                    )
                    waveform = torch.from_numpy(audio_array).float()
                    return waveform, audio_sr
                except Exception as e2:
                    raise ValueError(
                        f"Failed to decode audio data: {e}. Tried soundfile and librosa."
                    )
        
        # If it's a file path (string)
        if isinstance(audio_data, str):
            return self.preprocessor.load_audio(audio_data)
        
        # Try to access as if it's an Audio object with bytes
        try:
            if hasattr(audio_data, 'bytes'):
                # Audio object with bytes attribute
                return self._decode_audio(audio_data.bytes)
            elif hasattr(audio_data, 'path'):
                # Audio object with path attribute
                return self.preprocessor.load_audio(audio_data.path)
        except Exception:
            pass
        
        # Last resort: try to convert to numpy and then to tensor
        try:
            audio_array = np.array(audio_data)
            waveform = torch.from_numpy(audio_array).float()
            return waveform, self.sample_rate
        except Exception as e:
            raise ValueError(
                f"Could not decode audio data of type {type(audio_data)}: {e}"
            )
    
    def __len__(self) -> int:
        """Get dataset length"""
        # Arrow dataset supports len() directly
        if hasattr(self.dataset, '__len__'):
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
        # Access dataset - should now have raw bytes instead of decoded audio
        # This avoids the torchcodec requirement
        try:
            sample = self.dataset[idx]
            audio_data = sample[self.audio_column]
            text = sample[self.text_column]
        except Exception as e:
            # If access fails, try with format disabled
            try:
                original_format = getattr(self.dataset, '_format_type', None)
                self.dataset = self.dataset.with_format(None)
                sample = self.dataset[idx]
                audio_data = sample[self.audio_column]
                text = sample[self.text_column]
            except Exception as e2:
                raise ValueError(
                    f"Failed to access dataset sample {idx}: {e}. "
                    f"Tried with format disabled: {e2}"
                )
        
        # Decode audio manually using our helper method
        # This avoids requiring torchcodec
        waveform, audio_sr = self._decode_audio(audio_data)
        
        # Preprocess audio (resample, normalize, trim)
        waveform = self.preprocessor.process(waveform, sample_rate=audio_sr)
        
        # Ensure text is string
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
        return speech_collate_fn

