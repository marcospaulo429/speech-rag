"""Audio preprocessing utilities"""

import torch
import torchaudio
import numpy as np
from typing import Union, Tuple
import librosa


class AudioPreprocessor:
    """Audio preprocessing for speech models"""
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_length_seconds: float = 30.0,
        normalize: bool = True
    ):
        """
        Args:
            target_sample_rate: Target sample rate (Hz)
            max_length_seconds: Maximum audio length in seconds
            normalize: Whether to normalize audio amplitude
        """
        self.target_sample_rate = target_sample_rate
        self.max_length_seconds = max_length_seconds
        self.max_length_samples = int(max_length_seconds * target_sample_rate)
        self.normalize = normalize
    
    def load_audio(
        self,
        audio_path: str,
        sample_rate: int = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Expected sample rate (None for auto-detect)
        
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            # Fallback to librosa if torchaudio fails
            waveform, sr = librosa.load(audio_path, sr=None, mono=False)
            waveform = torch.from_numpy(waveform).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze(0), sr
    
    def resample(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_sample_rate: int = None
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        
        Args:
            waveform: Audio waveform
            sample_rate: Original sample rate
            target_sample_rate: Target sample rate (uses self.target_sample_rate if None)
        
        Returns:
            Resampled waveform
        """
        if target_sample_rate is None:
            target_sample_rate = self.target_sample_rate
        
        if sample_rate == target_sample_rate:
            return waveform
        
        # Ensure waveform is 2D: (channels, samples)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            target_sample_rate
        )
        resampled = resampler(waveform)
        
        return resampled.squeeze(0) if resampled.shape[0] == 1 else resampled
    
    def trim_or_pad(
        self,
        waveform: torch.Tensor,
        target_length: int = None
    ) -> torch.Tensor:
        """
        Trim or pad audio to target length.
        
        Args:
            waveform: Audio waveform
            target_length: Target length in samples (uses self.max_length_samples if None)
        
        Returns:
            Trimmed/padded waveform
        """
        if target_length is None:
            target_length = self.max_length_samples
        
        current_length = waveform.shape[-1]
        
        if current_length > target_length:
            # Trim
            waveform = waveform[..., :target_length]
        elif current_length < target_length:
            # Pad with zeros
            pad_length = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return waveform
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio amplitude.
        
        Args:
            waveform: Audio waveform
        
        Returns:
            Normalized waveform
        """
        if not self.normalize:
            return waveform
        
        # Normalize to [-1, 1] range
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    
    def process(
        self,
        audio_input: Union[str, torch.Tensor, np.ndarray],
        sample_rate: int = None
    ) -> torch.Tensor:
        """
        Complete preprocessing pipeline.
        
        Args:
            audio_input: Audio file path, tensor, or numpy array
            sample_rate: Sample rate (required if audio_input is tensor/array)
        
        Returns:
            Preprocessed waveform at target sample rate
        """
        # Load if path
        if isinstance(audio_input, str):
            waveform, sr = self.load_audio(audio_input)
            sample_rate = sr
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.from_numpy(audio_input).float()
        else:
            waveform = audio_input
        
        # Ensure 1D
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # Resample
        if sample_rate is not None:
            waveform = self.resample(waveform, sample_rate)
        
        # Trim/pad
        waveform = self.trim_or_pad(waveform)
        
        # Normalize
        waveform = self.normalize_audio(waveform)
        
        return waveform

