"""Speech Encoder using HuBERT-large"""

import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, Wav2Vec2Processor
from typing import Union, List
import numpy as np


class SpeechEncoder(nn.Module):
    """
    Speech encoder wrapper for HuBERT-large.
    Processes raw audio at 16kHz to generate deep speech representations.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960",
        freeze: bool = True,
        target_sample_rate: int = 16000
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze model parameters
            target_sample_rate: Target audio sample rate (16kHz)
        """
        super().__init__()
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        
        # Load HuBERT model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        
        # Freeze all parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get output dimension
        self.hidden_size = self.model.config.hidden_size  # 1024 for HuBERT-large
    
    def preprocess_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, str],
        sample_rate: int = None
    ) -> torch.Tensor:
        """
        Preprocess audio to 16kHz mono format.
        
        Args:
            audio: Audio tensor/array or path to audio file
            sample_rate: Original sample rate (if audio is tensor/array)
        
        Returns:
            Preprocessed audio tensor at 16kHz
        """
        # Load audio if path provided
        if isinstance(audio, str):
            waveform, sr = torchaudio.load(audio)
            sample_rate = sr
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = audio
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to target sample rate if needed
        if sample_rate is not None and sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                self.target_sample_rate
            )
            waveform = resampler(waveform)
        
        # Ensure correct shape: (1, samples)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        return waveform.squeeze(0)  # Return (samples,)
    
    def encode(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, List],
        normalize: bool = False,
        device: str = None
    ) -> torch.Tensor:
        """
        Encode audio into speech representations.
        
        Args:
            audio: Audio input(s) - tensor, array, file path, or list
            normalize: Whether to normalize output (usually not needed for adapter input)
            device: Device to run on (auto-detect if None)
        
        Returns:
            Hidden states tensor of shape (batch_size, seq_len, hidden_size)
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Handle batch input
        if isinstance(audio, list):
            # Preprocess each audio
            waveforms = []
            for a in audio:
                wav = self.preprocess_audio(a)
                waveforms.append(wav)
            
            # Pad to same length
            max_len = max(w.shape[0] for w in waveforms)
            padded = []
            for w in waveforms:
                pad_len = max_len - w.shape[0]
                padded_w = torch.nn.functional.pad(w, (0, pad_len))
                padded.append(padded_w)
            waveform = torch.stack(padded).to(device)
        else:
            waveform = self.preprocess_audio(audio).unsqueeze(0).to(device)
        
        # Process with HuBERT processor
        # Ensure waveform is numpy array for processor
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform
        
        # Handle batch dimension
        if len(waveform_np.shape) == 1:
            waveform_np = waveform_np.reshape(1, -1)
        
        inputs = self.processor(
            waveform_np,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Encode
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**inputs)
            # Get last hidden state: (batch_size, seq_len, hidden_size)
            hidden_states = outputs.last_hidden_state
        
        if normalize:
            hidden_states = nn.functional.normalize(hidden_states, p=2, dim=-1)
        
        return hidden_states
    
    def forward(self, audio: Union[torch.Tensor, np.ndarray, str, List], **kwargs) -> torch.Tensor:
        """Forward pass for compatibility with nn.Module"""
        return self.encode(audio, **kwargs)

