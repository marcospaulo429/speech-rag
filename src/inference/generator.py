"""Audio-Conditioned Generator using Qwen-Audio-Chat"""

import torch
import torchaudio
from typing import List, Dict, Optional, Union
from pathlib import Path
import os
import warnings
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np

# Workaround for BeamSearchScorer - Qwen-Audio-Chat's custom code needs this
# The custom code tries to import from 'transformers' directly, but BeamSearchScorer
# is in transformers.generation.beam_search. We add it to the main namespace.
try:
    # Try direct import first (may work in some versions)
    from transformers import BeamSearchScorer
except ImportError:
    # Import from the actual location and add to transformers namespace
    from transformers.generation.beam_search import BeamSearchScorer
    # Add to transformers namespace so Qwen-Audio-Chat's custom code can find it
    transformers.BeamSearchScorer = BeamSearchScorer
    # Also add to __all__ if it exists
    if hasattr(transformers, '__all__'):
        if 'BeamSearchScorer' not in transformers.__all__:
            transformers.__all__.append('BeamSearchScorer')


class AudioConditionedGenerator:
    """
    Generator that uses Qwen-Audio-Chat (7B) to generate textual responses
    conditioned on text queries and retrieved audio passages.
    
    This model accepts native audio tokens and text in the input, enabling
    zero-shot generation without ASR transcription.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Audio-Chat",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None
    ):
        """
        Initialize the Audio-Conditioned Generator.
        
        Args:
            model_name: HuggingFace model identifier for Qwen-Audio-Chat
            device: Device to run on (auto-detect if None)
            torch_dtype: Data type for model (float16 for GPU, float32 for CPU)
            token: HuggingFace token for private models (defaults to HF_TOKEN env var)
        """
        self.model_name = model_name
        
        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Data type for model
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        # Get token from parameter, environment variable, or None
        hf_token = token or os.getenv("HF_TOKEN")
        
        print(f"Loading Qwen-Audio-Chat model: {model_name}")
        print(f"Device: {self.device}, dtype: {self.torch_dtype}")
        
        # Load processor (handles both audio and text tokenization)
        # Qwen-Audio-Chat requires trust_remote_code=True
        load_kwargs = {
            "trust_remote_code": True
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            # If AutoProcessor fails, try to get processor class from model config
            print(f"Warning: Failed to load processor with AutoProcessor: {e}")
            try:
                # Try loading config to find processor class
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, **load_kwargs)
                
                # Try to use the processor class specified in the config
                if hasattr(config, 'processor_class'):
                    processor_class_name = config.processor_class
                    print(f"Found processor class: {processor_class_name}")
                    # Import and use the processor class
                    # This is a fallback - most models should work with AutoProcessor
                    raise RuntimeError(
                        f"AutoProcessor failed. Model uses custom processor class: {processor_class_name}. "
                        f"Please check the model documentation at https://huggingface.co/{model_name}"
                    )
                else:
                    raise e
            except Exception as e2:
                print(f"Error: Could not load processor: {e2}")
                raise RuntimeError(
                    f"Failed to load processor for {model_name}. "
                    f"Please ensure the model is available and check the documentation at "
                    f"https://huggingface.co/{model_name}"
                ) from e2
        
        # Load model
        # Use separate kwargs for model to avoid conflicts with load_kwargs
        model_load_kwargs = {
            "trust_remote_code": True
        }
        if hf_token:
            model_load_kwargs["token"] = hf_token
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.torch_dtype,  # Use dtype instead of deprecated torch_dtype
                device_map="auto" if self.device == "cuda" else None,
                **model_load_kwargs
            )
        except Exception as e:
            print(f"Warning: Failed to load model with device_map: {e}")
            print("Trying without device_map...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.torch_dtype,  # Use dtype instead of deprecated torch_dtype
                **model_load_kwargs
            ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully")
    
    def prepare_audio_inputs(
        self,
        audio_paths: List[Union[str, Path]],
        target_sample_rate: int = 16000
    ) -> List[Dict]:
        """
        Prepare audio files for Qwen-Audio-Chat input.
        
        The audio is processed as native audio tokens, not converted to text.
        
        Args:
            audio_paths: List of paths to audio files
            target_sample_rate: Target sample rate for audio (Qwen-Audio typically uses 16kHz)
        
        Returns:
            List of audio input dictionaries ready for the processor
        """
        audio_inputs = []
        
        for audio_path in audio_paths:
            audio_path = str(audio_path)
            
            # Load audio file
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                # Fallback to librosa if torchaudio fails
                import librosa
                waveform, sr = librosa.load(audio_path, sr=None, mono=False)
                waveform = torch.from_numpy(waveform).float()
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                waveform = resampler(waveform)
            
            # Ensure correct shape: (1, samples) or (samples,)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            # Convert to numpy array for processor
            audio_array = waveform.squeeze(0).numpy()
            
            audio_inputs.append({
                "raw": audio_array,
                "sampling_rate": target_sample_rate
            })
        
        return audio_inputs
    
    def build_prompt(
        self,
        query: str,
        num_audios: int = 1
    ) -> str:
        """
        Build prompt for Qwen-Audio-Chat with text query and audio references.
        
        Args:
            query: Text query/question
            num_audios: Number of audio passages being provided
        
        Returns:
            Formatted prompt string
        """
        # Qwen-Audio-Chat uses a specific prompt format
        # The audio will be inserted by the processor
        prompt = f"<|startofanalysis|>Question: {query}\n"
        
        if num_audios > 1:
            prompt += f"Based on the {num_audios} audio passages provided, "
        else:
            prompt += "Based on the audio passage provided, "
        
        prompt += "please provide a detailed answer.\n<|endofanalysis|>\n<|startofanswer|>"
        
        return prompt
    
    def generate(
        self,
        query: str,
        audio_paths: List[Union[str, Path]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Generate textual response conditioned on query and audio passages.
        
        Args:
            query: Text query/question
            audio_paths: List of paths to retrieved audio files
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with:
                - "response": Generated text response
                - "audio_paths": List of audio paths used
                - "num_audios": Number of audio passages used
        """
        if not audio_paths:
            raise ValueError("At least one audio path must be provided")
        
        # Prepare audio inputs
        audio_inputs = self.prepare_audio_inputs(audio_paths)
        
        # Build prompt
        prompt = self.build_prompt(query, num_audios=len(audio_paths))
        
        # Prepare inputs for the model
        # Qwen-Audio-Chat processor handles multimodal input (text + audio)
        # The processor expects audio in a specific format
        audio_arrays = [audio_input["raw"] for audio_input in audio_inputs]
        sampling_rate = audio_inputs[0]["sampling_rate"]
        
        try:
            # Try standard format: single text with multiple audios
            # Qwen-Audio processor may expect different formats
            # Try the most common format first
            inputs = self.processor(
                text=prompt,
                audios=audio_arrays,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
        except TypeError as e:
            # If TypeError, try with audio as a single list item
            print(f"Warning: Processor format issue: {e}")
            print("Trying alternative format...")
            try:
                # Try with audio as single array if multiple audios
                if len(audio_arrays) == 1:
                    inputs = self.processor(
                        text=prompt,
                        audio=audio_arrays[0],
                        sampling_rate=sampling_rate,
                        return_tensors="pt"
                    )
                else:
                    # For multiple audios, try concatenating or using first audio
                    print("Warning: Multiple audios detected. Using first audio for now.")
                    inputs = self.processor(
                        text=prompt,
                        audio=audio_arrays[0],
                        sampling_rate=sampling_rate,
                        return_tensors="pt"
                    )
            except Exception as e2:
                raise RuntimeError(
                    f"Could not process audio inputs with processor. "
                    f"Error: {e2}. Please check the Qwen-Audio-Chat documentation "
                    f"for the correct input format."
                ) from e2
        except Exception as e:
            raise RuntimeError(
                f"Failed to process inputs with processor: {e}. "
                f"Please ensure the processor is correctly loaded and the audio format is correct."
            ) from e
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode response
        # Extract only the generated tokens (exclude input)
        input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_text = self.processor.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True
        )
        
        return {
            "response": generated_text.strip(),
            "audio_paths": [str(p) for p in audio_paths],
            "num_audios": len(audio_paths),
            "query": query
        }
    
    def generate_batch(
        self,
        queries: List[str],
        audio_paths_list: List[List[Union[str, Path]]],
        **generation_kwargs
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Generate responses for multiple queries in batch.
        
        Args:
            queries: List of text queries
            audio_paths_list: List of lists of audio paths (one per query)
            **generation_kwargs: Generation parameters
        
        Returns:
            List of generation results
        """
        results = []
        for query, audio_paths in zip(queries, audio_paths_list):
            result = self.generate(query, audio_paths, **generation_kwargs)
            results.append(result)
        return results

