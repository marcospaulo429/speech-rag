"""Local Spoken-SQuAD dataset loader"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from .preprocessing import AudioPreprocessor


class LocalSpokenSQuADDataset(Dataset):
    """
    Dataset loader for local Spoken-SQuAD files.
    Loads from JSON files and WAV files in extracted directories.
    
    Based on the structure from: https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD
    - JSON files: spoken_train-v1.1.json and spoken_test-v1.1.json (SQuAD format)
    - WAV files: Format TopicIndex_ParagraphIndex_SentenceIndex.wav
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" or "dev"
        sample_rate: int = 16000,
        max_audio_length: float = 60.0,
        json_dir: str = "data",  # Directory containing JSON files
        json_train_file: str = "spoken_train-v1.1.json",
        json_dev_file: str = "spoken_test-v1.1.json",
        audio_train_dir: str = "train_wav",
        audio_dev_dir: str = "dev_wav"
    ):
        """
        Args:
            data_dir: Path to directory containing extracted WAV directories
            split: "train" or "dev" (test)
            sample_rate: Target sample rate
            max_audio_length: Maximum audio length in seconds
            json_dir: Path to directory containing JSON files (default: "data")
            json_train_file: Name of training JSON file
            json_dev_file: Name of dev/test JSON file
            audio_train_dir: Directory name with training WAV files (after extraction)
            audio_dev_dir: Directory name with dev WAV files (after extraction)
        """
        self.data_dir = Path(data_dir)
        self.json_dir = Path(json_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # Determine JSON and audio directory based on split
        # JSON files are in json_dir, audio files are in data_dir
        if split == "train":
            json_file = self.json_dir / json_train_file
            self.audio_dir = self.data_dir / audio_train_dir
        else:  # dev/test
            json_file = self.json_dir / json_dev_file
            self.audio_dir = self.data_dir / audio_dev_dir
        
        # Validate files exist
        if not json_file.exists():
            raise FileNotFoundError(
                f"JSON file not found: {json_file}\n"
                f"Make sure you have extracted the JSON files from the Spoken-SQuAD repository."
            )
        
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found: {self.audio_dir}\n"
                f"Make sure you have extracted the ZIP files (train_wav.zip and dev_wav.zip)."
            )
        
        # Load JSON data
        print(f"Loading JSON from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            max_length_seconds=max_audio_length
        )
        
        # Build list of (audio_path, text) pairs
        print(f"Building samples from {self.audio_dir}...")
        self.samples = self._build_samples()
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        if len(self.samples) == 0:
            print(f"Warning: No samples found! Check if WAV files are in {self.audio_dir}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        Uses simple regex-based splitting on sentence endings.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split by sentence endings (. ! ?) followed by whitespace
        sentences = re.split(r'([.!?]+)\s+', text)
        
        # Recombine sentences with their punctuation
        result = []
        current_sentence = ""
        for i, part in enumerate(sentences):
            if part.strip():
                if re.match(r'^[.!?]+$', part):
                    # This is punctuation, add to current sentence
                    current_sentence += part
                    if current_sentence.strip():
                        result.append(current_sentence.strip())
                    current_sentence = ""
                else:
                    # This is text
                    if current_sentence:
                        current_sentence += " " + part
                    else:
                        current_sentence = part
        
        # Add last sentence if exists
        if current_sentence.strip():
            result.append(current_sentence.strip())
        
        # Filter out empty sentences
        return [s for s in result if s.strip()]
    
    def _build_samples(self) -> List[Dict[str, str]]:
        """
        Build list of samples from JSON structure.
        Each sample is a sentence with its corresponding audio file.
        
        Returns:
            List of dicts with 'audio_path' and 'text' keys
        """
        samples = []
        
        # SQuAD format: data -> articles -> paragraphs -> context
        articles = self.data.get("data", [])
        
        for article_idx, article in enumerate(articles):
            paragraphs = article.get("paragraphs", [])
            
            for paragraph_idx, paragraph in enumerate(paragraphs):
                context_text = paragraph.get("context", "")
                
                if not context_text:
                    continue
                
                # Split context into sentences
                sentences = self._split_into_sentences(context_text)
                
                # Map each sentence to its corresponding WAV file
                # Format: TopicIndex_ParagraphIndex_SentenceIndex.wav
                for sentence_idx, sentence in enumerate(sentences):
                    # Construct expected audio filename
                    audio_filename = f"{article_idx}_{paragraph_idx}_{sentence_idx}.wav"
                    audio_path = self.audio_dir / audio_filename
                    
                    # Only add if audio file exists
                    if audio_path.exists():
                        samples.append({
                            "audio_path": str(audio_path),
                            "text": sentence.strip()
                        })
                    # Note: We skip sentences without corresponding audio files
                    # This is expected as not all sentences may have audio files
        
        return samples
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'audio' (tensor), 'text' (string), 'sample_rate' (int)
        """
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        text = sample["text"]
        
        # Load and preprocess audio
        try:
            waveform, audio_sr = self.preprocessor.load_audio(audio_path)
            waveform = self.preprocessor.process(waveform, sample_rate=audio_sr)
        except Exception as e:
            # Fallback: return zeros if audio loading fails
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            waveform = torch.zeros(int(self.sample_rate * self.max_audio_length))
        
        return {
            "audio": waveform,
            "text": text,
            "sample_rate": self.sample_rate
        }
    
    def get_collate_fn(self):
        """
        Get collate function for DataLoader.
        Handles variable-length audio sequences.
        Compatible with SpeechDataset interface.
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

