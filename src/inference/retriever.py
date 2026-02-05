"""Speech Retriever for audio passage retrieval from text queries"""

import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle

from ..models import TextEncoder, SpeechEncoder, SpeechAdapter


class SpeechRetriever:
    """
    Retrieval system for finding audio passages using text queries.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        speech_encoder: SpeechEncoder,
        adapter: SpeechAdapter,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        index_type: str = "flat"  # "flat" or "ivf"
    ):
        """
        Args:
            text_encoder: Text encoder for query embeddings
            speech_encoder: Speech encoder for audio processing
            adapter: Trained speech adapter
            device: Device to run on
            index_type: FAISS index type ("flat" for exact search, "ivf" for approximate)
        """
        self.text_encoder = text_encoder.to(device)
        self.speech_encoder = speech_encoder.to(device)
        self.adapter = adapter.to(device)
        self.device = device
        self.index_type = index_type
        
        # FAISS index
        self.index = None
        self.embedding_dim = adapter.get_embedding_dim()
        
        # Metadata storage
        self.audio_paths = []
        self.audio_metadata = []
    
    def build_index(
        self,
        audio_files: List[Union[str, Path]],
        batch_size: int = 16,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Build FAISS index from audio files.
        
        Args:
            audio_files: List of paths to audio files
            batch_size: Batch size for processing
            metadata: Optional metadata for each audio file
        """
        print(f"Building index from {len(audio_files)} audio files...")
        
        self.audio_paths = [str(f) for f in audio_files]
        self.audio_metadata = metadata if metadata else [{}] * len(audio_files)
        
        # Process audio files in batches
        all_embeddings = []
        
        self.adapter.eval()
        self.speech_encoder.eval()
        
        with torch.no_grad():
            for i in range(0, len(audio_files), batch_size):
                batch_files = audio_files[i:i + batch_size]
                
                # Process batch
                batch_embeddings = []
                for audio_file in batch_files:
                    # Encode audio
                    speech_reprs = self.speech_encoder.encode(
                        str(audio_file),
                        device=self.device
                    )
                    audio_embedding = self.adapter(speech_reprs)
                    batch_embeddings.append(audio_embedding.cpu().numpy())
                
                # Stack batch
                batch_embeddings = np.vstack(batch_embeddings)
                all_embeddings.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_files)}/{len(audio_files)} files")
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings).astype('float32')
        
        print(f"Created embeddings of shape: {all_embeddings.shape}")
        
        # Create FAISS index
        if self.index_type == "flat":
            # Exact search using L2 distance
            # For cosine similarity, we normalize and use inner product
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            # Approximate search (faster for large datasets)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(100, len(audio_files) // 10)  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.nprobe = 10
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Normalize embeddings for cosine similarity (inner product = cosine for normalized vectors)
        faiss.normalize_L2(all_embeddings)
        
        # Add to index
        if self.index_type == "ivf":
            # Train index first
            print("Training IVF index...")
            self.index.train(all_embeddings)
        
        self.index.add(all_embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for audio passages using text query.
        
        Args:
            query: Text query string
            k: Number of results to return
            return_scores: Whether to return similarity scores
        
        Returns:
            List of dictionaries with audio paths, metadata, and scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        self.text_encoder.eval()
        with torch.no_grad():
            query_embedding = self.text_encoder.encode(
                query,
                device=self.device
            ).cpu().numpy().astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            result = {
                "rank": i + 1,
                "audio_path": self.audio_paths[idx],
                "metadata": self.audio_metadata[idx],
                "index": int(idx)
            }
            if return_scores:
                result["score"] = float(score)
            results.append(result)
        
        return results
    
    def save_index(self, index_path: str):
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path to save index
        """
        if self.index is None:
            raise ValueError("No index to save.")
        
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = index_path.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "audio_paths": self.audio_paths,
                "audio_metadata": self.audio_metadata,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str):
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to saved index
        """
        index_path = Path(index_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_path.with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.audio_paths = metadata["audio_paths"]
        self.audio_metadata = metadata["audio_metadata"]
        self.embedding_dim = metadata["embedding_dim"]
        self.index_type = metadata["index_type"]
        
        print(f"Index loaded from {index_path} with {self.index.ntotal} vectors")

