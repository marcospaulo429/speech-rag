"""Speech RAG Pipeline integrating retrieval and generation"""

from typing import List, Dict, Optional, Union
from pathlib import Path

from .retriever import SpeechRetriever
from .generator import AudioConditionedGenerator


class SpeechRAGPipeline:
    """
    End-to-end Speech RAG pipeline that combines:
    - SpeechRetriever: Retrieves relevant audio passages from text queries
    - AudioConditionedGenerator: Generates textual responses using Qwen-Audio-Chat
    """
    
    def __init__(
        self,
        retriever: SpeechRetriever,
        generator: AudioConditionedGenerator,
        top_k_audio: int = 3
    ):
        """
        Initialize the Speech RAG Pipeline.
        
        Args:
            retriever: SpeechRetriever instance for audio retrieval
            generator: AudioConditionedGenerator instance for response generation
            top_k_audio: Number of top audio passages to use for generation
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k_audio = top_k_audio
    
    def retrieve_and_generate(
        self,
        query: str,
        k: Optional[int] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        return_retrieval_results: bool = True,
        **generation_kwargs
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve relevant audios and generate response.
        
        Args:
            query: Text query/question
            k: Number of audio passages to retrieve (uses self.top_k_audio if None)
            temperature: Generation temperature
            max_new_tokens: Maximum tokens to generate
            return_retrieval_results: Whether to include retrieval results in output
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dictionary with:
                - "response": Generated text response
                - "query": Original query
                - "retrieval_results": List of retrieved audio results (if return_retrieval_results=True)
                - "audio_paths": List of audio paths used for generation
                - "num_audios": Number of audio passages used
        """
        if k is None:
            k = self.top_k_audio
        
        # Step 1: Retrieve relevant audio passages
        retrieval_results = self.retriever.search(query, k=k)
        
        if not retrieval_results:
            return {
                "response": "No relevant audio passages found for the query.",
                "query": query,
                "retrieval_results": [],
                "audio_paths": [],
                "num_audios": 0
            }
        
        # Step 2: Extract audio paths from retrieval results
        audio_paths = [result["audio_path"] for result in retrieval_results]
        
        # Step 3: Generate response using retrieved audios
        generation_result = self.generator.generate(
            query=query,
            audio_paths=audio_paths,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )
        
        # Step 4: Combine results
        output = {
            "response": generation_result["response"],
            "query": query,
            "audio_paths": generation_result["audio_paths"],
            "num_audios": generation_result["num_audios"]
        }
        
        if return_retrieval_results:
            output["retrieval_results"] = retrieval_results
        
        return output
    
    def retrieve_only(
        self,
        query: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve audio passages without generation.
        
        Args:
            query: Text query
            k: Number of results to return
        
        Returns:
            List of retrieval results
        """
        return self.retriever.search(query, k=k)
    
    def generate_only(
        self,
        query: str,
        audio_paths: List[Union[str, Path]],
        **generation_kwargs
    ) -> Dict:
        """
        Generate response from query and audio paths (skip retrieval).
        
        Args:
            query: Text query
            audio_paths: List of audio file paths
            **generation_kwargs: Generation parameters
        
        Returns:
            Generation result dictionary
        """
        return self.generator.generate(query, audio_paths, **generation_kwargs)

