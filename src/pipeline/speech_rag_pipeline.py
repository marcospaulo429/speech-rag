"""Pipeline end-to-end SpeechRAG para question answering."""

from typing import List, Dict, Any, Optional, Union
import torch

from ..retriever.speech_retriever import SpeechRetriever
from ..generator.slm_generator import SLMGenerator
from ..utils.audio_chunking import AudioChunker


class SpeechRAGPipeline:
    """Pipeline completo SpeechRAG para question answering."""
    
    def __init__(
        self,
        retriever: SpeechRetriever,
        generator: SLMGenerator,
        top_k: int = 5,
        chunk_long_audio: bool = True,
        max_audio_length: float = 30.0
    ):
        """
        Inicializa pipeline.
        
        Args:
            retriever: Retriever cross-modal
            generator: Gerador SLM
            top_k: Número de passagens a recuperar
            chunk_long_audio: Se True, faz chunking de áudio longo
            max_audio_length: Comprimento máximo de áudio em segundos
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.chunk_long_audio = chunk_long_audio
        self.max_audio_length = max_audio_length
        
        if chunk_long_audio:
            self.chunker = AudioChunker(
                chunk_duration=max_audio_length,
                sample_rate=16000
            )
        else:
            self.chunker = None
    
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Responde pergunta usando SpeechRAG.
        
        Args:
            question: Pergunta em texto
            top_k: Número de passagens a recuperar (None = usa self.top_k)
        
        Returns:
            Dicionário com resposta e metadados
        """
        top_k = top_k or self.top_k
        
        # 1. Recupera passagens de áudio relevantes
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved:
            return {
                "answer": "Nenhuma passagem relevante encontrada.",
                "retrieved_passages": [],
                "num_retrieved": 0
            }
        
        # 2. Prepara contexto de áudio
        # Usa a passagem mais relevante (ou concatena top-k)
        audio_context = retrieved[0]["audio"]
        
        # Chunking se necessário
        if self.chunker is not None:
            if isinstance(audio_context, torch.Tensor):
                audio_length = len(audio_context) / 16000  # Assume 16kHz
                if audio_length > self.max_audio_length:
                    chunks = self.chunker.chunk_audio(audio_context)
                    audio_context = chunks[0].audio  # Usa primeiro chunk
        
        # 3. Gera resposta condicionada em áudio
        prompt = f"Question: {question}\n\nAnswer based on the audio context:"
        answer = self.generator.generate(audio_context, prompt=prompt)
        
        return {
            "answer": answer,
            "retrieved_passages": retrieved,
            "num_retrieved": len(retrieved),
            "top_score": retrieved[0]["score"] if retrieved else 0.0
        }
    
    def answer_with_multiple_passages(
        self,
        question: str,
        top_k: Optional[int] = None,
        combine_method: str = "first"
    ) -> Dict[str, Any]:
        """
        Responde usando múltiplas passagens.
        
        Args:
            question: Pergunta
            top_k: Número de passagens
            combine_method: Método de combinação ("first", "concatenate")
        
        Returns:
            Resposta com múltiplas passagens
        """
        top_k = top_k or self.top_k
        
        # Recupera passagens
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved:
            return {
                "answer": "Nenhuma passagem relevante encontrada.",
                "retrieved_passages": []
            }
        
        if combine_method == "first":
            # Usa apenas a primeira passagem
            audio_context = retrieved[0]["audio"]
            prompt = f"Question: {question}\n\nAnswer based on the audio context:"
            answer = self.generator.generate(audio_context, prompt=prompt)
        elif combine_method == "concatenate":
            # Concatena múltiplas passagens (simplificado)
            audio_context = retrieved[0]["audio"]  # Por enquanto usa apenas a primeira
            prompt = f"Question: {question}\n\nAnswer based on the audio context from {len(retrieved)} passages:"
            answer = self.generator.generate(audio_context, prompt=prompt)
        else:
            raise ValueError(f"Método de combinação desconhecido: {combine_method}")
        
        return {
            "answer": answer,
            "retrieved_passages": retrieved,
            "num_retrieved": len(retrieved)
        }

