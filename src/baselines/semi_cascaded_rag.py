"""Baseline: Semi-cascaded RAG (ASR com correção → Text RAG)."""

from typing import Dict, Any, List
import torch

from .cascaded_rag import CascadedRAG


class SemiCascadedRAG(CascadedRAG):
    """Sistema RAG semi-cascaded: ASR com correção → Text RAG."""
    
    def __init__(
        self,
        asr_model: Any,
        text_retriever: Any,
        text_generator: Any,
        error_corrector: Any = None  # Modelo de correção de erros ASR
    ):
        """
        Inicializa sistema semi-cascaded.
        
        Args:
            asr_model: Modelo ASR
            text_retriever: Retriever de texto
            text_generator: Gerador de texto
            error_corrector: Corretor de erros ASR opcional
        """
        super().__init__(asr_model, text_retriever, text_generator)
        self.error_corrector = error_corrector
    
    def transcribe(self, audio: torch.Tensor) -> str:
        """Transcreve áudio com correção de erros."""
        # 1. Transcreve
        transcription = super().transcribe(audio)
        
        # 2. Corrige erros se corretor disponível
        if self.error_corrector is not None:
            transcription = self.error_corrector.correct(transcription)
        
        return transcription

