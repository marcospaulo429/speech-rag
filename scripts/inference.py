"""Script de inferência com pipeline SpeechRAG."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import yaml
from src.pipeline.speech_rag_pipeline import SpeechRAGPipeline
from src.retriever.speech_retriever import SpeechRetriever
from src.generator.slm_generator import SLMGenerator
from src.speech_encoder.speech_encoder import SpeechEncoder
from src.text_embedder.text_embedder import TextEmbedder
from src.adapter.adapter import SpeechAdapter


def load_pipeline(config_path: str, device: str = "cuda") -> SpeechRAGPipeline:
    """Carrega pipeline completo."""
    # Carrega modelos
    speech_encoder = SpeechEncoder.from_config("configs/speech_encoder.yaml", device=device)
    text_embedder = TextEmbedder.from_config("configs/text_embedder.yaml", device=device)
    adapter = SpeechAdapter.from_config("configs/adapter.yaml")
    adapter.load_state_dict(torch.load("checkpoints/best_adapter.pt"))  # Ajustar caminho
    adapter = adapter.to(device)
    
    generator = SLMGenerator.from_config("configs/generator.yaml", device=device)
    
    # Cria retriever
    retriever = SpeechRetriever(
        speech_encoder=speech_encoder,
        text_embedder=text_embedder,
        adapter=adapter,
        top_k=5
    )
    
    # Carrega índice se existir
    try:
        retriever.load_index("speech_index")
    except:
        print("Índice não encontrado. Indexe passagens primeiro.")
    
    # Pipeline
    pipeline = SpeechRAGPipeline(
        retriever=retriever,
        generator=generator
    )
    
    return pipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(args.config, device)
    
    result = pipeline.answer(args.question)
    print(f"Question: {args.question}")
    print(f"Answer: {result['answer']}")

