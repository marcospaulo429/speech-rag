"""Script para indexar passagens de áudio do LibriSpeech."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import argparse
import yaml
from pathlib import Path

from src.speech_encoder.speech_encoder import SpeechEncoder
from src.text_embedder.text_embedder import TextEmbedder
from src.adapter.adapter import SpeechAdapter
from src.retriever.speech_retriever import SpeechRetriever
from src.data.librispeech_loader import load_librispeech_split


def main():
    parser = argparse.ArgumentParser(description="Indexar passagens de áudio do LibriSpeech")
    parser.add_argument("--split", type=str, default="test-clean",
                       help="Split do LibriSpeech para indexar")
    parser.add_argument("--adapter_checkpoint", type=str, required=True,
                       help="Caminho para checkpoint do adaptador treinado")
    parser.add_argument("--index_name", type=str, default="librispeech_index",
                       help="Nome do índice a ser salvo")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limitar número de passagens (para testes rápidos)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, default: auto)")
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Carrega modelos (frozen)
    print("Carregando modelos...")
    speech_encoder = SpeechEncoder.from_config(
        "configs/speech_encoder.yaml",
        device=device
    )
    text_embedder = TextEmbedder.from_config(
        "configs/text_embedder.yaml",
        device=device
    )
    
    # Carrega adaptador treinado
    print(f"Carregando adaptador de {args.adapter_checkpoint}...")
    adapter = SpeechAdapter.from_config("configs/adapter.yaml")
    checkpoint = torch.load(args.adapter_checkpoint, map_location=device)
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    adapter = adapter.to(device)
    adapter.eval()
    
    # Cria retriever
    retriever = SpeechRetriever(
        speech_encoder=speech_encoder,
        text_embedder=text_embedder,
        adapter=adapter,
        top_k=5,
        device=device
    )
    
    # Carrega dados do LibriSpeech
    print(f"Carregando split {args.split} do LibriSpeech...")
    data = load_librispeech_split(args.split, limit=args.limit)
    
    # Prepara passagens de áudio
    print(f"Preparando {len(data)} passagens de áudio...")
    audio_passages = [item["audio"] for item in data]
    metadata = [
        {"text": item["text"], "index": i, "split": args.split}
        for i, item in enumerate(data)
    ]
    
    # Indexa passagens
    print("Indexando passagens...")
    retriever.index_audio_passages(audio_passages, metadata)
    
    # Salva índice
    print(f"Salvando índice como '{args.index_name}'...")
    index_path = retriever.save_index(args.index_name)
    print(f"Índice salvo em: {index_path}")
    print(f"Total de passagens indexadas: {retriever.indexer.get_size()}")


if __name__ == "__main__":
    main()

