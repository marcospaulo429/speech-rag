"""Script de teste end-to-end com LibriSpeech."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import argparse
import random

from src.speech_encoder.speech_encoder import SpeechEncoder
from src.text_embedder.text_embedder import TextEmbedder
from src.adapter.adapter import SpeechAdapter
from src.generator.slm_generator import SLMGenerator
from src.retriever.speech_retriever import SpeechRetriever
from src.pipeline.speech_rag_pipeline import SpeechRAGPipeline
from src.data.librispeech_loader import load_librispeech_split


def generate_test_queries(data: list, num_queries: int = 5) -> list:
    """Gera queries de teste a partir de transcrições."""
    queries = []
    indices = random.sample(range(len(data)), min(num_queries, len(data)))
    
    for idx in indices:
        text = data[idx]["text"]
        words = text.split()
        
        # Cria query a partir das primeiras palavras
        if len(words) > 3:
            query = " ".join(words[:3]) + "?"
        else:
            query = text + "?"
        
        queries.append({
            "query": query,
            "expected_topic": text  # Para referência
        })
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Teste end-to-end com LibriSpeech")
    parser.add_argument("--adapter_checkpoint", type=str, required=True,
                       help="Caminho para checkpoint do adaptador")
    parser.add_argument("--index_name", type=str, default="librispeech_index",
                       help="Nome do índice carregado")
    parser.add_argument("--split", type=str, default="test-clean",
                       help="Split do LibriSpeech para teste")
    parser.add_argument("--num_queries", type=int, default=5,
                       help="Número de queries para testar")
    parser.add_argument("--limit", type=int, default=100,
                       help="Limitar número de amostras do split")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Carrega modelos
    print("Carregando modelos...")
    speech_encoder = SpeechEncoder.from_config("configs/speech_encoder.yaml", device=device)
    text_embedder = TextEmbedder.from_config("configs/text_embedder.yaml", device=device)
    
    # Carrega adaptador
    print(f"Carregando adaptador de {args.adapter_checkpoint}...")
    adapter = SpeechAdapter.from_config("configs/adapter.yaml")
    checkpoint = torch.load(args.adapter_checkpoint, map_location=device)
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    adapter = adapter.to(device)
    adapter.eval()
    
    # Carrega gerador (pode ser pesado, então opcional)
    try:
        print("Carregando gerador SLM...")
        generator = SLMGenerator.from_config("configs/generator.yaml", device=device)
        use_generator = True
    except Exception as e:
        print(f"Aviso: Não foi possível carregar gerador SLM: {e}")
        print("Continuando apenas com retrieval...")
        generator = None
        use_generator = False
    
    # Cria retriever
    retriever = SpeechRetriever(
        speech_encoder=speech_encoder,
        text_embedder=text_embedder,
        adapter=adapter,
        top_k=5,
        device=device
    )
    
    # Carrega índice
    print(f"Carregando índice '{args.index_name}'...")
    try:
        retriever.load_index(args.index_name)
        print(f"Índice carregado: {retriever.indexer.get_size()} passagens")
    except Exception as e:
        print(f"Erro ao carregar índice: {e}")
        print("Execute scripts/index_librispeech.py primeiro.")
        return
    
    # Carrega dados para gerar queries
    print(f"Carregando split {args.split}...")
    data = load_librispeech_split(args.split, limit=args.limit)
    
    # Gera queries de teste
    print(f"Gerando {args.num_queries} queries de teste...")
    queries = generate_test_queries(data, num_queries=args.num_queries)
    
    # Cria pipeline se gerador disponível
    if use_generator:
        pipeline = SpeechRAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=5
        )
    else:
        pipeline = None
    
    # Executa testes
    print("\n" + "="*80)
    print("TESTE END-TO-END")
    print("="*80 + "\n")
    
    for i, query_info in enumerate(queries, 1):
        query = query_info["query"]
        expected = query_info["expected_topic"]
        
        print(f"Query {i}: {query}")
        print(f"Tópico esperado: {expected[:100]}...")
        print("-" * 80)
        
        # Retrieval
        retrieved = retriever.retrieve(query, top_k=5)
        print(f"\nRetrieved {len(retrieved)} passagens:")
        for j, result in enumerate(retrieved, 1):
            print(f"  {j}. Score: {result['score']:.4f}")
            if 'metadata' in result and 'text' in result['metadata']:
                print(f"     Text: {result['metadata']['text'][:80]}...")
        
        # Geração (se disponível)
        if pipeline:
            try:
                print("\nGerando resposta...")
                answer_result = pipeline.answer(query)
                print(f"Resposta: {answer_result['answer'][:200]}...")
            except Exception as e:
                print(f"Erro na geração: {e}")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

