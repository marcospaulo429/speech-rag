"""Script para avaliar retrieval usando LibriSpeech."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import argparse
import numpy as np
from typing import List, Dict, Any

from src.speech_encoder.speech_encoder import SpeechEncoder
from src.text_embedder.text_embedder import TextEmbedder
from src.adapter.adapter import SpeechAdapter
from src.retriever.speech_retriever import SpeechRetriever
from src.data.librispeech_loader import load_librispeech_split
import sys
import os
# Adiciona path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.evaluate.evaluate_retrieval import (
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k
)


def generate_queries_from_transcriptions(
    data: List[Dict[str, Any]],
    num_queries: int = 100
) -> List[Dict[str, Any]]:
    """
    Gera queries de teste a partir de transcrições.
    
    Args:
        data: Dados do LibriSpeech
        num_queries: Número de queries a gerar
    
    Returns:
        Lista de queries com ground truth
    """
    queries = []
    
    # Seleciona amostras aleatórias
    indices = np.random.choice(len(data), min(num_queries, len(data)), replace=False)
    
    for idx in indices:
        item = data[idx]
        text = item["text"]
        
        # Cria query simples (primeiras palavras da transcrição)
        words = text.split()
        if len(words) > 5:
            query_text = " ".join(words[:5])  # Primeiras 5 palavras
        else:
            query_text = text
        
        queries.append({
            "query": query_text,
            "ground_truth_indices": [idx],  # A passagem original é a relevante
            "audio": item["audio"],
            "full_text": text
        })
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Avaliar retrieval com LibriSpeech")
    parser.add_argument("--corpus_split", type=str, default="test-clean",
                       help="Split do LibriSpeech para usar como corpus")
    parser.add_argument("--index_name", type=str, default="librispeech_index",
                       help="Nome do índice carregado")
    parser.add_argument("--adapter_checkpoint", type=str, required=True,
                       help="Caminho para checkpoint do adaptador")
    parser.add_argument("--num_queries", type=int, default=100,
                       help="Número de queries para avaliação")
    parser.add_argument("--corpus_limit", type=int, default=None,
                       help="Limitar tamanho do corpus (para testes rápidos)")
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
    
    # Cria retriever
    retriever = SpeechRetriever(
        speech_encoder=speech_encoder,
        text_embedder=text_embedder,
        adapter=adapter,
        top_k=10,
        device=device
    )
    
    # Carrega índice
    print(f"Carregando índice '{args.index_name}'...")
    try:
        retriever.load_index(args.index_name)
        print(f"Índice carregado: {retriever.indexer.get_size()} passagens")
    except Exception as e:
        print(f"Erro ao carregar índice: {e}")
        print("Execute scripts/index_librispeech.py primeiro para criar o índice.")
        return
    
    # Carrega corpus (mesmo usado para indexação)
    print(f"Carregando corpus do split {args.corpus_split}...")
    corpus_data = load_librispeech_split(args.corpus_split, limit=args.corpus_limit)
    
    # Gera queries
    print(f"Gerando {args.num_queries} queries...")
    queries = generate_queries_from_transcriptions(corpus_data, num_queries=args.num_queries)
    
    # Executa retrieval
    print("Executando retrieval...")
    results = []
    for i, query_info in enumerate(queries):
        query = query_info["query"]
        retrieved = retriever.retrieve(query, top_k=10)
        
        # Extrai índices e scores
        retrieved_indices = [r["rank"] - 1 for r in retrieved]  # rank começa em 1
        scores = [r["score"] for r in retrieved]
        
        results.append({
            "indices": retrieved_indices,
            "scores": scores
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processadas {i + 1}/{len(queries)} queries...")
    
    # Ground truth
    ground_truth = [q["ground_truth_indices"] for q in queries]
    
    # Calcula métricas
    print("\nCalculando métricas...")
    k_values = [1, 5, 10]
    
    # Recall@K
    for k in k_values:
        recalls = []
        for result, gt in zip(results, ground_truth):
            recall = recall_at_k(gt, result["indices"], k)
            recalls.append(recall)
        print(f"Recall@{k}: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    
    # MRR
    mrrs = []
    for result, gt in zip(results, ground_truth):
        mrr = mean_reciprocal_rank(gt, result["indices"])
        mrrs.append(mrr)
    print(f"MRR: {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
    
    # NDCG@K
    for k in k_values:
        ndcgs = []
        for result, gt in zip(results, ground_truth):
            ndcg = ndcg_at_k(gt, result["indices"], result["scores"], k)
            ndcgs.append(ndcg)
        print(f"NDCG@{k}: {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f}")
    
    print("\nAvaliação concluída!")


if __name__ == "__main__":
    main()

