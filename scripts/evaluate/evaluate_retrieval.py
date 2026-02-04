"""Avaliação de retrieval: Recall@K, MRR, NDCG."""

import numpy as np
from typing import List, Dict, Any


def recall_at_k(relevant_indices: List[int], retrieved_indices: List[int], k: int) -> float:
    """
    Calcula Recall@K.
    
    Args:
        relevant_indices: Índices relevantes (ground truth)
        retrieved_indices: Índices recuperados
        k: Top-K
    
    Returns:
        Recall@K
    """
    if len(relevant_indices) == 0:
        return 0.0
    
    top_k_retrieved = retrieved_indices[:k]
    relevant_retrieved = len(set(relevant_indices) & set(top_k_retrieved))
    
    return relevant_retrieved / len(relevant_indices)


def mean_reciprocal_rank(relevant_indices: List[int], retrieved_indices: List[int]) -> float:
    """
    Calcula Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_indices: Índices relevantes
        retrieved_indices: Índices recuperados
    
    Returns:
        MRR
    """
    if len(relevant_indices) == 0:
        return 0.0
    
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(relevant_indices: List[int], retrieved_indices: List[int], scores: List[float], k: int) -> float:
    """
    Calcula Normalized Discounted Cumulative Gain (NDCG@K).
    
    Args:
        relevant_indices: Índices relevantes
        retrieved_indices: Índices recuperados
        scores: Scores dos resultados
        k: Top-K
    
    Returns:
        NDCG@K
    """
    if len(relevant_indices) == 0:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k], start=1):
        if idx in relevant_indices:
            dcg += scores[i-1] / np.log2(i + 1)
    
    # IDCG (ideal DCG)
    ideal_scores = sorted([scores[i] for i in relevant_indices if i < len(scores)], reverse=True)
    idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores[:k]))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_retrieval(
    results: List[Dict[str, Any]],
    ground_truth: List[List[int]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Avalia retrieval completo.
    
    Args:
        results: Lista de resultados de retrieval
        ground_truth: Lista de listas de índices relevantes
        k_values: Valores de K para Recall@K
    
    Returns:
        Dicionário com métricas
    """
    metrics = {}
    
    # Recall@K
    for k in k_values:
        recalls = []
        for i, (result, gt) in enumerate(zip(results, ground_truth)):
            retrieved = result.get("indices", [])
            recall = recall_at_k(gt, retrieved, k)
            recalls.append(recall)
        metrics[f"recall@{k}"] = np.mean(recalls)
    
    # MRR
    mrrs = []
    for result, gt in zip(results, ground_truth):
        retrieved = result.get("indices", [])
        mrr = mean_reciprocal_rank(gt, retrieved)
        mrrs.append(mrr)
    metrics["mrr"] = np.mean(mrrs)
    
    # NDCG@K
    for k in k_values:
        ndcgs = []
        for result, gt in zip(results, ground_truth):
            retrieved = result.get("indices", [])
            scores = result.get("scores", [1.0] * len(retrieved))
            ndcg = ndcg_at_k(gt, retrieved, scores, k)
            ndcgs.append(ndcg)
        metrics[f"ndcg@{k}"] = np.mean(ndcgs)
    
    return metrics


if __name__ == "__main__":
    # Exemplo de uso
    results = [
        {"indices": [0, 1, 2, 3, 4], "scores": [0.9, 0.8, 0.7, 0.6, 0.5]},
        {"indices": [1, 0, 2, 3, 4], "scores": [0.85, 0.75, 0.65, 0.55, 0.45]}
    ]
    ground_truth = [[0, 1], [1, 2]]
    
    metrics = evaluate_retrieval(results, ground_truth)
    print(metrics)

