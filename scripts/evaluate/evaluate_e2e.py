"""Avaliação end-to-end: F1, EM sobre respostas finais."""

from typing import List, Dict, Any
from .evaluate_generation import evaluate_generation, f1_score, exact_match


def evaluate_e2e(
    predictions: List[Dict[str, Any]],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Avalia pipeline end-to-end.
    
    Args:
        predictions: Lista de predições com campo "answer"
        ground_truths: Lista de respostas verdadeiras
    
    Returns:
        Dicionário com métricas
    """
    answers = [pred.get("answer", "") for pred in predictions]
    
    # Métricas de geração
    metrics = evaluate_generation(answers, ground_truths)
    
    # Métricas adicionais de retrieval (se disponíveis)
    if predictions and "retrieved_passages" in predictions[0]:
        num_retrieved = [len(pred.get("retrieved_passages", [])) for pred in predictions]
        metrics["avg_retrieved"] = sum(num_retrieved) / len(num_retrieved) if num_retrieved else 0.0
    
    return metrics


if __name__ == "__main__":
    # Exemplo
    predictions = [
        {"answer": "The answer is 42", "retrieved_passages": [{"score": 0.9}]},
        {"answer": "It depends", "retrieved_passages": [{"score": 0.7}]}
    ]
    ground_truths = ["The answer is 42", "It depends on the context"]
    
    metrics = evaluate_e2e(predictions, ground_truths)
    print(metrics)

