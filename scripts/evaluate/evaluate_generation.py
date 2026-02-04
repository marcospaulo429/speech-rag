"""Avaliação de geração: F1, Exact Match, BLEU."""

import re
from typing import List, Dict, Any
from collections import Counter


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calcula F1 score entre predição e ground truth.
    
    Args:
        prediction: Texto predito
        ground_truth: Texto verdadeiro
    
    Returns:
        F1 score
    """
    def normalize_text(text: str) -> List[str]:
        """Normaliza e tokeniza texto."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    pred_tokens = normalize_text(prediction)
    gt_tokens = normalize_text(ground_truth)
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # Contagens
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # Intersecção
    common = sum((pred_counter & gt_counter).values())
    
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Calcula Exact Match.
    
    Args:
        prediction: Texto predito
        ground_truth: Texto verdadeiro
    
    Returns:
        1.0 se exato, 0.0 caso contrário
    """
    pred_normalized = prediction.lower().strip()
    gt_normalized = ground_truth.lower().strip()
    
    return 1.0 if pred_normalized == gt_normalized else 0.0


def bleu_score(prediction: str, ground_truth: str, n: int = 4) -> float:
    """
    Calcula BLEU score simplificado (n-gram overlap).
    
    Args:
        prediction: Texto predito
        ground_truth: Texto verdadeiro
        n: Ordem máxima de n-grams
    
    Returns:
        BLEU score
    """
    def get_ngrams(text: str, n: int) -> Counter:
        """Obtém n-grams de texto."""
        tokens = text.lower().split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    pred_ngrams = get_ngrams(prediction, n)
    gt_ngrams = get_ngrams(ground_truth, n)
    
    if len(pred_ngrams) == 0:
        return 0.0
    
    # Overlap
    overlap = sum((pred_ngrams & gt_ngrams).values())
    precision = overlap / len(pred_ngrams)
    
    return precision


def evaluate_generation(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Avalia geração completa.
    
    Args:
        predictions: Lista de predições
        ground_truths: Lista de ground truths
    
    Returns:
        Dicionário com métricas
    """
    f1_scores = [f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    em_scores = [exact_match(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    bleu_scores = [bleu_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    
    return {
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    }


if __name__ == "__main__":
    # Exemplo
    predictions = ["The cat sat on the mat", "The dog ran"]
    ground_truths = ["A cat sat on a mat", "The dog ran fast"]
    
    metrics = evaluate_generation(predictions, ground_truths)
    print(metrics)

