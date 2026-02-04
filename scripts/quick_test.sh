#!/bin/bash
# Script rápido para testar o sistema completo com LibriSpeech (dados limitados)

set -e  # Para em caso de erro

echo "=========================================="
echo "Teste Rápido SpeechRAG com LibriSpeech"
echo "=========================================="
echo ""

# Ativa ambiente virtual se existir
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Ambiente virtual ativado"
fi

# Configurações (ajustáveis)
TRAIN_LIMIT=1000
VAL_LIMIT=100
INDEX_LIMIT=500
NUM_QUERIES=50

# Ajusta número de épocas no config ou usa padrão
# (O número de épocas vem do configs/training.yaml)

echo "Configurações:"
echo "  - Treinamento: $TRAIN_LIMIT amostras"
echo "  - Validação: $VAL_LIMIT amostras"
echo "  - Indexação: $INDEX_LIMIT passagens"
echo "  - Queries de avaliação: $NUM_QUERIES"
echo ""

# 1. Treinar adaptador
echo "=========================================="
echo "1. Treinando Adaptador..."
echo "=========================================="
python scripts/train.py \
    --config configs/training.yaml \
    --train_data train-clean-100 \
    --val_data dev-clean \
    --train_limit $TRAIN_LIMIT \
    --val_limit $VAL_LIMIT

# Encontra último checkpoint
LAST_CHECKPOINT=$(ls -t checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$LAST_CHECKPOINT" ]; then
    echo "Erro: Nenhum checkpoint encontrado!"
    exit 1
fi

echo "Checkpoint encontrado: $LAST_CHECKPOINT"
echo ""

# 2. Indexar passagens
echo "=========================================="
echo "2. Indexando Passagens de Áudio..."
echo "=========================================="
python scripts/index_librispeech.py \
    --split test-clean \
    --adapter_checkpoint "$LAST_CHECKPOINT" \
    --index_name librispeech_test \
    --limit $INDEX_LIMIT

echo ""

# 3. Avaliar retrieval
echo "=========================================="
echo "3. Avaliando Retrieval..."
echo "=========================================="
python scripts/evaluate_librispeech_retrieval.py \
    --corpus_split test-clean \
    --index_name librispeech_test \
    --adapter_checkpoint "$LAST_CHECKPOINT" \
    --num_queries $NUM_QUERIES \
    --corpus_limit $INDEX_LIMIT

echo ""

# 4. Teste E2E
echo "=========================================="
echo "4. Teste End-to-End..."
echo "=========================================="
python scripts/test_librispeech_e2e.py \
    --adapter_checkpoint "$LAST_CHECKPOINT" \
    --index_name librispeech_test \
    --num_queries 5 \
    --limit $INDEX_LIMIT

echo ""
echo "=========================================="
echo "Teste Completo Finalizado!"
echo "=========================================="

