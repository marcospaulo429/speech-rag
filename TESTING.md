# Guia de Testes - SpeechRAG com LibriSpeech

Este guia explica como testar o sistema SpeechRAG completo usando o dataset LibriSpeech.

## Pré-requisitos

### 1. Ambiente Virtual

```bash
# Criar ambiente com uv
cd /media/marcos-paulo/f395f9db-cdb2-4e13-bdd7-a81f3e504c6a1/speech-rag
uv venv
source .venv/bin/activate
```

### 2. Instalar Dependências

```bash
# Instalar todas as dependências
uv pip install -r requirements.txt

# Instalar pacote em modo desenvolvimento
uv pip install -e .
```

### 3. Verificar Instalação

```bash
# Testar imports básicos
python -c "from src.data.librispeech_loader import load_librispeech_split; print('OK')"
```

## Fluxo Completo de Teste

### Passo 1: Testes Unitários (Opcional)

Antes de rodar com dados reais, você pode executar os testes unitários:

```bash
# Testes básicos (sem modelos)
pytest tests/test_normalization.py -v
pytest tests/test_audio_chunking.py -v
pytest tests/test_adapter.py -v
pytest tests/test_retriever.py -v
```

### Passo 2: Treinar o Adaptador

O adaptador é o único componente que precisa ser treinado. Todos os outros modelos são carregados pré-treinados do Hugging Face.

```bash
# Treinar com LibriSpeech (usando split pequeno para teste rápido)
python scripts/train.py \
    --config configs/training.yaml \
    --train_data train-clean-100 \
    --val_data dev-clean \
    --train_limit 1000 \
    --val_limit 100

# Ou treinar com mais dados (sem limite)
python scripts/train.py \
    --config configs/training.yaml \
    --train_data train-clean-100 \
    --val_data dev-clean
```

**Nota**: Na primeira execução, o LibriSpeech será baixado automaticamente do Hugging Face (pode demorar).

**Checkpoints salvos em**: `checkpoints/checkpoint_epoch_N.pt`

### Passo 3: Indexar Passagens de Áudio

Após treinar o adaptador, indexe passagens de áudio para retrieval:

```bash
# Indexar split de teste (limitado para teste rápido)
python scripts/index_librispeech.py \
    --split test-clean \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --index_name librispeech_index \
    --limit 1000

# Ou indexar mais passagens
python scripts/index_librispeech.py \
    --split test-clean \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --index_name librispeech_index
```

**Índice salvo em**: `indices/librispeech_index_*.index`

### Passo 4: Avaliar Retrieval

Avalie a qualidade do retrieval:

```bash
python scripts/evaluate_librispeech_retrieval.py \
    --corpus_split test-clean \
    --index_name librispeech_index \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --num_queries 100 \
    --corpus_limit 1000
```

Isso calculará:
- **Recall@K** (K=1, 5, 10)
- **MRR** (Mean Reciprocal Rank)
- **NDCG@K** (K=1, 5, 10)

### Passo 5: Teste End-to-End

Teste o pipeline completo:

```bash
python scripts/test_librispeech_e2e.py \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --index_name librispeech_index \
    --split test-clean \
    --num_queries 5 \
    --limit 100
```

## Splits Disponíveis do LibriSpeech

- `train-clean-100`: 100 horas de treinamento (limpo)
- `train-clean-360`: 360 horas de treinamento (limpo)
- `train-other-500`: 500 horas de treinamento (outros)
- `dev-clean`: Validação (limpo)
- `dev-other`: Validação (outros)
- `test-clean`: Teste (limpo)
- `test-other`: Teste (outros)

## Exemplo Completo (Teste Rápido)

### Opção 1: Script Automatizado (Recomendado)

Execute tudo de uma vez com o script automatizado:

```bash
# Executa treinamento, indexação, avaliação e teste E2E
bash scripts/quick_test.sh
```

### Opção 2: Passo a Passo Manual

Para um teste rápido com dados limitados:

```bash
# 1. Treinar adaptador (1000 amostras)
python scripts/train.py \
    --config configs/training.yaml \
    --train_data train-clean-100 \
    --val_data dev-clean \
    --train_limit 1000 \
    --val_limit 100

# 2. Indexar (500 passagens)
python scripts/index_librispeech.py \
    --split test-clean \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --index_name librispeech_test \
    --limit 500

# 3. Avaliar retrieval (50 queries)
python scripts/evaluate_librispeech_retrieval.py \
    --corpus_split test-clean \
    --index_name librispeech_test \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --num_queries 50 \
    --corpus_limit 500

# 4. Teste E2E (5 queries)
python scripts/test_librispeech_e2e.py \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --index_name librispeech_test \
    --num_queries 5 \
    --limit 500
```

## Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'src'"

**Solução**: Instale o pacote em modo desenvolvimento:
```bash
uv pip install -e .
```

### Erro: "CUDA out of memory"

**Solução**: 
- Use `--device cpu` nos scripts
- Reduza `--limit` ou `--batch_size` no config
- Use splits menores

### Erro: "Index not found"

**Solução**: Execute `scripts/index_librispeech.py` primeiro para criar o índice.

### LibriSpeech demora muito para baixar

**Solução**: 
- Use `--limit` para testar com menos dados primeiro
- O dataset é baixado apenas uma vez e fica em cache

### Modelos do Hugging Face não carregam

**Solução**:
- Verifique conexão com internet
- Os modelos são baixados automaticamente na primeira execução
- Para modelos grandes, pode demorar

## Estrutura de Arquivos Gerados

```
speech-rag/
├── checkpoints/           # Checkpoints do adaptador
│   └── checkpoint_epoch_*.pt
├── indices/               # Índices FAISS salvos
│   ├── librispeech_index_*.index
│   └── librispeech_index_*.metadata
└── data/                 # Dados (opcional, se salvar JSON)
    ├── train.json
    └── val.json
```

## Próximos Passos

Após validar com LibriSpeech, você pode:

1. **Treinar com mais dados**: Remover `--limit` e usar splits maiores
2. **Avaliar em outros datasets**: Adaptar scripts para SpokenSQuAD ou VoxPopuli
3. **Ajustar hiperparâmetros**: Editar `configs/training.yaml`
4. **Experimentar arquiteturas**: Mudar entre MLP e Transformer no `configs/adapter.yaml`

## Referências

- Paper: "Speech Retrieval-Augmented Generation without Automatic Speech Recognition" (2412.16500v3)
- LibriSpeech: https://huggingface.co/datasets/librispeech_asr

