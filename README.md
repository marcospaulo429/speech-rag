# SpeechRAG: Speech Retrieval-Augmented Generation without ASR

Implementação modular do framework SpeechRAG para question answering sobre dados de fala sem usar Automatic Speech Recognition (ASR).

## Arquitetura

O SpeechRAG consiste em três componentes principais:

1. **Speech Retriever**: Recupera passagens de áudio a partir de consultas em texto
2. **Speech Adapter**: Adaptador leve que projeta embeddings de fala no espaço de embeddings de texto
3. **Audio-conditioned Generator**: SLM que gera respostas condicionadas em passagens de áudio

## Estrutura do Projeto

```
speech-rag/
├── src/
│   ├── speech_encoder/      # Módulo 1: Speech Encoder
│   ├── text_embedder/       # Módulo 2: Text Embedder
│   ├── adapter/             # Módulo 3: Speech Adapter (implementado do zero)
│   ├── retriever/           # Módulo 4: Cross-modal Retriever
│   ├── generator/            # Módulo 5: SLM Generator
│   ├── training/            # Módulo 6: Training Pipeline
│   ├── baselines/            # Sistemas baseline
│   ├── pipeline/             # Módulo 7: End-to-End Pipeline
│   └── utils/               # Utilitários (normalização, chunking)
├── tests/                   # Testes unitários
├── configs/                 # Configurações YAML
├── scripts/                 # Scripts de treinamento e inferência
└── indices/                 # Índices FAISS salvos
```

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Ou usando uv (recomendado)
uv pip install -r requirements.txt
```

## Uso

### Testes Rápidos com LibriSpeech

Para testar o sistema completo com LibriSpeech, consulte o guia detalhado:

**[TESTING.md](TESTING.md)** - Guia completo de testes com LibriSpeech

**Teste rápido automatizado:**
```bash
bash scripts/quick_test.sh
```

### Treinamento do Adaptador

```bash
# Com LibriSpeech (split do Hugging Face)
python scripts/train.py \
    --config configs/training.yaml \
    --train_data train-clean-100 \
    --val_data dev-clean \
    --train_limit 1000

# Ou com arquivo JSON
python scripts/train.py \
    --config configs/training.yaml \
    --train_data data/train.json \
    --val_data data/val.json
```

### Indexação de Passagens

```bash
python scripts/index_librispeech.py \
    --split test-clean \
    --adapter_checkpoint checkpoints/checkpoint_epoch_15.pt \
    --index_name librispeech_index
```

### Inferência

```bash
python scripts/inference.py --question "What is this audio about?"
```

## Componentes Implementados

### Módulo 1: Speech Encoder
- Wrapper para HuBERT/Wav2Vec2 (Hugging Face)
- Pooling (mean, max, attention)
- Pré-processamento de áudio

### Módulo 2: Text Embedder
- Wrapper para modelos E5/Sentence-BERT (Hugging Face)
- Normalização de embeddings

### Módulo 3: Speech Adapter (Implementado do Zero)
- Arquiteturas MLP e Transformer
- Loss de distilação (MSE, Cosine, Combinada)

### Módulo 4: Cross-modal Retriever
- Indexação FAISS
- Busca por similaridade
- Persistência de índices

### Módulo 5: SLM Generator
- Wrapper para Qwen-Audio/SpeechSERSE
- Condicionamento em áudio

### Módulo 6: Training Pipeline
- DataLoader para pares (áudio, transcrição)
- Trainer com loops de treinamento
- Data augmentation opcional
- Avaliação durante treinamento

### Módulo 7: End-to-End Pipeline
- Pipeline completo SpeechRAG
- Integração retriever + generator

## Estratégia de Modelos

| Componente        | Origem                       | Status    |
|------------------|------------------------------|-----------|
| Speech Encoder   | Hugging Face (HuBERT/Wav2Vec2) | Frozen    |
| Text Embedder    | Hugging Face (E5/Sentence-BERT) | Frozen    |
| **Speech Adapter** | **Implementado do zero**     | **Treinável** |
| SLM Generator    | Hugging Face (Qwen-Audio)    | Frozen    |

## Testes

```bash
pytest tests/
```

## Configurações

Arquivos de configuração YAML estão em `configs/`:
- `speech_encoder.yaml`: Modelo e pooling
- `text_embedder.yaml`: Modelo de embedding
- `adapter.yaml`: Arquitetura do adaptador
- `retriever.yaml`: Configurações de retrieval
- `generator.yaml`: Modelo SLM
- `training.yaml`: Hiperparâmetros de treinamento

## Referências

- Paper: "Speech Retrieval-Augmented Generation without Automatic Speech Recognition" (2412.16500v3)
