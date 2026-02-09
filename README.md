# Speech RAG - Speech Retriever

Implementação do módulo de recuperação de fala (Speech Retriever) baseado no paper "Speech Retrieval-Augmented Generation without Automatic Speech Recognition".

## Descrição

Este projeto implementa um sistema que permite buscar trechos de áudio usando queries em texto, sem necessidade de transcrição. O sistema alinha o espaço latente do áudio ao espaço latente do modelo de texto pré-treinado através de um adaptador treinado via distilação.

## Arquitetura

- **Text Encoder**: Suporta dois modelos:
  - **E5-Mistral-7B-Instruct** (4096 dimensões) - padrão
  - **Qwen3-Embedding-0.6B** (1024 dimensões) - alternativa mais leve
- **Speech Encoder**: HuBERT-large (congelado, 1024 dimensões)
- **Speech Adapter**: Adaptador treinável que projeta embeddings de áudio (1024) para o espaço de texto (4096 ou 1024, dependendo do encoder escolhido)

## Dataset

Este projeto utiliza o **Spoken Squad Test** como dataset principal. O Spoken Squad é um dataset de question answering sobre áudio baseado no SQuAD, contendo:
- Passagens de áudio lidas
- Transcrições das passagens (ground truth)
- Perguntas sobre o conteúdo
- Respostas correspondentes

O dataset é carregado automaticamente via HuggingFace Datasets. Certifique-se de ter acesso ao dataset ou configure um dataset customizado no arquivo de configuração.

## Instalação

### Usando UV (Recomendado)

[UV](https://github.com/astral-sh/uv) é um gerenciador de pacotes Python extremamente rápido (10-100x mais rápido que pip).

#### Instalação Automática

```bash
# Linux/Mac
bash scripts/setup.sh
# ou
./scripts/setup.sh

# Windows
setup.bat
```

#### Instalação Manual

```bash
# 1. Instalar uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Criar ambiente virtual
uv venv

# 3. Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# 4. Instalar dependências
uv pip install -e .

# 5. (Opcional) Instalar dependências de desenvolvimento
uv pip install -e ".[dev]"

# 6. (Opcional) Instalar com suporte GPU
uv pip install -e ".[gpu]"
```

### Usando pip (Alternativa)

```bash
pip install -r requirements.txt
```

## Uso

### Treinamento

#### Com UV (Recomendado)

```bash
# Usando script de conveniência
./scripts/run-train.sh --config config/config.yaml

# Ou diretamente com uv
uv run python scripts/train.py --config config/config.yaml
```

#### Com pip tradicional

```bash
python scripts/train.py --config config/config.yaml
```

### Inferência

#### Com UV (Recomendado)

```bash
# Usando script de conveniência
./scripts/run-inference.sh --checkpoint checkpoints/adapter.pt --query "sua query aqui"

# Ou diretamente com uv
uv run python scripts/inference.py --checkpoint checkpoints/adapter.pt --query "sua query aqui"
```

#### Com pip tradicional

```bash
python scripts/inference.py --checkpoint checkpoints/adapter.pt --query "sua query aqui"
```

## Estrutura

- `src/models/`: Modelos (text encoder, speech encoder, adapter)
- `src/data/`: Dataset loaders e pré-processamento
- `src/training/`: Trainer e losses
- `src/inference/`: Sistema de recuperação
- `scripts/`: Scripts de treinamento e inferência

## Documentação Adicional

- [Guia Completo de UV](UV_GUIDE.md) - Instruções detalhadas sobre uso do UV
- [Exemplos de Uso](EXAMPLE.md) - Exemplos práticos de treinamento e inferência
