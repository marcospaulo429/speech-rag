# Exemplo de Uso - Speech Retriever

## Instalação

### Usando UV (Recomendado)

```bash
# Instalação automática
./scripts/setup.sh

# Ou manualmente
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Usando pip (Alternativa)

```bash
pip install -r requirements.txt
```

## Treinamento

### 1. Preparar Configuração

Edite `config/config.yaml` conforme necessário, especialmente:
- `dataset_name`: Nome do dataset HuggingFace (padrão: "spoken_squad")
- `audio_column`: Nome da coluna de áudio (padrão: "audio")
- `text_column`: Nome da coluna de texto (padrão: "passage_text")
- `batch_size`: Tamanho do batch (ajuste conforme memória disponível)
- `learning_rate`: Taxa de aprendizado

### 2. Executar Treinamento

#### Com UV

```bash
# Usando script de conveniência
./scripts/run-train.sh --config config/config.yaml

# Ou diretamente
uv run python scripts/train.py --config config/config.yaml
```

#### Com pip tradicional

```bash
python scripts/train.py --config config/config.yaml
```

### 3. Treinamento com Resumo

#### Com UV

```bash
./scripts/run-train.sh \
    --config config/config.yaml \
    --resume outputs/checkpoint_epoch_5.pt
```

#### Com pip tradicional

```bash
python scripts/train.py \
    --config config/config.yaml \
    --resume outputs/checkpoint_epoch_5.pt
```

## Inferência

### 1. Construir Índice de Áudio

Primeiro, você precisa construir um índice a partir de arquivos de áudio:

#### Com UV

```bash
./scripts/run-inference.sh \
    --checkpoint outputs/best_model.pt \
    --audio-dir /path/to/audio/files \
    --index indices/audio_index.faiss
```

#### Com pip tradicional

```bash
python scripts/inference.py \
    --checkpoint outputs/best_model.pt \
    --audio-dir /path/to/audio/files \
    --index indices/audio_index.faiss
```

### 2. Buscar com Query em Texto

#### Com UV

```bash
./scripts/run-inference.sh \
    --checkpoint outputs/best_model.pt \
    --index indices/audio_index.faiss \
    --query "What is machine learning?" \
    --k 10 \
    --output results.json
```

#### Com pip tradicional

```bash
python scripts/inference.py \
    --checkpoint outputs/best_model.pt \
    --index indices/audio_index.faiss \
    --query "What is machine learning?" \
    --k 10 \
    --output results.json
```

## Uso Programático

### Treinamento

```python
from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.data import SpeechDataset
from src.training import Trainer, DistillationLoss
import torch

# Carregar modelos
text_encoder = TextEncoder(freeze=True)
speech_encoder = SpeechEncoder(freeze=True)
adapter = SpeechAdapter(
    input_dim=1024,  # HuBERT hidden size
    output_dim=4096  # E5-Mistral embedding dim
)

# Carregar dataset
train_dataset = SpeechDataset(
    dataset_name="spoken_squad",
    dataset_config=None,
    split="train",
    audio_column="audio",
    text_column="passage_text"
)

# Criar trainer
trainer = Trainer(
    text_encoder=text_encoder,
    speech_encoder=speech_encoder,
    adapter=adapter,
    train_dataset=train_dataset,
    device="cuda"
)

# Treinar
trainer.train(num_epochs=10, batch_size=16)
```

### Inferência

```python
from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.inference import SpeechRetriever
import torch
from pathlib import Path

# Carregar modelos
text_encoder = TextEncoder(freeze=True)
speech_encoder = SpeechEncoder(freeze=True)
adapter = SpeechAdapter(input_dim=1024, output_dim=4096)

# Carregar checkpoint
checkpoint = torch.load("outputs/best_model.pt")
adapter.load_state_dict(checkpoint["adapter_state_dict"])

# Criar retriever
retriever = SpeechRetriever(
    text_encoder=text_encoder,
    speech_encoder=speech_encoder,
    adapter=adapter
)

# Construir índice
audio_files = list(Path("audio/").glob("*.wav"))
retriever.build_index(audio_files)

# Buscar
results = retriever.search("What is artificial intelligence?", k=5)
for result in results:
    print(f"{result['rank']}: {result['audio_path']} (score: {result['score']:.4f})")
```

## Notas Importantes

1. **Memória**: O E5-Mistral-7B é um modelo grande (~14GB). Considere usar quantização ou gradient checkpointing se tiver limitações de memória.

2. **Dataset**: O Spoken Squad contém passagens de áudio com perguntas e respostas. As passagens podem ser mais longas que outros datasets, então ajuste `max_audio_length` se necessário.

3. **GPU**: Recomenda-se usar GPU para treinamento e inferência devido ao tamanho dos modelos.

4. **Checkpoints**: Os checkpoints são salvos automaticamente durante o treinamento. O melhor modelo é salvo como `best_model.pt`.

