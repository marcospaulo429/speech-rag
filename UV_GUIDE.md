# Guia Rápido - UV

Este guia explica como usar `uv` para gerenciar o projeto Speech RAG.

## O que é UV?

[UV](https://github.com/astral-sh/uv) é um gerenciador de pacotes Python extremamente rápido (10-100x mais rápido que pip), escrito em Rust. Ele oferece:

- ⚡ **Velocidade**: Instalação de pacotes muito mais rápida
- 🔒 **Reproduzibilidade**: Lock files para dependências exatas
- 🛠️ **Gerenciamento de ambiente**: Criação automática de ambientes virtuais
- 📦 **Compatibilidade**: Suporta `pyproject.toml` (padrão moderno)

## Instalação do UV

### Linux/Mac

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

Após instalar, adicione ao PATH ou reinicie o terminal.

## Comandos Principais

### Setup Inicial

```bash
# Criar ambiente virtual
uv venv

# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# Instalar projeto em modo desenvolvimento
uv pip install -e .
```

### Gerenciamento de Dependências

```bash
# Adicionar nova dependência
uv add package-name

# Adicionar dependência de desenvolvimento
uv add --dev package-name

# Remover dependência
uv remove package-name

# Atualizar todas as dependências
uv pip install --upgrade -e .

# Instalar grupo opcional de dependências
uv pip install -e ".[dev]"      # Ferramentas de desenvolvimento
uv pip install -e ".[gpu]"     # Suporte GPU
uv pip install -e ".[all]"      # Todas as dependências
```

### Execução

```bash
# Executar script Python com uv
uv run python scripts/train.py --config config/config.yaml

# Executar com ambiente virtual automático (sem ativar)
uv run --python 3.10 python scripts/train.py
```

### Outros Comandos Úteis

```bash
# Ver versão do uv
uv --version

# Ver dependências instaladas
uv pip list

# Verificar dependências
uv pip check

# Sincronizar com lock file (se existir)
uv pip sync
```

## Scripts de Conveniência

O projeto inclui scripts que facilitam o uso:

```bash
# Setup inicial
./scripts/setup.sh          # Linux/Mac
scripts\setup.bat            # Windows

# Executar treinamento
./scripts/run-train.sh --config config/config.yaml

# Executar inferência
./scripts/run-inference.sh --checkpoint model.pt --query "texto"
```

## Estrutura do Projeto com UV

```
speech-rag/
├── pyproject.toml          # Configuração do projeto e dependências
├── uv.lock                 # Lock file (gerado automaticamente)
├── .venv/                   # Ambiente virtual (criado por uv venv)
├── .python-version         # Versão do Python (para pyenv)
└── requirements.txt         # Mantido para compatibilidade com pip
```

## Migração de pip para uv

Se você já estava usando pip:

1. **Remover ambiente virtual antigo** (opcional):
   ```bash
   rm -rf venv/ .venv/
   ```

2. **Criar novo ambiente com uv**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Instalar dependências**:
   ```bash
   uv pip install -e .
   ```

## Vantagens sobre pip

| Recurso | pip | uv |
|---------|-----|-----|
| Velocidade | ⭐ | ⭐⭐⭐⭐⭐ |
| Lock files | ❌ | ✅ |
| Gerenciamento de ambiente | Manual | Automático |
| pyproject.toml | Parcial | Completo |
| Resolução de dependências | Básica | Avançada |

## Troubleshooting

### UV não encontrado

```bash
# Verificar instalação
which uv  # Linux/Mac
where uv  # Windows

# Adicionar ao PATH manualmente
export PATH="$HOME/.cargo/bin:$PATH"  # Linux/Mac
```

### Problemas com dependências

```bash
# Limpar cache
uv cache clean

# Reinstalar tudo
rm -rf .venv
uv venv
uv pip install -e .
```

### Versão do Python

O projeto requer Python >= 3.8. Verifique sua versão:

```bash
python --version
```

Se necessário, instale uma versão compatível ou use `uv` com versão específica:

```bash
uv run --python 3.10 python scripts/train.py
```

## Mais Informações

- [Documentação oficial do UV](https://github.com/astral-sh/uv)
- [Guia de migração](https://github.com/astral-sh/uv/blob/main/README.md)

