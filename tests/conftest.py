"""Configuração do pytest para os testes."""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

