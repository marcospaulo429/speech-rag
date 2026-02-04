"""Gerenciamento de índices FAISS: salvar e carregar."""

import faiss
import numpy as np
import pickle
import os
from typing import Optional, Dict, Any
from datetime import datetime


class IndexManager:
    """Gerencia persistência de índices FAISS."""
    
    def __init__(self, index_dir: str = "indices"):
        """
        Inicializa o gerenciador de índices.
        
        Args:
            index_dir: Diretório para salvar/carregar índices
        """
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
    
    def save_index(
        self,
        index: faiss.Index,
        metadata: Dict[str, Any],
        name: str,
        version: Optional[str] = None
    ) -> str:
        """
        Salva índice FAISS e metadados.
        
        Args:
            index: Índice FAISS
            metadata: Metadados do índice (ex: mapeamento de IDs)
            name: Nome do índice
            version: Versão do índice (None = timestamp)
        
        Returns:
            Caminho do arquivo salvo
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Nome do arquivo
        index_filename = f"{name}_{version}.index"
        metadata_filename = f"{name}_{version}.metadata"
        
        index_path = os.path.join(self.index_dir, index_filename)
        metadata_path = os.path.join(self.index_dir, metadata_filename)
        
        # Salva índice
        faiss.write_index(index, index_path)
        
        # Salva metadados
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return index_path
    
    def load_index(
        self,
        name: str,
        version: Optional[str] = None
    ) -> tuple:
        """
        Carrega índice FAISS e metadados.
        
        Args:
            name: Nome do índice
            version: Versão do índice (None = mais recente)
        
        Returns:
            Tupla (índice, metadados)
        """
        if version is None:
            # Encontra versão mais recente
            files = [f for f in os.listdir(self.index_dir) if f.startswith(f"{name}_") and f.endswith(".index")]
            if not files:
                raise FileNotFoundError(f"Índice não encontrado: {name}")
            
            # Extrai versões e pega a mais recente
            versions = [f.replace(f"{name}_", "").replace(".index", "") for f in files]
            version = max(versions)
        
        index_filename = f"{name}_{version}.index"
        metadata_filename = f"{name}_{version}.metadata"
        
        index_path = os.path.join(self.index_dir, index_filename)
        metadata_path = os.path.join(self.index_dir, metadata_filename)
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Índice não encontrado: {index_path}")
        
        # Carrega índice
        index = faiss.read_index(index_path)
        
        # Carrega metadados
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata
    
    def list_indices(self) -> Dict[str, list]:
        """
        Lista todos os índices disponíveis.
        
        Returns:
            Dicionário {nome: [versões]}
        """
        indices = {}
        
        for filename in os.listdir(self.index_dir):
            if filename.endswith(".index"):
                # Extrai nome e versão
                parts = filename.replace(".index", "").split("_")
                if len(parts) >= 2:
                    version = "_".join(parts[-2:])  # Assume formato timestamp
                    name = "_".join(parts[:-2])
                    
                    if name not in indices:
                        indices[name] = []
                    indices[name].append(version)
        
        return indices

