"""Trainer para treinamento do adaptador."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
from tqdm import tqdm

from ..adapter.adapter import SpeechAdapter
from ..adapter.distillation_loss import DistillationLoss
from .evaluator import AdapterEvaluator


class AdapterTrainer:
    """Trainer para treinamento do adaptador por distilação."""
    
    def __init__(
        self,
        adapter: SpeechAdapter,
        loss_fn: DistillationLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        evaluator: Optional[AdapterEvaluator] = None
    ):
        """
        Inicializa trainer.
        
        Args:
            adapter: Adaptador a ser treinado
            loss_fn: Função de loss
            optimizer: Optimizer
            scheduler: Learning rate scheduler opcional
            device: Device
            evaluator: Evaluator opcional
        """
        self.adapter = adapter
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = evaluator or AdapterEvaluator()
        
        # Move adapter para device
        self.adapter = self.adapter.to(self.device)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        speech_encoder: Any,
        text_embedder: Any,
        progress_bar: bool = True
    ) -> Dict[str, float]:
        """
        Treina uma época.
        
        Args:
            dataloader: DataLoader de treinamento
            speech_encoder: Encoder de fala (frozen)
            text_embedder: Embedder de texto (frozen)
            progress_bar: Se True, mostra progress bar
        
        Returns:
            Dicionário com métricas
        """
        self.adapter.train()
        
        total_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(dataloader) if progress_bar else dataloader
        
        for batch in iterator:
            audios = batch['audio'].to(self.device)
            texts = batch['text']
            
            # Extrai embeddings de áudio
            with torch.no_grad():
                speech_embeddings = speech_encoder.encode(audios, normalize=False)
            
            # Extrai embeddings de texto
            with torch.no_grad():
                text_embeddings = text_embedder.encode(texts, normalize=False)
            
            # Projeta via adaptador
            projected_embeddings = self.adapter(speech_embeddings)
            
            # Calcula loss
            loss = self.loss_fn(projected_embeddings, text_embeddings, normalize=True)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def validate(
        self,
        dataloader: DataLoader,
        speech_encoder: Any,
        text_embedder: Any
    ) -> Dict[str, float]:
        """
        Valida em conjunto de validação.
        
        Args:
            dataloader: DataLoader de validação
            speech_encoder: Encoder de fala
            text_embedder: Embedder de texto
        
        Returns:
            Dicionário com métricas
        """
        self.adapter.eval()
        
        total_loss = 0.0
        all_projected = []
        all_text = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                audios = batch['audio'].to(self.device)
                texts = batch['text']
                
                # Embeddings
                speech_embeddings = speech_encoder.encode(audios, normalize=False)
                text_embeddings = text_embedder.encode(texts, normalize=False)
                
                # Projeção
                projected_embeddings = self.adapter(speech_embeddings)
                
                # Loss
                loss = self.loss_fn(projected_embeddings, text_embeddings, normalize=True)
                total_loss += loss.item()
                
                all_projected.append(projected_embeddings.cpu())
                all_text.append(text_embeddings.cpu())
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Métricas adicionais
        all_projected = torch.cat(all_projected, dim=0)
        all_text = torch.cat(all_text, dim=0)
        metrics = self.evaluator.evaluate(
            torch.zeros_like(all_projected),  # Não usado
            all_text,
            all_projected
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Salva checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'adapter_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path: str):
        """Carrega checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint.get('metrics', {})

