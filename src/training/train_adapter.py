"""Script de treinamento do adaptador."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
from typing import Optional

from ..adapter.adapter import SpeechAdapter
from ..adapter.distillation_loss import DistillationLoss
from ..speech_encoder.speech_encoder import SpeechEncoder
from ..text_embedder.text_embedder import TextEmbedder
from .trainer import AdapterTrainer
from .data_loader import create_dataloader
from .augmentation import AudioAugmentation
from ..data.librispeech_loader import (
    load_librispeech_split,
    load_librispeech_from_json,
    get_librispeech_splits
)


def create_optimizer(adapter: SpeechAdapter, config: dict) -> optim.Optimizer:
    """Cria optimizer."""
    optimizer_name = config.get("optimizer", "adamw").lower()
    lr = float(config.get("learning_rate", 1e-4))  # Garante que é float
    weight_decay = float(config.get("weight_decay", 1e-5))  # Garante que é float
    
    if optimizer_name == "adamw":
        return optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer desconhecido: {optimizer_name}")


def create_scheduler(optimizer: optim.Optimizer, config: dict, num_epochs: int):
    """Cria learning rate scheduler."""
    scheduler_type = config.get("scheduler", "none").lower()
    scheduler_params = config.get("scheduler_params", {})
    
    if scheduler_type == "cosine":
        T_max = scheduler_params.get("T_max", num_epochs)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "reduce_on_plateau":
        patience = scheduler_params.get("patience", 3)
        factor = scheduler_params.get("factor", 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    else:
        return None


def load_data(data_path: str, limit: Optional[int] = None):
    """
    Carrega dados de treinamento/validação.
    
    Args:
        data_path: Caminho para JSON ou nome de split do LibriSpeech
        limit: Limitar número de amostras (para testes rápidos)
    
    Returns:
        Lista de dados
    """
    # Verifica se é um split do LibriSpeech
    librispeech_splits = get_librispeech_splits()
    
    if data_path in librispeech_splits:
        # É um split do LibriSpeech
        print(f"Carregando split do LibriSpeech: {data_path}")
        return load_librispeech_split(data_path, limit=limit)
    elif data_path.endswith('.json'):
        # É um arquivo JSON
        print(f"Carregando dados de JSON: {data_path}")
        return load_librispeech_from_json(data_path)
    else:
        raise ValueError(
            f"Formato de dados não reconhecido: {data_path}. "
            f"Use um split do LibriSpeech ({', '.join(librispeech_splits)}) "
            f"ou um arquivo JSON."
        )


def main():
    parser = argparse.ArgumentParser(description="Treinar Speech Adapter")
    parser.add_argument("--config", type=str, required=True, help="Caminho para config YAML")
    parser.add_argument("--train_data", type=str, required=True, 
                       help="Caminho para dados de treinamento (JSON) ou split do LibriSpeech")
    parser.add_argument("--val_data", type=str, 
                       help="Caminho para dados de validação (JSON) ou split do LibriSpeech")
    parser.add_argument("--train_limit", type=int, default=None,
                       help="Limitar número de amostras de treinamento (para testes)")
    parser.add_argument("--val_limit", type=int, default=None,
                       help="Limitar número de amostras de validação (para testes)")
    args = parser.parse_args()
    
    # Carrega configuração
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Carrega modelos (frozen)
    speech_encoder = SpeechEncoder.from_config(
        "configs/speech_encoder.yaml",
        device=device
    )
    text_embedder = TextEmbedder.from_config(
        "configs/text_embedder.yaml",
        device=device
    )
    
    # Cria adaptador
    adapter = SpeechAdapter.from_config("configs/adapter.yaml")
    
    # Loss
    loss_fn = DistillationLoss(
        loss_type=config.get("loss_type", "mse"),
        temperature=config.get("temperature", 1.0)
    )
    
    # Optimizer
    optimizer = create_optimizer(adapter, config)
    
    # Scheduler
    num_epochs = config.get("num_epochs", 15)
    scheduler = create_scheduler(optimizer, config, num_epochs)
    
    # Trainer
    trainer = AdapterTrainer(
        adapter=adapter,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Data loaders
    # Carrega dados reais
    train_data = load_data(args.train_data, limit=args.train_limit)
    val_data = load_data(args.val_data, limit=args.val_limit) if args.val_data else None
    
    augmentation = None
    if config.get("augmentation", {}).get("enabled", False):
        aug_config = config["augmentation"]
        augmentation = AudioAugmentation(
            noise_injection=aug_config.get("noise_injection", False),
            speed_variation=aug_config.get("speed_variation", False)
        )
    
    train_loader = create_dataloader(
        train_data,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        augmentation=augmentation
    )
    
    val_loader = None
    if val_data:
        val_loader = create_dataloader(
            val_data,
            batch_size=config.get("batch_size", 32),
            shuffle=False
        )
    
    # Treinamento
    save_dir = config.get("save_dir", "checkpoints")
    save_every = config.get("save_every", 5)
    eval_every = config.get("eval_every", 1)
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            speech_encoder,
            text_embedder
        )
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        if val_loader and (epoch + 1) % eval_every == 0:
            val_metrics = trainer.validate(
                val_loader,
                speech_encoder,
                text_embedder
            )
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}, "
                  f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
            
            # Scheduler step
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
            print(f"Checkpoint salvo: {checkpoint_path}")


if __name__ == "__main__":
    main()

