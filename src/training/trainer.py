"""Trainer for Speech Adapter"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
from tqdm import tqdm
import wandb
from pathlib import Path
import time

# Add src to path if needed, though usually handled by execution context
import sys

from ..models import TextEncoder, SpeechEncoder, SpeechAdapter
from ..data import SpeechDataset
from .losses import DistillationLoss


class Trainer:
    """
    Trainer for Speech Adapter using distillation.
    Only the adapter is trained; text and speech encoders are frozen.
    """
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        speech_encoder: SpeechEncoder,
        adapter: SpeechAdapter,
        train_dataset: SpeechDataset,
        val_dataset: Optional[SpeechDataset] = None,
        loss_fn: Optional[DistillationLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs/",
        use_wandb: bool = True,
        project_name: str = "speech-rag",
        collate_fn: Optional[Callable] = None,  # FIXED: Added default value = None
        config: Optional[Dict] = None  # Config dict for logging hyperparameters
    ):
        self.text_encoder = text_encoder.to(device)
        self.speech_encoder = speech_encoder.to(device)
        self.adapter = adapter.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collate_fn = collate_fn  # Store it for use in DataLoader
        
        # Loss function
        if loss_fn is None:
            self.loss_fn = DistillationLoss(loss_type="mse")
        else:
            self.loss_fn = loss_fn.to(device)
        
        # Optimizer (only for adapter parameters)
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.adapter.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            # Log config if provided, otherwise empty dict
            wandb_config = config if config is not None else {}
            wandb.init(project=project_name, config=wandb_config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_patience = None
        self.early_stopping_min_delta = 0.0
        self.early_stopping_counter = 0
    
    def _compute_entropy(self, embeddings: torch.Tensor) -> float:
        """Compute entropy of normalized embeddings."""
        # Normalize embeddings
        normalized = F.normalize(embeddings, p=2, dim=-1)
        # Compute dot products (similarity matrix)
        similarity_matrix = torch.mm(normalized, normalized.t())
        # Convert to probabilities (softmax over each row)
        probs = F.softmax(similarity_matrix, dim=-1)
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        return entropy
    
    def train_epoch(self, dataloader: DataLoader, gradient_accumulation_steps: int = 1, log_batch_frequency: int = 1) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation support."""
        self.adapter.train()
        self.text_encoder.eval()
        self.speech_encoder.eval()
        
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        # Metrics accumulation
        total_grad_norm = 0.0
        total_audio_norm = 0.0
        total_text_norm = 0.0
        total_alignment_distance = 0.0
        total_entropy = 0.0
        grad_norm_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        epoch_start_time = time.time()
        
        self.optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            audio = batch["audio"].to(self.device)
            # Texts are usually a list of strings, so we don't .to(device) them directly
            # The TextEncoder handles tokenization and device movement internally
            texts = batch["text"]
            
            # Forward pass
            # 1. Get speech representations from HuBERT
            with torch.no_grad():
                # Note: Ensure your speech_encoder.encode handles padding masks if needed
                speech_reprs = self.speech_encoder.encode(audio)
            
            # 2. Get text embeddings from E5-Mistral (ground truth)
            with torch.no_grad():
                text_embeddings = self.text_encoder.encode(texts, device=self.device)
            
            # 3. Get audio embeddings from adapter (trainable)
            audio_embeddings = self.adapter(speech_reprs)
            
            # 4. Compute loss and normalize by accumulation steps
            loss = self.loss_fn(audio_embeddings, text_embeddings)
            loss = loss / gradient_accumulation_steps  # Normalize loss
            
            # 5. Backward pass (accumulate gradients)
            loss.backward()
            
            # Metrics (use non-normalized loss for logging)
            with torch.no_grad():
                similarity = self.loss_fn.compute_similarity(
                    audio_embeddings, text_embeddings
                ).mean().item()
                
                # Embedding statistics
                audio_norm = audio_embeddings.norm(dim=-1).mean().item()
                text_norm = text_embeddings.norm(dim=-1).mean().item()
                alignment_distance = (audio_embeddings - text_embeddings).norm(dim=-1).mean().item()
                
                # Entropy
                entropy = self._compute_entropy(audio_embeddings)
            
            accumulated_loss += loss.item() * gradient_accumulation_steps  # Denormalize for logging
            total_similarity += similarity
            total_audio_norm += audio_norm
            total_text_norm += text_norm
            total_alignment_distance += alignment_distance
            total_entropy += entropy
            num_batches += 1
            
            # Update optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Compute gradient norm before step
                grad_norm = 0.0
                for param in self.adapter.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                total_grad_norm += grad_norm
                grad_norm_count += 1
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Denormalize loss for batch logging
                batch_loss = loss.item() * gradient_accumulation_steps
                
                # Logging por batch (se configurado)
                if self.use_wandb and (self.global_step % log_batch_frequency == 0 or log_batch_frequency == 1):
                    wandb.log({
                        "batch/loss": batch_loss,
                        "batch/similarity": similarity,
                        "batch/grad_norm": grad_norm,
                        "batch/audio_norm": audio_norm,
                        "batch/text_norm": text_norm,
                        "batch/alignment_distance": alignment_distance,
                        "batch/entropy": entropy,
                        "batch/learning_rate": current_lr,
                        "batch/step": self.global_step,
                        "batch/batch_idx": batch_idx + 1,
                        "batch/epoch": self.epoch
                    })
                
                # Update progress bar with accumulated loss
                avg_loss_so_far = accumulated_loss / num_batches
                progress_bar.set_postfix({
                    "loss": f"{avg_loss_so_far:.4f}",
                    "sim": f"{similarity:.4f}"
                })
                
                # Logging agregado a cada 100 steps (mantido para compatibilidade)
                if self.use_wandb and self.global_step % 100 == 0:
                    avg_grad_norm = total_grad_norm / grad_norm_count if grad_norm_count > 0 else 0.0
                    wandb.log({
                        "train/loss": avg_loss_so_far,
                        "train/similarity": similarity,
                        "train/step": self.global_step,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": avg_grad_norm,
                        "train/audio_norm": total_audio_norm / num_batches,
                        "train/text_norm": total_text_norm / num_batches,
                        "train/alignment_distance": total_alignment_distance / num_batches,
                        "train/entropy": total_entropy / num_batches
                    })
                    # Reset accumulators for next logging period
                    total_grad_norm = 0.0
                    grad_norm_count = 0
        
        # Handle remaining gradients if batch doesn't divide evenly
        if num_batches % gradient_accumulation_steps != 0:
            # Compute gradient norm before step
            grad_norm = 0.0
            for param in self.adapter.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm
            grad_norm_count += 1
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        
        epoch_time = time.time() - epoch_start_time
        
        avg_loss = accumulated_loss / num_batches if num_batches > 0 else 0.0
        avg_similarity = total_similarity / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / grad_norm_count if grad_norm_count > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "similarity": avg_similarity,
            "grad_norm": avg_grad_norm,
            "audio_norm": total_audio_norm / num_batches if num_batches > 0 else 0.0,
            "text_norm": total_text_norm / num_batches if num_batches > 0 else 0.0,
            "alignment_distance": total_alignment_distance / num_batches if num_batches > 0 else 0.0,
            "entropy": total_entropy / num_batches if num_batches > 0 else 0.0,
            "epoch_time": epoch_time
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.adapter.eval()
        self.text_encoder.eval()
        self.speech_encoder.eval()
        
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        
        # Metrics accumulation
        total_audio_norm = 0.0
        total_text_norm = 0.0
        total_alignment_distance = 0.0
        total_entropy = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move to device
                audio = batch["audio"].to(self.device)
                texts = batch["text"]
                
                # Forward pass
                speech_reprs = self.speech_encoder.encode(audio)
                text_embeddings = self.text_encoder.encode(texts, device=self.device)
                audio_embeddings = self.adapter(speech_reprs)
                
                # Loss
                loss = self.loss_fn(audio_embeddings, text_embeddings)
                similarity = self.loss_fn.compute_similarity(
                    audio_embeddings, text_embeddings
                ).mean().item()
                
                # Embedding statistics
                audio_norm = audio_embeddings.norm(dim=-1).mean().item()
                text_norm = text_embeddings.norm(dim=-1).mean().item()
                alignment_distance = (audio_embeddings - text_embeddings).norm(dim=-1).mean().item()
                
                # Entropy
                entropy = self._compute_entropy(audio_embeddings)
                
                total_loss += loss.item()
                total_similarity += similarity
                total_audio_norm += audio_norm
                total_text_norm += text_norm
                total_alignment_distance += alignment_distance
                total_entropy += entropy
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_similarity = total_similarity / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "similarity": avg_similarity,
            "audio_norm": total_audio_norm / num_batches if num_batches > 0 else 0.0,
            "text_norm": total_text_norm / num_batches if num_batches > 0 else 0.0,
            "alignment_distance": total_alignment_distance / num_batches if num_batches > 0 else 0.0,
            "entropy": total_entropy / num_batches if num_batches > 0 else 0.0
        }
    
    def save_checkpoint(
        self,
        checkpoint_name: str = "checkpoint.pt",
        is_best: bool = False
    ):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / checkpoint_name
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "adapter_state_dict": self.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    
    def train(
        self,
        num_epochs: int,
        batch_size: int = 16,
        save_steps: int = 1000,
        eval_steps: int = 500,
        resume_from: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        log_batch_frequency: int = 1
    ):
        """Main training loop with early stopping and gradient accumulation."""
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed from checkpoint: {resume_from}")
        
        # Set early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        
        # Create dataloaders
        # FIXED: Use self.collate_fn instead of calling non-existent get_collate_fn()
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,  
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn, # Use same collate_fn for val
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, gradient_accumulation_steps, log_batch_frequency)
            
            print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                  f"Similarity: {train_metrics['similarity']:.4f}")
            
            # Validate every epoch if validation dataset exists
            # (Early stopping requires validation every epoch)
            if val_loader:
                val_metrics = self.validate(val_loader)
                print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                      f"Similarity: {val_metrics['similarity']:.4f}")
                
                # Compute train/val gap
                train_val_gap = train_metrics["loss"] - val_metrics["loss"]
                
                # Log metrics per epoch with epoch/ prefix
                if self.use_wandb:
                    wandb.log({
                        # Train metrics
                        "epoch/train/loss": train_metrics["loss"],
                        "epoch/train/similarity": train_metrics["similarity"],
                        "epoch/train/grad_norm": train_metrics.get("grad_norm", 0.0),
                        "epoch/train/audio_norm": train_metrics.get("audio_norm", 0.0),
                        "epoch/train/text_norm": train_metrics.get("text_norm", 0.0),
                        "epoch/train/alignment_distance": train_metrics.get("alignment_distance", 0.0),
                        "epoch/train/entropy": train_metrics.get("entropy", 0.0),
                        "epoch/train/time": train_metrics.get("epoch_time", 0.0),
                        # Val metrics
                        "epoch/val/loss": val_metrics["loss"],
                        "epoch/val/similarity": val_metrics["similarity"],
                        "epoch/val/audio_norm": val_metrics.get("audio_norm", 0.0),
                        "epoch/val/text_norm": val_metrics.get("text_norm", 0.0),
                        "epoch/val/alignment_distance": val_metrics.get("alignment_distance", 0.0),
                        "epoch/val/entropy": val_metrics.get("entropy", 0.0),
                        # Comparative metrics
                        "epoch/train_val_gap": train_val_gap,
                        "epoch/best_val_loss": self.best_val_loss,
                        "epoch/early_stopping_counter": self.early_stopping_counter,
                        "epoch/epoch": epoch
                    })
                
                # Save best model and check for improvement
                is_best = val_metrics["loss"] < (self.best_val_loss - early_stopping_min_delta)
                if is_best:
                    self.best_val_loss = val_metrics["loss"]
                    self.early_stopping_counter = 0
                    self.save_checkpoint("best_model.pt", is_best=True)
                    print(f"✓ Best model saved! Val loss: {val_metrics['loss']:.4f}")
                else:
                    # Only increment counter if early stopping is enabled
                    if early_stopping_patience is not None:
                        self.early_stopping_counter += 1
                
                # Check early stopping (only if enabled)
                if early_stopping_patience is not None and self.early_stopping_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                    break
            
            # Save checkpoint
            if self.global_step % save_steps == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", is_best=False)
        
        # Final save
        self.save_checkpoint("final_checkpoint.pt", is_best=False)
        print("Training completed!")