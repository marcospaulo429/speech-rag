"""Training script for Speech Retriever"""

import argparse
import yaml
import torch
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
# Added speech_collate_fn import
from src.data import SpeechDataset, speech_collate_fn
from src.training import Trainer, DistillationLoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Speech Retriever")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Fallback for Mac M1/M2 if cuda is not available
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    text_encoder = TextEncoder(
        model_name=config["models"]["text_encoder"],
        freeze=True
    )
    
    speech_encoder = SpeechEncoder(
        model_name=config["models"]["speech_encoder"],
        freeze=True
    )
    
    # Get embedding dimensions
    text_embedding_dim = text_encoder.embedding_dim
    speech_hidden_dim = speech_encoder.hidden_size
    
    # Create adapter
    adapter = SpeechAdapter(
        input_dim=speech_hidden_dim,
        output_dim=text_embedding_dim,
        downsample_factor=4
    )
    
    print(f"Text embedding dim: {text_embedding_dim}")
    print(f"Speech hidden dim: {speech_hidden_dim}")
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
    # --- LOAD DATASETS (UPDATED) ---
    print("Loading datasets...")
    
    # Get data root from config, default to local 'data' folder
    data_root = config["paths"].get("data_dir", "data")
    
    # 1. Training Set
    # Maps to: data/spoken_train-v1.1.json AND data/train_wav
    train_metadata = os.path.join(data_root, "spoken_train-v1.1.json")
    train_audio_dir = os.path.join(data_root, "train_wav")
    
    if not os.path.exists(train_metadata) or not os.path.exists(train_audio_dir):
        raise FileNotFoundError(
            f"Training data not found at {train_metadata} or {train_audio_dir}"
        )

    train_dataset = SpeechDataset(
        metadata_path=train_metadata,
        audio_dir=train_audio_dir,
        sample_rate=config["data"]["sample_rate"],
        max_audio_length=config["data"]["max_audio_length"]
    )
    print(f"Training dataset size: {len(train_dataset)}")

    # 2. Validation Set
    # Maps to: data/spoken_test-v1.1.json AND data/dev_wav
    val_dataset = None
    val_metadata = os.path.join(data_root, "spoken_test-v1.1.json")
    val_audio_dir = os.path.join(data_root, "dev_wav")

    if os.path.exists(val_metadata) and os.path.exists(val_audio_dir):
        try:
            val_dataset = SpeechDataset(
                metadata_path=val_metadata,
                audio_dir=val_audio_dir,
                sample_rate=config["data"]["sample_rate"],
                max_audio_length=config["data"]["max_audio_length"]
            )
            print(f"Validation dataset size: {len(val_dataset)}")
        except Exception as e:
            print(f"Could not load validation dataset: {e}")
    else:
        print(f"Warning: Validation data not found at {val_audio_dir}. Skipping validation.")

    # Loss function
    loss_fn = DistillationLoss(
        loss_type=config["training"]["loss_type"]
    )
    
    # Optimizer
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    optimizer_type = config["training"].get("optimizer", "adamw").lower()
    
    # Get beta parameters for Adam/AdamW
    beta1 = float(config["training"].get("beta1", 0.9))
    beta2 = float(config["training"].get("beta2", 0.999))
    
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            adapter.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'adamw'")
    
    # Create trainer
    # NOTE: We pass 'collate_fn' here. Ensure your Trainer class accepts it!
    trainer = Trainer(
        text_encoder=text_encoder,
        speech_encoder=speech_encoder,
        adapter=adapter,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        output_dir=config["paths"]["output_dir"],
        use_wandb=not args.no_wandb,
        project_name="speech-rag",
        collate_fn=speech_collate_fn,  # <--- CRITICAL for padding
        config=config  # Pass config for wandb logging
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        save_steps=config["training"].get("save_steps", 1000),
        eval_steps=config["training"].get("eval_steps", 500),
        resume_from=args.resume,
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        early_stopping_patience=config["training"].get("early_stopping_patience"),
        early_stopping_min_delta=config["training"].get("early_stopping_min_delta", 0.0),
        log_batch_frequency=config["training"].get("log_batch_frequency", 1)
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()