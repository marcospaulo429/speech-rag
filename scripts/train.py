"""Training script for Speech Retriever"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.data import SpeechDataset
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
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SpeechDataset(
        dataset_name=config["data"]["dataset_name"],
        dataset_config=config["data"].get("dataset_config"),
        split="train",
        sample_rate=config["data"]["sample_rate"],
        max_audio_length=config["data"]["max_audio_length"],
        cache_dir=config["paths"].get("data_dir")
    )
    
    val_dataset = None
    try:
        val_dataset = SpeechDataset(
            dataset_name=config["data"]["dataset_name"],
            dataset_config=config["data"].get("dataset_config"),
            split="validation",
            sample_rate=config["data"]["sample_rate"],
            max_audio_length=config["data"]["max_audio_length"],
            cache_dir=config["paths"].get("data_dir")
        )
        print(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        print(f"Could not load validation dataset: {e}")
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Loss function
    loss_fn = DistillationLoss(
        loss_type=config["training"]["loss_type"]
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01)
    )
    
    # Create trainer
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
        project_name="speech-rag"
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        save_steps=config["training"].get("save_steps", 1000),
        eval_steps=config["training"].get("eval_steps", 500),
        resume_from=args.resume
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()

