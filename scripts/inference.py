"""Inference script for Speech Retriever"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from src.inference import SpeechRetriever


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Speech Retriever Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to adapter checkpoint"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to saved FAISS index (if exists)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory with audio files to index"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text query to search"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create retriever
    retriever = SpeechRetriever(
        text_encoder=text_encoder,
        speech_encoder=speech_encoder,
        adapter=adapter,
        device=device
    )
    
    # Build or load index
    if args.index and Path(args.index).exists():
        print(f"Loading index from {args.index}...")
        retriever.load_index(args.index)
    elif args.audio_dir:
        print(f"Building index from audio files in {args.audio_dir}...")
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flac"))
        
        if not audio_files:
            print(f"No audio files found in {args.audio_dir}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        retriever.build_index(audio_files)
        
        # Save index if path provided
        if args.index:
            retriever.save_index(args.index)
    else:
        print("Error: Either --index or --audio-dir must be provided")
        return
    
    # Search if query provided
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        results = retriever.search(args.query, k=args.k)
        
        print(f"\nTop {len(results)} results:")
        print("-" * 80)
        for result in results:
            print(f"Rank {result['rank']}: {result['audio_path']}")
            print(f"  Score: {result['score']:.4f}")
            if result.get('metadata'):
                print(f"  Metadata: {result['metadata']}")
            print()
        
        # Save results if output specified
        if args.output:
            output_data = {
                "query": args.query,
                "k": args.k,
                "results": results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output}")
    else:
        print("No query provided. Use --query to search.")
        print("Index is ready for queries.")


if __name__ == "__main__":
    main()

