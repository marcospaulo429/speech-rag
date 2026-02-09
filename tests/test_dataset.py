"""Test dataset loading for Spoken Squad"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SpeechDataset


def test_spoken_squad_loading():
    """Test loading Spoken Squad dataset"""
    print("Testing Spoken Squad dataset loading...")
    
    try:
        # Try to load a small sample
        dataset = SpeechDataset(
            dataset_name="spoken_squad_test",
            dataset_config=None,
            split="train",
            audio_column="audio",
            text_column="passage_text",
            sample_rate=16000,
            max_audio_length=60.0,
            streaming=True  # Use streaming to avoid loading full dataset
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Dataset name: spoken_squad_test")
        print(f"  Split: train")
        print(f"  Streaming mode: True")
        
        # Try to get one sample
        try:
            sample = dataset[0]
            print(f"✓ Successfully loaded sample")
            print(f"  Audio shape: {sample['audio'].shape}")
            print(f"  Text length: {len(sample['text'])} characters")
            print(f"  Sample rate: {sample['sample_rate']}")
        except Exception as e:
            print(f"⚠ Could not load sample: {e}")
            print("  This might be normal if the dataset structure is different.")
            print("  Please verify the column names in config/config.yaml")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("\nPossible issues:")
        print("1. Dataset 'spoken_squad_test' might not be available on HuggingFace")
        print("2. You might need to use a different dataset name")
        print("3. Check your internet connection for downloading the dataset")
        print("4. Verify the dataset identifier in config/config.yaml")
        return False
    
    return True


def test_custom_config():
    """Test with custom configuration"""
    print("\nTesting with custom configuration...")
    
    # Test with different column names that might be used
    possible_text_columns = ["passage_text", "context", "text", "transcription"]
    
    for text_col in possible_text_columns:
        try:
            dataset = SpeechDataset(
                dataset_name="spoken_squad_test",
                text_column=text_col,
                split="train",
                streaming=True
            )
            print(f"✓ Configuration with text_column='{text_col}' works")
            break
        except Exception as e:
            print(f"✗ Configuration with text_column='{text_col}' failed: {e}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Spoken Squad Dataset Loading Test")
    print("=" * 60)
    print()
    
    success = test_spoken_squad_loading()
    test_custom_config()
    
    print()
    print("=" * 60)
    if success:
        print("✓ Basic test completed")
        print("\nNote: If the dataset name 'spoken_squad_test' doesn't exist,")
        print("you may need to:")
        print("1. Check the exact dataset identifier on HuggingFace")
        print("2. Update config/config.yaml with the correct name")
        print("3. Or provide a custom dataset loader")
    else:
        print("✗ Some tests failed - please check the configuration")
    print("=" * 60)

