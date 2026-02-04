"""Script principal de treinamento."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.train_adapter import main

if __name__ == "__main__":
    main()

