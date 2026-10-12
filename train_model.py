"""Train the optional Random Forest.  Usage: python train_model.py path/to/Training.csv"""
import sys

from src.predictor import train

if len(sys.argv) != 2:
    sys.exit(__doc__)
train(sys.argv[1])
