import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Hyperparameters
MAX_LEN = 128  # Max sentence length
BATCH_SIZE = 32
D_MODEL = 512  # Embedding dimension
N_HEADS = 8  # Number of attention heads
D_FF = 2048  # Feed-forward network hidden size
NUM_LAYERS = 6  # Number of encoder layers
NUM_CLASSES = 2  # Positive and negative classes

# Step 1: Load IMDB dataset
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

print(train_data)
print(test_data)