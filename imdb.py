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

# Step 2: Load the BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize, truncate, and pad sequences
def tokenize_data(data, max_len=MAX_LEN):
    tokens = tokenizer(
        data['text'], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
    )
    return tokens['input_ids'], tokens['attention_mask'], torch.tensor(data['label'])

# Create dataloaders for train and test data
def create_dataloader(dataset, batch_size=BATCH_SIZE):
    inputs, masks, labels = tokenize_data(dataset)
    data = torch.utils.data.TensorDataset(inputs, masks, labels)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_data)
test_loader = create_dataloader(test_data)
