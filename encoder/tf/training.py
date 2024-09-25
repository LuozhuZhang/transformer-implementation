import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
from encoder.tf.transformer.core import TransformerEncoder

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6
NUM_CLASSES = 2

# Load IMDB dataset
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

# Load the BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def tokenize_data(data, max_len=MAX_LEN):
    tokens = tokenizer(
        data['text'], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
    )
    return tokens['input_ids'], torch.tensor(data['label'])

# Create DataLoader
def create_dataloader(dataset, batch_size=BATCH_SIZE):
    inputs, labels = tokenize_data(dataset)
    data = torch.utils.data.TensorDataset(inputs, labels)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_data)
test_loader = create_dataloader(test_data)

# Set device and model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoder(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=N_HEADS, d_ff=D_FF, vocab_size=tokenizer.vocab_size)
model.to(device)

# Define loss function, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels = [x.to(device) for x in batch]
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Train and evaluate
train_model(model, train_loader, criterion, optimizer, num_epochs=3)
evaluate_model(model, test_loader)
