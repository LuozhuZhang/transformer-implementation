import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

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

# Custom Transformer components
class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, x):
        return self.embedding[x]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.fc(out)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)

class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, residual):
        return self.norm(x + residual)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = CustomMultiheadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.add_norm1(x, attn_out)
        ffn_out = self.ffn(x)
        return self.add_norm2(x, ffn_out)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size):
        super().__init__()
        self.embedding = CustomEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x.mean(dim=1))

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
