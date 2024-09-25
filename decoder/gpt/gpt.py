import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import numpy as np

# Hyperparameters
MAX_LEN = 128  # Max sentence length
BATCH_SIZE = 32
D_MODEL = 512  # Embedding dimension
N_HEADS = 8  # Number of attention heads
D_FF = 2048  # Feed-forward network hidden size
NUM_LAYERS = 6  # Number of decoder layers
VOCAB_SIZE = 50257  # GPT-2 vocabulary size

# Step 1: Load the dataset (e.g., a small text corpus)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
train_data = dataset['text']
train_data = [text for text in train_data if len(text.split()) < MAX_LEN]  # Filter long texts

# Step 2: Load the GPT-2 tokenizer for tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to tokenize and pad sequences
def tokenize_data(data, max_len=MAX_LEN):
  tokens = tokenizer(
    data, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
  )
  return tokens['input_ids']

# Create dataloader for train data
def create_dataloader(data, batch_size=BATCH_SIZE):
  inputs = tokenize_data(data)
  data = torch.utils.data.TensorDataset(inputs)
  return DataLoader(data, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_data)

# Step 3: Define the Positional Encoding for Transformer
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
    pe = self.pe.to(x.device)
    return x + pe[:, :x.size(1), :]

# Transformer Decoder Layer with Multi-head Attention and Feed-Forward Network
class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, num_heads)
    self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
    self.ffn = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.ReLU(),
      nn.Linear(d_ff, d_model)
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)

  def forward(self, x, enc_output, mask=None):
    attn_output, _ = self.self_attn(x, x, x)
    x = self.norm1(x + attn_output)
    attn_cross_output, _ = self.cross_attn(x, enc_output, enc_output)  # 交叉注意力
    x = self.norm2(x + attn_cross_output)
    ffn_output = self.ffn(x)
    return self.norm3(x + ffn_output)

# Transformer Decoder consisting of multiple decoder layers
class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model)
    self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
    self.fc = nn.Linear(d_model, vocab_size)

  def forward(self, x, enc_output, mask=None):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = x.transpose(0, 1)  # Change to [seq_len, batch_size, d_model]
    for layer in self.layers:
      x = layer(x, enc_output)
    return self.fc(x)

# Step 4: Define loss function, optimizer and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerDecoder(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=N_HEADS, d_ff=D_FF, vocab_size=VOCAB_SIZE)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 5: Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=3):
  model.train()
  for epoch in range(num_epochs):
    for batch in train_loader:
      input_ids = batch[0].to(device)
      optimizer.zero_grad()
      outputs = model(input_ids, input_ids)  # 使用自身作为输入
      loss = criterion(outputs.view(-1, VOCAB_SIZE), input_ids.view(-1))  # 计算损失
      loss.backward()
      optimizer.step()
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the model
train_model(model, train_loader, criterion, optimizer)

