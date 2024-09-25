import torch
import torch.nn as nn
import numpy as np

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

  def forward(self, x, mask=None):
    attn_out = self.attn(x)
    x = self.add_norm1(x, attn_out)
    ffn_out = self.ffn(x)
    return self.add_norm2(x, ffn_out)

class TransformerEncoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, num_classes):
    super().__init__()
    self.embedding = CustomEmbedding(vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model)
    self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
    self.fc = nn.Linear(d_model, num_classes)

  def forward(self, src, mask=None):
    src = self.embedding(src)
    src = self.positional_encoding(src)
    src = src.transpose(0, 1)
    for layer in self.layers:
      src = layer(src, mask)
    src = src.mean(dim=0)  # Global average pooling across the sequence length
    return self.fc(src)
