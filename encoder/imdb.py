import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

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

# Step 3: Define the Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Register pe as a buffer, so it moves with the model

    def forward(self, x):
        # Move positional encoding to the same device as input `x`
        pe = self.pe.to(x.device)
        return x + pe[:, :x.size(1), :]


# Transformer Encoder Layer with Multi-head Attention and Feed-Forward Network
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)  #! here is MHA implementation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # Self-attention layer
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + attn_output)
        # Feed-forward network
        ffn_output = self.ffn(src)
        return self.norm2(src + ffn_output)

# Transformer Encoder consisting of multiple encoder layers
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  #! inputs embedding
        self.positional_encoding = PositionalEncoding(d_model)  #! positional encoding
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])  #! MHA process
        self.fc = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, src, mask=None):
        # Embedding + positional encoding
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = src.transpose(0, 1)  # Change to [seq_len, batch_size, d_model]
        # Pass through the encoder layers
        for layer in self.layers:
            src = layer(src, mask)
        src = src.mean(dim=0)  # Global average pooling
        return self.fc(src)
    
# Step 4: Define loss function, optimizer and device
if torch.backends.mps.is_available():  # Check if MPS is available, set for Mac
  device = torch.device('mps')
elif torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
model = TransformerEncoder(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=N_HEADS, d_ff=D_FF, vocab_size=tokenizer.vocab_size)  # 6 layers + 512 d + 8 heads + 2048 FFN (hidden size) + 30522 vocab size
# model.load_state_dict(torch.load('./trained_transformer_encoder.pth'))  #! Re-use
model.to(device)
# print("Model loaded successfully!")  #! Re-use

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 5: Training function with time tracking and saving log
def train_model(model, train_loader, criterion, optimizer, num_epochs=3, save_path='./trained_transformer_encoder.pth', log_path='./imdb_record.txt'):
    model.train()
    training_records = []  # To store log information
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Print and record training info
        epoch_info = f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds'
        print(epoch_info)
        training_records.append(epoch_info)

    # Save the trained model's state dict after training
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Save training records to a file
    with open(log_path, 'w') as f:
        for record in training_records:
            f.write(record + '\n')
    print(f'Training records saved to {log_path}')

# Step 6: Evaluation function
def evaluate_model(model, test_loader, model_path='./trained_transformer_encoder.pth'):
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load the saved model state
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

# train_model(model, train_loader, criterion, optimizer, num_epochs=3, save_path='./trained_transformer_encoder.pth', log_path='./imdb_record.txt')
# evaluate_model(model, test_loader)
