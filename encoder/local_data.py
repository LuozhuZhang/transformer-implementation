import time
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from imdb import TransformerEncoder  # Importing the model from imdb.py

# Hyperparameters
MAX_LEN = 128  # Max sentence length
BATCH_SIZE = 32

# Load the BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize, truncate, and pad sequences
def tokenize_data(data, max_len=MAX_LEN):
  tokens = tokenizer(
    data, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
  )
  return tokens['input_ids'], tokens['attention_mask']

# Create dataloader for new data
def create_dataloader(dataset, batch_size=BATCH_SIZE):
  inputs, masks = tokenize_data(dataset)
  data = torch.utils.data.TensorDataset(inputs, masks)
  return DataLoader(data, batch_size=batch_size, shuffle=False)

# Load IMDB test data
dataset = load_dataset('imdb')
test_data = dataset['test']

# Create test dataloader
test_loader = create_dataloader([item['text'] for item in test_data['train']])  # Just an example, modify as needed

# Define custom reviews
custom_reviews = [
  "I loved this movie! It was fantastic.",
  "This was the worst film I have ever seen.",
  "It was okay, nothing special.",
  "Absolutely brilliant! I would watch it again.",
  "Not my cup of tea, unfortunately."
]

# Create dataloader for custom reviews
custom_loader = create_dataloader(custom_reviews)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('./trained_transformer_encoder.pth'))  # Load your local model
model.to(device)

# Step 6: Evaluation function
def evaluate_custom_reviews(model, custom_loader):
  model.eval()
  predictions = []
  with torch.no_grad():
    for batch in custom_loader:
      input_ids, attention_mask = [x.to(device) for x in batch]
      outputs = model(input_ids)
      _, predicted = torch.max(outputs, 1)
      predictions.extend(predicted.cpu().numpy())  # Store predictions

  # Interpret predictions (0 = negative, 1 = positive)
  for review, prediction in zip(custom_reviews, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'Review: "{review}" - Sentiment: {sentiment}')

# Evaluate the model on custom reviews
print("\nEvaluating on custom reviews:")
evaluate_custom_reviews(model, custom_loader)