import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from imdb import TransformerEncoder  # Importing the model from imdb.py

# Hyperparameters
MAX_LEN = 128  # Max sentence length
BATCH_SIZE = 1  # Since we are evaluating only a few reviews

# Load the BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define custom reviews
custom_reviews = [
  "I loved this movie! It was fantastic.",
  "This was the worst film I have ever seen.",
  "It was okay, nothing special.",
  "Absolutely brilliant! I would watch it again.",
  "Not my cup of tea, unfortunately."
]

# Function to tokenize, truncate, and pad sequences
def tokenize_data(data, max_len=MAX_LEN):
  tokens = tokenizer(
    data, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
  )
  return tokens['input_ids'], tokens['attention_mask']

# Create a dataloader for custom reviews
def create_dataloader(reviews, batch_size=BATCH_SIZE):
  inputs, masks = tokenize_data(reviews)
  data = TensorDataset(inputs, masks)
  return DataLoader(data, batch_size=batch_size, shuffle=False)

# Create dataloader for custom reviews
custom_loader = create_dataloader(custom_reviews)

# Load the model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('./trained_transformer_encoder.pth', weights_only=True))  # Load your local model
model.to(device)
model.eval()

# Evaluate the model on custom reviews
results = []
with torch.no_grad():
  for batch in custom_loader:
    input_ids, attention_mask = [x.to(device) for x in batch]
    
    # Get model output
    outputs = model(input_ids)
    _, predicted = torch.max(outputs, 1)
    sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
    
    # Store the result
    results.append(f'Review: "{custom_reviews[0]}" - Sentiment: {sentiment}')
    custom_reviews.pop(0)  # Remove the processed review from the list

# Print the results
for result in results:
  print(result)
