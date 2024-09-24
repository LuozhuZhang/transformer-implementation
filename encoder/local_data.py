import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
from imdb import TransformerEncoder

# Hyperparameters
MAX_LEN = 128  # Max sentence length
BATCH_SIZE = 32
D_MODEL = 512  # Embedding dimension
N_HEADS = 8  # Number of attention heads
D_FF = 2048  # Feed-forward network hidden size
NUM_LAYERS = 6  # Number of encoder layers
NUM_CLASSES = 2  # Positive and negative classes

# Step 1: Load the BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 2: Define the Transformer Encoder and other necessary classes (not shown here for brevity)

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerEncoder(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=N_HEADS, d_ff=D_FF, vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('./trained_transformer_encoder.pth', map_location=device))
model.to(device)

# Step 3: Custom dataset containing reviews
custom_reviews = [
    "I absolutely loved this movie! It was fantastic.",
    "The plot was boring and the characters were dull.",
    "A remarkable film that I would recommend to everyone.",
    "It was a waste of time, I didn't enjoy it at all.",
    "Great acting and a wonderful story!",
    "I found it very tedious and hard to follow.",
    "One of the best films I've ever seen.",
    "Not my cup of tea, but I can see why others might like it.",
    "An outstanding achievement in cinema!",
    "I didn't like it, the pacing was too slow."
]

# Step 4: Tokenize the custom reviews
def tokenize_custom_data(reviews, max_len=MAX_LEN):
    tokens = tokenizer(
        reviews, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
    )
    return tokens['input_ids'], tokens['attention_mask']

# Step 5: Make predictions on the custom reviews
def predict_sentiments(model, reviews):
    model.eval()
    input_ids, attention_mask = tokenize_custom_data(reviews)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)
    
    # Display results
    for review, sentiment in zip(reviews, predicted):
        sentiment_label = "Positive" if sentiment.item() == 1 else "Negative"
        print(f"Review: {review}\nPredicted Sentiment: {sentiment_label}\n")

# Step 6: Call the prediction function
predict_sentiments(model, custom_reviews)
