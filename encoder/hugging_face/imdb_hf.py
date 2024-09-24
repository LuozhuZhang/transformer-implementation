from transformers import BertTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import time

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('Luozhu/nlp-sentiment-classification-model')
model.to(device)
model.eval()

# Custom reviews
custom_reviews = [
    "I loved this movie! It was fantastic.",
    "This was the worst film I have ever seen.",
    "It was okay, nothing special.",
    "Absolutely brilliant! I would watch it again.",
    "Not my cup of tea, unfortunately."
]

# Function to tokenize, truncate, and pad sequences
def tokenize_data(data, max_len=128):
    tokens = tokenizer(data, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']

# Create DataLoader
def create_dataloader(reviews, batch_size=1):
    inputs, masks = tokenize_data(reviews)
    data = TensorDataset(inputs, masks)
    return DataLoader(data, batch_size=batch_size, shuffle=False)

# Create DataLoader for custom reviews
custom_loader = create_dataloader(custom_reviews)

# Evaluate the model and time the evaluation
start_time = time.time()
results = []

with torch.no_grad():
    for batch in custom_loader:
        input_ids, attention_mask = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probabilities, 1)
        sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
        results.append(f'Review: "{custom_reviews[0]}" - Sentiment: {sentiment} - Probabilities: {probabilities.cpu().numpy()}')
        custom_reviews.pop(0)

# Calculate time
end_time = time.time()
time_taken = end_time - start_time

# Print results
for result in results:
    print(result)

# Save results to log file
with open('./local_data.log', 'w') as log_file:
    log_file.write(f"Evaluation Time: {time_taken:.2f} seconds\n")
    for result in results:
        log_file.write(result + '\n')

print(f'Results saved to ./local_data.log')
