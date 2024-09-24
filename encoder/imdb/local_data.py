import time
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from imdb import TransformerEncoder

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 1

# Load the BERT tokenizer for tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define custom reviews
custom_reviews_easy = [
  "I loved this movie! It was fantastic.",
  "This was the worst film I have ever seen.",
  "It was okay, nothing special.",
  "Absolutely brilliant! I would watch it again.",
  "Not my cup of tea, unfortunately."
]

custom_reviews = [
  "Armies converge in an epic, final showdown that determines the fate of a king's dream for England - and Uhtred's hope for securing his family's future.",
  "As a huge battle with the Danes is close, Edward has to make pivotal decisions that will affect the fates of Uhtred, Wessex - and England.",
  "The future of England is at stake and the King refuses to hide. He is a man on a mission. Leaving the cover of the marshes, a united King Alfred and Uhtred must get word to Saxon forces; they make a rallying call to all Saxon armies to converge at Egbert's Stone to fight a decisive battle against the Danes. Treachery, deception and heartache follow, though hope remains. Wessex can still be saved, but at what cost to Uhtred?",
  "I'll keep this short. If you are into Marvel don't watch this show. If you are into comics then move onto something else. If you are into tight well written dialog, character development and story telling then this show is a big pass.The writing in this series, so far, is convoluted, amateurish and contradictory. I ended up not caring about the characters or the story line. The acting goes from boring to over the top which given the actors involved make me think with have a major issue with the director. Who is also the writer. Go figure. There were so many missed opportunities in this first episode to make a compelling and interesting story. Another waste of money that could have been spent on something of quality.",
  "There are lot of magical TV series/ movies which provides better pacing, thrill and a wonderful experience than Agatha all along. The first half of the first episode kept me thinking that it might be a good show. That marvel is actually back. But I am afraid to say that Marvel is still in its Mediocre era. After the first half of the 1st episode, it becomes very slow. And although I don't have much problem with that but atleast that slownessness should come with a reward. The plot is so boring that I had to watch it in parts. So, don't keep waiting for the new episodes. Watch it in 30 Oct if you want to binge and fast-forward some parts."
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
model.load_state_dict(torch.load('./trained_transformer_encoder.pth', weights_only=True))
model.to(device)
model.eval()

# Initialize timing
total_time_start = time.time()

# Time tokenization
tokenization_time_start = time.time()
custom_loader = create_dataloader(custom_reviews)  # Re-create dataloader for timing
tokenization_time_end = time.time()
tokenization_time = tokenization_time_end - tokenization_time_start

# Time the evaluation
evaluation_time_start = time.time()
results = []

with torch.no_grad():
  for batch in custom_loader:
    input_ids, attention_mask = [x.to(device) for x in batch]

    # Get model output
    outputs = model(input_ids)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted = torch.argmax(probabilities, 1)

    # Store the result
    sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
    results.append(f'Review: "{custom_reviews[0]}" - Sentiment: {sentiment} - Probabilities: {probabilities.cpu().numpy()}')
    custom_reviews.pop(0)

# Calculate time taken for evaluation
evaluation_time_end = time.time()
evaluation_time = evaluation_time_end - evaluation_time_start

# Calculate total time
total_time_end = time.time()
total_time = total_time_end - total_time_start

# Print the results
for result in results:
  print(result)

# Save results to a log file
with open('./local_data.txt', 'w') as log_file:
  log_file.write(f"Tokenization Time: {tokenization_time:.2f} seconds\n")
  log_file.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n")
  log_file.write(f"Total Evaluation Time: {total_time:.2f} seconds\n")
  for result in results:
    log_file.write(result + '\n')

print(f'Results saved to ./local_data.txt')
