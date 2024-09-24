import torch
from encoder.imdb import TransformerEncoder
from transformers import BertTokenizer

MAX_LEN = 128  # Max sentence length
BATCH_SIZE = 32
D_MODEL = 512  # Embedding dimension
N_HEADS = 8  # Number of attention heads
D_FF = 2048  # Feed-forward network hidden size
NUM_LAYERS = 6  # Number of encoder layers
NUM_CLASSES = 2  # Positive and negative classes

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define your model architecture
model = TransformerEncoder(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=N_HEADS, d_ff=D_FF, vocab_size=tokenizer.vocab_size)

# Load your existing weights
model.load_state_dict(torch.load('trained_transformer_encoder.pth'))

# Save the model in the Hugging Face format
torch.save(model.state_dict(), 'pytorch_model.bin')
