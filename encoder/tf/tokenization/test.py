from transformers import BertTokenizer
from tokenizer import CustomTokenizer

MAX_LEN = 128

# Assume you have some sample texts
sample_texts = ["I loved this movie!", "This was a terrible movie."]

# Using PyTorch's BertTokenizer
torch_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch_input_ids = torch_tokenizer(sample_texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
torch_ids = torch_input_ids['input_ids']
torch_masks = torch_input_ids['attention_mask']

# Using custom tokenizer
custom_tokenizer = CustomTokenizer()
custom_input_ids, custom_attention_masks, _ = custom_tokenizer.encode(sample_texts, [0, 1])  # Assuming labels are 0 and 1

# Print results
print("PyTorch Tokenizer Input IDs:")
print(torch_ids)

print("Custom Tokenizer Input IDs:")
print(custom_input_ids)

print("PyTorch Tokenizer Attention Masks:")
print(torch_masks)

print("Custom Tokenizer Attention Masks:")
print(custom_attention_masks)
