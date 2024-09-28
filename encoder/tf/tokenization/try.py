from transformers import BertTokenizer
import pandas as pd

# Load the uncased BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "I love Ethereum and Transformer."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Map tokens to their real Token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Create a DataFrame to display the mapping
token_table = pd.DataFrame({'Token': tokens, 'Token ID': token_ids})

# Print the table
print(token_table)
