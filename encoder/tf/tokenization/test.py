import re
import numpy as np
import torch
from transformers import BertTokenizer

MAX_LEN = 128

class CustomTokenizer:
  def __init__(self, texts=None, vocab=None, max_len=MAX_LEN):
    self.max_len = max_len
    if vocab is None:
      self.vocab = self.build_vocab(texts)
    else:
      self.vocab = vocab
    self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
    self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

  def build_vocab(self, texts):
    if texts is None:
      return ['[PAD]', '[UNK]'] 
    words = ['[PAD]', '[UNK]'] + list(set(re.findall(r'\w+', ' '.join(texts))))
    return words

  def tokenize(self, text):
    tokens = re.findall(r'\w+', text.lower())
    return [self.word_to_id.get(token, self.word_to_id['[UNK]']) for token in tokens]

  def pad_sequence(self, seq):
    if len(seq) < self.max_len:
      seq += [self.word_to_id['[PAD]']] * (self.max_len - len(seq))
    return seq[:self.max_len]

  def encode(self, texts, labels):
    input_ids = []
    attention_masks = []
    for text in texts:
      tokens = self.tokenize(text)
      input_ids.append(self.pad_sequence(tokens))
      attention_masks.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)

# Sample texts for comparison
sample_texts = ["I loved this movie!", "This was a terrible movie."]
sample_labels = [1, 0]

# Using PyTorch's BertTokenizer
torch_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch_input_ids = torch_tokenizer(sample_texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
torch_ids = torch_input_ids['input_ids']
torch_masks = torch_input_ids['attention_mask']

# Using custom tokenizer
custom_tokenizer = CustomTokenizer(texts=sample_texts)
custom_input_ids, custom_attention_masks, _ = custom_tokenizer.encode(sample_texts, sample_labels)

# Print results
print("PyTorch Tokenizer Input IDs:")
print(torch_ids)

print("Custom Tokenizer Input IDs:")
print(custom_input_ids)

print("PyTorch Tokenizer Attention Masks:")
print(torch_masks)

print("Custom Tokenizer Attention Masks:")
print(custom_attention_masks)
