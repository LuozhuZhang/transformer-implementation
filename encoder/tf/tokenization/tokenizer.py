import re
from collections import Counter
import numpy as np
import torch

MAX_LEN = 128

class CustomTokenizer:
  def __init__(self, vocab=None, max_len=MAX_LEN):
    self.max_len = max_len
    if vocab is None:
      self.vocab = self.build_vocab()
    else:
      self.vocab = vocab
    self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
    self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

  def build_vocab(self):
    # Replace with an actual vocabulary build process, usually from the dataset
    words = ['[PAD]', '[UNK]'] + list(set(re.findall(r'\w+', ' '.join(data['text']))))
    return words

  def tokenize(self, text):
    # Simple tokenization
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

# Use custom tokenizer
custom_tokenizer = CustomTokenizer()

# Modify tokenize_data function
def tokenize_data(data, max_len=MAX_LEN):
  texts = data['text']
  labels = data['label']
  return custom_tokenizer.encode(texts, labels)
