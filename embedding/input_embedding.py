import numpy as np

# Token Embedding and Correct Positional Embedding
token_embeddings = {
  'i': [0.12, -0.34, 0.56, 0.78],
  'love': [0.22, 0.33, -0.11, 0.44],
  'eth': [-0.12, 0.56, -0.44, 0.33],
  '##ereum': [0.65, -0.34, 0.89, -0.12],
  'and': [0.14, -0.45, 0.33, -0.67],
  'trans': [0.22, 0.49, 0.55, -0.44],
  '##former': [-0.12, 0.56, 0.77, -0.11],
  '.': [0.44, 0.11, 0.33, -0.22]
}

positional_embeddings = {
  'i': [0.0, 1.0, 0.0, 1.0],
  'love': [0.84147102, 0.54030228, 0.00999983, 0.99994999],
  'eth': [0.090929741, -0.41614681, 0.01999867, 0.9980003],
  '##ereum': [0.14112, -0.9899925, 0.0299955, 0.9955004],
  'and': [-0.7568025, -0.65364361, 0.03998933, 0.99920011],
  'trans': [-0.95892429, 0.28366217, 0.04997917, 0.99875027],
  '##former': [-0.27941549, 0.96017027, 0.059964, 0.99820054],
  '.': [0.65698669, 0.75390226, 0.06994285, 0.99755102]
}

# Calculate Input Embeddings
input_embeddings = {}
for token in token_embeddings:
  input_embeddings[token] = np.add(token_embeddings[token], positional_embeddings[token])

# Output results
for token, embedding in input_embeddings.items():
  print(f"Input Embedding for '{token}': {embedding}")
