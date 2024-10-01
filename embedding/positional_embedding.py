import numpy as np

def positional_encoding(pos, d_model=4):
  pos_enc = np.zeros(d_model)
  for i in range(d_model):
    angle_rate = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle = pos * angle_rate
    
    if i % 2 == 0:  # Even dimensions use sin
      pos_enc[i] = np.sin(angle)
    else:  # Odd dimensions use cos
      pos_enc[i] = np.cos(angle)
  
  return pos_enc

# Calculate positional encodings for pos from 0 to 7
d_model = 4  # Dimension of the model
for pos in range(8):  # Pos from 0 to 7
  embedding = positional_encoding(pos, d_model)
  print(f"Positional Embedding for pos {pos}: {embedding}")
