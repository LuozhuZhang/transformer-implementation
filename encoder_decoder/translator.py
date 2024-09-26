import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, emb_dim)
    self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)

  def forward(self, src):
    embedded = self.embedding(src)
    outputs, hidden = self.rnn(embedded)
    return hidden

class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
    super().__init__()
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)
    self.fc_out = nn.Linear(hidden_dim, output_dim)

  def forward(self, input, hidden):
    input = input.unsqueeze(1)
    embedded = self.embedding(input)
    output, hidden = self.rnn(embedded, hidden)
    prediction = self.fc_out(output)
    return prediction, hidden

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src, trg, trg_len):
    hidden = self.encoder(src)
    outputs = torch.zeros(trg.size(0), trg.size(1), self.decoder.fc_out.out_features)

    input = trg[0, :]  # Start token

    for t in range(1, trg_len):
      output, hidden = self.decoder(input, hidden)
      outputs[t] = output.squeeze(1)
      input = output.argmax(2)  # Get the predicted token

    return outputs

# Example parameters
INPUT_DIM = 1000  # Input vocabulary size
OUTPUT_DIM = 1000  # Output vocabulary size
EMB_DIM = 256  # Embedding dimension
HIDDEN_DIM = 512  # Hidden layer dimension
N_LAYERS = 2  # Number of layers

encoder = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS)
model = Seq2Seq(encoder, decoder)

# Assuming we have input and target sequences
src = torch.randint(0, INPUT_DIM, (32, 10))  # Example source sequence (batch_size, seq_len)
trg = torch.randint(0, OUTPUT_DIM, (32, 10))  # Example target sequence
trg_len = trg.size(1)

# Forward propagation
outputs = model(src, trg, trg_len)
