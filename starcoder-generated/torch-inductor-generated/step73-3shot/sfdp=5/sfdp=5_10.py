
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 128

        self.layers = nn.Sequential(
            nn.Linear(50, 100),  # 100
            nn.LayerNorm(128),
            # Self-attention
            nn.Linear(self.heads * self.dim * self.seq_len, self.heads * self.dim * self.seq_len),  # 6400
        )

    def forward(self, q, v, attn_mask):
        q, k = self.layers(q)
        k, v = self.layers(k)
        k += attn_mask
        return k, v

# Model begins
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer = torch.nn.TransformerEncoderLayer(768, 12, 512, 64)
  def forward(self, src, src_mask):
    return self.layer(src, src_mask)

# Model Begnins
class Model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=output_size
        )
        self.lstm_layer = torch.nn.LSTM(
            input_size=output_size, hidden_size=hidden_size, batch_first=True
        )
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.to(device)
      def forward(self, input_tensor, initial_hidden_state):
        embedding = self.embedding_layer(input_tensor).view(-1, 1, self.embedding_dim)
        lstm_out, _ = self.lstm_layer(
            embedding, initial_hidden_state
        )  # lstm_out: tensor of shape (batch_size, seq_length, hidden_size)
        logits = self.output_projection(lstm_out)  # logits: tensor of shape (batch_size, seq_length, vocab_size)
        return logits

    # Inputs to the model
input_tensor = torch.tensor([[5, 6, 7, 0, 0], [7, 6, 4, 4, 3], [2, 5, 0, 2, 3]])
initial_hidden_state = torch.zeros((3, 1, 512)) # (3, 1, 512)

# Model Ends

# Model Begins
class Model(torch.nn.Module):
  def __init__(self, device='cpu'):
    super().__init__()
    self.layer = torch.nn.TransformerEncoderLayer(768, 12, 512, 64)
    self.to(device)
    self.output_projection = torch.nn.Linear(768, 805) ##
  def forward(self, src, src_mask):
    x = self.layer(src, src_mask)
    output = self.output_projection(x)
    return output
    
# Model Ends