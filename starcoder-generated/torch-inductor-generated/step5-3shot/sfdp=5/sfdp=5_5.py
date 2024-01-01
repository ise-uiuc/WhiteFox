
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, d_k, d_v, n_heads, dropout_p=0.1):
      super().__init__()
      self.d_model = d_model
      self.d_k = d_k
      self.d_v = d_v
      self.n_heads = n_heads
      self.dropout_p = dropout_p

      self.d_k = d_k
      self.d_v = d_v

      self.W_Q = nn.Linear(d_model, n_heads * d_k)
      self.W_K = nn.Linear(d_model, n_heads * d_k)
      self.W_V = nn.Linear(d_model, n_heads * d_v)
      self.fc = nn.Linear(n_heads * d_v, d_model)
  
  def forward(self, qa, kb, attn_mask=None):
      qa = qa.view(-1, *qa.shape[2:])
      kb = kb.view(-1, *kb.shape[2:])
      qk = qa @ self.W_K.weight.T / math.sqrt(self.W_Q.weight.size(-1))
      if attn_mask is not None:
          qk = qk + attn_mask.unsqueeze(1)
      qk = torch.softmax(qk, dim=-1)
      qk = torch.dropout(qk, self.dropout_p, True)
      output = qk @ self.W_V.weight.T
      output = output.view(1, -1, *qa.shape[1:-1], self.n_heads * self.d_v).transpose(1, 4)
      output = output.reshape(1, -1, self.n_heads * self.d_v)
      output = self.fc(output)
      return output

class Transformer_encoder(nn.Module):
    def __init__(self, d_model=32, d_k=16, d_v=16, d_inner_hid=16, n_layers=2, n_heads=2, dropout_p=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder_layers.append(
                Transformer_Encoder_layer(d_model, d_inner_hid, n_heads, d_k, d_v, dropout_p))
    
    def forward(self, enc_inputs, non_pad_mask=None, slf_attn_mask=None):
        enc_self_attn_list = []
        for enc_layer in self.encoder_layers:
            enc_output, enc_self_attn = enc_layer(enc_inputs, non_pad_mask, slf_attn_mask)
            enc_self_attn_list += [enc_self_attn]

        return enc_output, enc_self_attn_list

class Transformer_Encoder_layer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_heads, d_k, d_v, dropout_p=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout_p=dropout_p)
        self.pos_ffn = Positionwise_FeedForward_layer(d_model, d_inner_hid, dropout_p=dropout_p)

    def forward(self, enc_inputs, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_self_attn = self.slf_attn(
            enc_inputs, enc_inputs, enc_inputs, slf_attn_mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_self_attn

class Positionwise_FeedForward_layer(nn.Module):
   def __init__(self, d_model, d_inner_hid, dropout_p=0.1, **kwargs):
      super().__init__()
      self.w_1 = nn.Conv1d(d_model, d_inner_hid, 5, **kwargs)
      self.w_2 = nn.Conv1d(d_inner_hid, d_model, 3, padding=1, **kwargs)
      self.dropout = nn.Dropout(p=dropout_p)
  
   def forward(self, x):
    # (B, F, D) -> (B, D, F)
    output = x.transpose(1, 2)
    # (B, D, F) -> (B, D, F)
    output = self.w_2(F.relu(self.w_1(output)))
    output = self.dropout(output.transpose(1, 2))
    output += x

    return output

# Initializing the model
enc_model = Transformer_encoder(d_model=32, d_k=16, d_v=16, d_inner_hid=16, n_layers=2, n_heads=2, dropout_p=0.1)

# Inputs to the model
enc_inputs = torch.rand([1, 32, 64])
