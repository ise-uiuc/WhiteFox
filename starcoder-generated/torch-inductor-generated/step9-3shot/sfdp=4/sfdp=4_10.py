
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = seq_len
 
  def forward(self, x1, x2, x3):
    qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1))
    qk = qk + x3
    attn_weight = torch.softmax(qk, dim=-1)
    output = attn_weight @ x4
    return output
 
# Initializing the model
m = Model(seq_len)

# Inputs to the model
x1 = torch.randn(batch_size, num_heads, seq_len, seq_len)
x2 = torch.randn(batch_size, num_heads, seq_len, seq_len)
x3 = torch.randn(batch_size, seq_len, seq_len)
x4 = torch.randn(batch_size, num_heads, seq_len, embedding_dim)
