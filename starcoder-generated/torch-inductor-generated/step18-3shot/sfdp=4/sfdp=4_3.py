
class Model(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, num_positions):
        super().__init__()
        self.wq = torch.nn.Linear(hidden_dim, hidden_dim // num_heads)
        self.wk = torch.nn.Linear(hidden_dim, hidden_dim // num_heads)
        self.wv = torch.nn.Linear(hidden_dim, hidden_dim // num_heads)

    def forward(self, query, key, value, attn_mask):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output.transpose(0, 1)

# Initializing the model.
m = Model(8, 1024, 1024)

# Inputs to the model
query = torch.randn(16, 8, 1024)
key = torch.randn(16, 8, 1024)
value = torch.randn(16, 8, 1024)
attention_mask = torch.zeros((16, 1, 1, 1024), dtype=torch.int64)
