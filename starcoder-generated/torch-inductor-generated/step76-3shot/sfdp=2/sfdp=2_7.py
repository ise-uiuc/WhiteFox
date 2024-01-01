
class Model(torch.nn.Module):
    def __init__(self, D_query, D_key, D_value, num_heads, dropout):
        super().__init__()
        self.query = torch.nn.Linear(D_query, D_key * num_heads * 3)
        self.key = torch.nn.Linear(D_key, D_key * num_heads)
        self.dropout_p = dropout
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        dk = q.shape[-1] // 3
        q, k = _split_heads(q, k, dk)
        scaled_qk = q @ k.transpose(-2, -1) / math.sqrt(dk)
        attn = scaled_qk.softmax(dim=-1)
        attn_out = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = attn_out @ val
        return out

# Initializing the model
m = Model(D_query=1024, D_key=1024, D_value=1024, num_heads=16, dropout=0.1)

# Inputs to the model
x1 = torch.randn(4, 1024, 196)
x2 = torch.randn(4, 1024, 384)
