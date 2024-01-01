
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, ff):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, eps=eps)
        self.attn = torch.nn.MultiheadAttention(dim, num_heads, dropout=dropout_p, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(dim, eps=eps)
        self.proj = torch.nn.Linear(dim, ff)

    def forward(self, x):
        x1, _ = self.norm1(x)
        x2, _ = self.attn(x1, x1, x1)
        x3 = x1 + x2
        x4 = self.norm2(x3)
        v5 = self.proj(x4)
        return v5

# Initializing model
m = Model(dim, num_heads, ff)

# Inputs to the model
x = torch.randn(1, seq_length, dim)
