
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.v_head = dim // heads
        self.dropout = nn.Dropout(dropout)
        self.to_q_k = nn.Linear(dim, dim * 2, bias=False)
        self.to_dense = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.size()

        kv = self.to_q_k(x).chunk(2, dim=-1)
        k, v = kv[0].transpose(1, 2), kv[1] # (B, T, D*2) -> (B, D*2, T) -> (B, T, D), (B, T, D)
        scale_factor = 1 / (d * (d - 1) / 2) ** 0.5
        q = k @ v.transpose(-1, -2) * scale_factor # (B, C, T) -> (C, T) @ (T, D) -> (B, C, D)
        q = q.softmax(dim=-1) @ v # (B, C, D) @ (B, D, T) -> (B, C, T)
        q = q.transpose(0, 1).chunk(self.heads, dim=0)

        q = torch.stack(q, dim=1) # (1, B, C, T)
        q = q.transpose(1, 2).reshape(b, t, d) # (B, C, T) -> (B, 1, C, T) -> (B, T, C, T) -> (B, T, C)
        q = self.dropout(q)
        return self.to_dense(q)

# Initializing a single-head attention module
m = SelfAttention(dim=512, heads=1, dropout=0)

# Inputs to the model
x1 = torch.randn(32, 10, 512)
