
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=4, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = heads

    def forward(self, x, mask=None):
        b, n, c = x.shape
        q = self.query(x).view(b, n, self.heads, -1).transpose(1, 2)
        k = self.key(x).view(b, n, self.heads, -1).transpose(1, 2)
        v = self.value(x).view(b, n, self.heads, -1).transpose(1, 2)

        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(k.size(-1))

        if mask is not None:
            qk = qk + mask

        a = self.dropout(torch.softmax(qk, -2))

        o = torch.matmul(a, v).transpose(1, 2)
        o = o.contiguous().view(b, n, -1)

        return self.norm(o)

# Initializing the model
m = SelfAttention(hidden_dim=32, input_dim=32)

# Inputs to the model
x = torch.randn(1, 32, 8)
mask = torch.randn(1, 1, 32)
