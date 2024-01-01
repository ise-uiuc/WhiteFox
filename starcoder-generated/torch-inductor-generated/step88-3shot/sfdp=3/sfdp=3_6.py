
class Model(torch.nn.Module):
    def __init__(self, hidden, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(data=query, requires_grad=False)
        self.key_transpose = torch.nn.Parameter(data=key.transpose(-2, -1), requires_grad=False)

    def forward(self, x3):
        v0 = scale_factor * torch.matmul(self.query, self.key_transpose)
        v1 = v0.softmax(dim=-1)
        v2 = torch.nn.functional.dropout(v1, p=dropout_p, training=True)
        v3 = torch.matmul(v2, value)
        v4 = v3.sum(dim=-2)
        return v4

# Initializing the model
hidden = 8
query = torch.randn(1, hidden)
key = torch.randn(8, 512, nhead * 8)
value = torch.randn(8, 512, nhead * 8)
scale_factor = 1 / math.sqrt(hidden)
dropout_p = 0.5
