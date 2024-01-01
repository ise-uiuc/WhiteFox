
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=False)

    def forward(self, q, k, v, mask=None):
        q = self.linear(q).unsqueeze(-1)
        k = self.linear(k).unsqueeze(1)
        v = self.linear(v).unsqueeze(1)
        dot = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)
        weight = F.softmax(dot, dim=-1)
        return torch.matmul(weight, v).squeeze(1)

# Initializing the model
m = Model()

q = torch.randn(2, 8, 64)
k = torch.randn(2, 8, 64)
v = torch.randn(2, 8, 64)
mask = torch.triu(torch.ones(1, 2, 8), 0)
