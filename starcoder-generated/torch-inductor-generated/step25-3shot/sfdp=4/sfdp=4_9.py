
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(m, n, bias=False)
        self.k = torch.nn.Linear(m, n, bias=False)
        self.v = torch.nn.Linear(n, n)
    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k).transpose(-2, -1)
        v = self.v(v)
        w = torch.matmul(q, k)
        w = w / math.sqrt(n)
        w = w + attn_mask
        w = torch.softmax(w, dim=-1)
        output = torch.matmul(w, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(batch, query_len, n)
k = torch.randn(batch, key_len, n)
v = torch.randn(batch, key_len, n)
