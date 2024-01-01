
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, m):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 1024, 104, 104)
K = torch.randn(1, 1024, 104, 104)
V = torch.randn(1, 1024, 104, 104)
mask = (torch.rand(1, 104, 104) > 0.7).fill_(-1000000000.0)
