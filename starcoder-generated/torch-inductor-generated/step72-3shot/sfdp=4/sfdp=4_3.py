
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k0, v, m0):
        qk = q @ k0.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m0
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
