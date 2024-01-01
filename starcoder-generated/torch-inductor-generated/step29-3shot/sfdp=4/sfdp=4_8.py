
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v2, mask):
        q = q.unsqueeze(-1)
        k = k.unsqueeze(-3)
        v2 = v2.unsqueeze(-3)
        qk = q @ k.transpose(-2, -1)
        qk = qk / math.sqrt(32)
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q = torch.randn(1, 64, 32)
K = torch.randn(1, 64, 32)
V = torch.randn(1, 64, 32)
mask = (torch.rand(1, 32) > 0.7).fill_(-1000000000.0)
