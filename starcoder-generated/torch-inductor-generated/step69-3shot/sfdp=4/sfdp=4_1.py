
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(16, 1, 1, 256, 56, 56)
K = torch.randn(16, 1, 1, 256, 56, 56)
V = torch.randn(16, 1, 1, 256, 56, 56)
mask = (torch.rand(16, 1, 1, 1, 56, 56) > 0.7).fill_(-1000000000.0)
