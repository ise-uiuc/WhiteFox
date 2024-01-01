
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k5, v2, mask):
        qk = q @ k5.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q34 = torch.randn(1, 64, 56, 56)
K0 = torch.randn(1, 64, 56, 56)
V7 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
