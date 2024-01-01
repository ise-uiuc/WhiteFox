
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q7, k, v0, mask):
        qk = q7 @ k.transpose(-2, -1) / math.sqrt(q7.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight - v0
        return output
# Inputs to the model
Q = torch.randn(1, 1, 3, 32)
K = torch.randn(1, 1, 2, 32)
V = torch.randn(1, 1, 2, 32)
mask = (torch.rand(1, 3, 3) > 0.7).fill_(-1000000000.0)
