
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v3, mask9):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask9
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        return output
# Inputs to the model
Q = torch.randn(1, 64, 22, 22)
K2 = torch.randn(1, 64, 22, 22)
V = torch.randn(1, 64, 22, 22)
mask = (torch.rand(1, 22, 22) > 0.7).fill_(-1000000000.0)
