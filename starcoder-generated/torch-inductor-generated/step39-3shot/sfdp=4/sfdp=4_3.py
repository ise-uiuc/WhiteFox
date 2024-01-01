
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q4, K2, mask, v7):
        qk = Q4 @ K2.transpose(-2, -1) / math.sqrt(Q4.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v7
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
Q4 = 3
K2 = 2
