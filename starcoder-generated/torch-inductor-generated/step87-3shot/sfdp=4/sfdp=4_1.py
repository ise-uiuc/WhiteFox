
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q2, K2, V, mask):
        qk = Q2 @ K2.transpose(-2, -1) / math.sqrt(Q2.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q4 = torch.randn(1, 64, 56, 56)
K5 = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
