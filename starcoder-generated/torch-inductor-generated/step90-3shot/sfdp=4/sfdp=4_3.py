
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q1, K5, V3, mask):
        qk = Q1 @ K5.transpose(-2, -1) / math.sqrt(Q1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V3
        return output
# Inputs to the model
Q1 = torch.randn(1, 64, 56, 56)
K12 = torch.randn(1, 64, 56, 56)
V7 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
