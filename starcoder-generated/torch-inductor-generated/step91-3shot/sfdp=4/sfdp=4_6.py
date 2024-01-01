
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, W22, K3, v3, mask):
        qk = W22 @ K3.transpose(-2, -1) / math.sqrt(W22.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        return output
# Inputs
W22 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
