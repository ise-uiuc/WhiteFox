
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x4, k3, v5, mask):
        qk = x4 @ k3.transpose(-2, -1) / math.sqrt(x4.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v5
        return output
# Inputs to the model
Q = torch.randn(1, 64, 11, 41)
K = torch.randn(1, 64, 11, 41)
V = torch.randn(1, 64, 11, 41)
mask = (torch.rand(1, 11, 41) > 0.7).fill_(-1000000000.0)
