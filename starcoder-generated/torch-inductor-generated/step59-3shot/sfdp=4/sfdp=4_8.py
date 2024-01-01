
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, k1, v2, mask):
        qk = x1 @ k1.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q = torch.randn(16, 64, 512, 512)
K = torch.randn(16, 64, 512, 512)
V = torch.randn(16, 64, 512, 512)
mask = (torch.rand(16, 512, 512) > 0.7).fill_(-1000000000.0)
