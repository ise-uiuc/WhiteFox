
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, d0, d1, mask):
        d = d0 + d1
        attn_weight = torch.softmax(d @ d.transpose(-2, -1) / math.sqrt(d0.size(-1)) + mask, dim=-1)
        output = attn_weight @ d
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
