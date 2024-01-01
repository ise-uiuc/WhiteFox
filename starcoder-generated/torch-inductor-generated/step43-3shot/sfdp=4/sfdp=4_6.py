


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, K2, V, mask):
        qk = x @ K2.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 1024, 56, 56)
K = torch.randn(1, 1024, 56, 56)
V = torch.randn(1, 1024, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-10000000.0)
# Inputs to the model
Q = torch.randn(1, 64, 1024)
K = torch.randn(1, 64, 1024)
V = torch.randn(1, 64, 1024)
mask = (torch.rand(1, 1024) > 0.7).fill_(float(-1000001))

