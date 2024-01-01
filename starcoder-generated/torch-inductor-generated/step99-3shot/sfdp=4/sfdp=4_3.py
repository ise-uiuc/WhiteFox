
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1, x2, x3, x4):
        qk = x0 @ x1.transpose(-2, -1) / math.sqrt(x0.size(-1))
        qk = qk + x3
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x2
        return output
# Inputs to the model
Q = torch.randn(1, 2304, 7, 7)
K = torch.randn(1, 2304, 7, 7)
V = torch.randn(1, 2304, 7, 7)
mask = (torch.rand(1, 7, 7) > 0.7).fill_(-1000000000.0)
