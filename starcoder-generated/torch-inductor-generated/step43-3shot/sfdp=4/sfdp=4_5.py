
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q4, K4, V4, mask):
        qk = Q4 @ K4.transpose(-2, -1) / math.sqrt(Q4.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V4
        return output
# Inputs to the model
Q = torch.randn(1, 2, 4, 4)
K = torch.randn(1, 2, 4, 4)
V = torch.randn(1, 2, 4, 4)
mask = (torch.rand(1, 4, 4) > 0.7).fill_(float(-100000))
