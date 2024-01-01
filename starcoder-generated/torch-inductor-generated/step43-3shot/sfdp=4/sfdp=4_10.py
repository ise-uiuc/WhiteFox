
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q3, K3, v2, mask):
        qk = Q3 @ K3.transpose(-2, -1) / math.sqrt(Q3.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q = torch.randn(1, 2, 56, 56)
K = torch.randn(1, 2, 56, 56)
V = torch.randn(1, 2, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(float(-100000))
