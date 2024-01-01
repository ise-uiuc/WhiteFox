
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q16, K16, V, mask):
        qk = Q16 @ K16.transpose(-2, -1) / math.sqrt(Q16.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 3, 512)
K = torch.randn(1, 3, 512)
V = torch.randn(1, 512, 56)
mask = (torch.rand(1, 56) > 0.7).fill_(float(-100000000.0))
