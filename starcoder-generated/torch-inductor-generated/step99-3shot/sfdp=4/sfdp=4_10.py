
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q22, Q23, Q26, mask):
        Q15 = Q22
        Q16 = Q23
        Q19 = Q26
        qk = Q15 @ Q16.transpose(-2, -1) / math.sqrt(Q15.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ Q19
        return output
# Inputs to the model
K1 = torch.randn(1, 64, 64, 64)
K2 = torch.randn(1, 64, 64, 64)
K5 = torch.randn(1, 64, 64, 64)
mask = (torch.rand(1, 64, 64) > 0.7).fill_(-1000000000.0)
