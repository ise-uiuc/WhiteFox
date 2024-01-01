
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q9, KK3, V4, mask):
        qk = Q9 @ KK3.transpose(-2, -1) / math.sqrt(Q9.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V4
        return output
# Inputs to the model
Q9 = torch.randn(1, 64, 56, 56)
KK3 = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
