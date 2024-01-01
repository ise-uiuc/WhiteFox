
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qk, v2, mask):
        qK = qk @ v2.transpose(-2, -1) / math.sqrt(qk.size(-1))
        qK = qK + mask
        attn_weight = torch.softmax(qK, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
qk1 = torch.randn(1, 64, 56, 56)
V3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
