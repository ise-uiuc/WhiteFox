
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qk, v0, mask):
        qK = qk @ v0.transpose(-2, -1) / math.sqrt(qk.size(-1))
        qK = qK + mask
        attn_weight = torch.softmax(qK, dim=-1)
        output = qK @ v0
        return output
# Inputs to the model
qk0 = torch.randn(1, 64, 192, 64)
v3 = torch.randn(1, 1, 1, 1)
mask = (torch.rand(1, 192, 64) > 0.7).fill_(-1000000000.0)
