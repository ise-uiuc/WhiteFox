
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q34, K1, V0, mask):
        qK = Q34 @ K1.transpose(-2, -1) / math.sqrt(Q34.size(-1))
        qK = qK + mask
        attn_weight = torch.softmax(qK, dim=-1)
        output = attn_weight @ V0
        return output
# Inputs to the model
QK = torch.randn(1, 64, 56, 56)
KV1 = torch.randn(1, 64, 56, 56)
Vlue300 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
