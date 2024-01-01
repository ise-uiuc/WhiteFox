
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q9, K34, V78, mask):
        qk = Q9 @ K34.transpose(-2, -1) / math.sqrt(Q9.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V78
        return output
# Inputs to the model
Q9 = torch.randn(1, 64, 56, 56)
K34 = torch.randn(1, 6, 56, 56)
V78 = torch.randn(1, 6, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
