
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q4, k, V5, mask):
        qk = Q4 @ k.transpose(-2, -1) / math.sqrt(Q4.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ V5
        return output
# Inputs to the model
Q2 = torch.randn(1, 64, 56, 56)
k5 = torch.randn(1, 64, 56, 56)
V3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
