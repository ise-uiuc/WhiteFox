
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q4, K0, v6, mask):
        qk = Q4 @ K0.transpose(-2, -1) / math.sqrt(Q4.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v6
        return output
# Inputs to the model
Q4 = torch.randn(1, 64, 56, 56)
K0 = torch.randn(1, 64, 56, 56)
v6 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
