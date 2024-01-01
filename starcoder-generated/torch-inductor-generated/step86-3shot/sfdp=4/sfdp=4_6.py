
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q0, K0, V9, mask):
        qk = Q0 @ K0.transpose(-2, -1) / math.sqrt(Q0.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V9
        return output
# Inputs to the model
Q2 = torch.randn(1, 64, 56, 56)
K2 = torch.randn(1, 64, 56, 56)
V9 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
