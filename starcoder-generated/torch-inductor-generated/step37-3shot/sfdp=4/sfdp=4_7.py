
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, m1, m2, mask):
        qk = x @ x.transpose(-2, -1)
        qk = qk + mask
        qk = qk + m1
        qk = qk + m2
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
