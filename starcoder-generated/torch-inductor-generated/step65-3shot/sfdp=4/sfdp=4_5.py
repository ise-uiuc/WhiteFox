
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, k, v, mask):
        qk = torch.nn.functional.softmax(Q @ k.transpose(-2, -1), dim=3)
        qk = qk + mask
        attn_weight = qk @ v
        return attn_weight
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
