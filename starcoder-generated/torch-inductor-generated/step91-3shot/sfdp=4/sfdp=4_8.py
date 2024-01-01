
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q2, k4, V5, mask):
        qk = Q2 @ k4.transpose(-2, -1) / math.sqrt(Q2.size(-1))
        qk = qk + mask
        K = torch.softmax(qk, dim=-1)
        output = K @ V5
        return output
# Inputs to the model
Q2 = torch.randn(1, 64, 56, 56)
k4 = torch.randn(1, 64, 56, 56)
V5 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
