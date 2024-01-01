
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q1, k3, v3, mask):
        A = Q1 @ k3.transpose(-2, -1)
        B = A + mask
        output = torch.softmax(B, dim=-1) @ v3
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
