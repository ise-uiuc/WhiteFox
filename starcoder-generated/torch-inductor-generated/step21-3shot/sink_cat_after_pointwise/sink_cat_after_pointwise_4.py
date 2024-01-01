
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        (A, B, C, D) = torch.chunk(x, 4, dim=-1)
        E = torch.relu(A + B + C + D)
        (a, b, c) = torch.chunk(E, 3, dim=1)
        F = a + b + c
        return F
# Inputs to the model
x = torch.randn(64, 128, 64)
y = torch.randn(64, 128, 32)
