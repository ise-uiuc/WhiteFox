
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = x.permute((2, 1, 0), (1, 0, 2))
        v2 = y.permute((0, 2, 1), (0, 2, 1))
        v3 = torch.matmul(v1, v2)
        return v3
# Inputs to the model
x = torch.randn(1, 2, 2)
y = torch.randn(1, 2, 2)
