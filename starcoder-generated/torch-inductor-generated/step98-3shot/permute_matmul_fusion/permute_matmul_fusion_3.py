
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v0 = x1.permute(0, 2, 1)
        v1 = x2.permute(0, 2, 1)
        v2 = x3.permute(0, 2, 1)
        v3 = torch.matmul(v0, v1)
        v4 = torch.matmul(v3, v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3, 2)
x3 = torch.randn(1, 3, 2)
