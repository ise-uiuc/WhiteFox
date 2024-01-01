
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = v1.permute(0, 2, 1)
        v4 = v2.permute(0, 2, 1)
        v5 = torch.matmul(x2, v1)
        v6 = torch.matmul(x1, v2)
        return torch.matmul(v5, v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
