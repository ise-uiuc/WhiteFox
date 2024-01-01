
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 1, 2)
        v2 = x2.permute(0, 1, 2)
        v3 = torch.matmul(x1, v1)
        v4 = torch.matmul(x2, v2)
        return torch.matmul(v3, v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
