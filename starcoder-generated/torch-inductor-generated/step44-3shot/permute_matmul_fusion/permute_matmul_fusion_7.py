
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x2):
        v0 = x0.permute(0, 2, 1)
        v1 = x0.permute(0, 2, 1)
        v2 = x0.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        v4 = torch.matmul(v1, v0)
        v5 = torch.matmul(v0, v2)
        return torch.randint(1, 10, (3, 3))
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
