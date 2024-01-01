
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(0, 2, 1)
        v1 = x2.permute(0, 2, 1)
        v2 = v0.permute(0, 2, 1)
        v3 = v1.permute(0, 2, 1)
        v4 = torch.bmm(v0, v1)
        v5 = torch.bmm(v2, v3)
        return torch.matmul(v4, v5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
