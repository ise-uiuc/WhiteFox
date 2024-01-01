
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.bmm(x1, x2)
        v4 = v1.permute(0, 2, 1)
        v5 = x2.permute(0, 2, 1)
        v6 = torch.bmm(v2, v4)
        return torch.matmul(x2, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
